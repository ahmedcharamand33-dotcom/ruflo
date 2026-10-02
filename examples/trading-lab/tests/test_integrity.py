"""Tests that guard against the classic ways a backtest lies."""
import pandas as pd
import pytest

from trading_lab.backtest import extract_trades, run_backtest
from trading_lab.data import synthetic_prices
from trading_lab.metrics import deflated_sharpe, probabilistic_sharpe, summary
from trading_lab.scorecard import scorecard
from trading_lab.strategies import BuyAndHold, default_candidates
from trading_lab.walkforward import walk_forward


@pytest.fixture(scope="module")
def close():
    return synthetic_prices(n=3000, seed=7)["close"]


@pytest.mark.parametrize("strategy", default_candidates(), ids=lambda s: s.label)
def test_strategies_do_not_look_ahead(close, strategy):
    """Positions computed on a truncated history must match the full-history positions."""
    full = strategy.positions(close, 252)
    cut = 2000
    partial = strategy.positions(close.iloc[:cut], 252)
    pd.testing.assert_series_equal(full.iloc[:cut], partial, check_names=False)


def test_lag_zero_is_rejected(close):
    with pytest.raises(ValueError):
        run_backtest(close, pd.Series(1.0, index=close.index), execution_lag=0)


def test_signal_is_held_after_lag():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    target = pd.Series([0.0, 1.0, 1.0, 1.0])
    res = run_backtest(close, target, cost_bps=0, slippage_bps=0, execution_lag=1)
    # bought at the close of bar 1, so the bar 1 move (+10%) is not earned
    assert res.returns.tolist() == pytest.approx([0.0, 0.0, 0.10, 0.10])


def test_costs_charged_on_every_change():
    close = pd.Series([100.0] * 6)
    target = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    res = run_backtest(close, target, cost_bps=5, slippage_bps=5)
    assert res.costs.sum() == pytest.approx(4 * 10 / 1e4)


def test_trade_extraction_round_trips():
    idx = pd.date_range("2020-01-01", periods=6)
    held = pd.Series([0, 1, 1, 0, -1, -1], index=idx, dtype=float)
    net = pd.Series([0, 0.1, 0.1, 0, 0.05, 0.05], index=idx)
    trades = extract_trades(held, net)
    assert [(t.side, t.bars) for t in trades] == [(1.0, 2), (-1.0, 2)]
    assert trades[0].ret == pytest.approx(0.21)


def test_buy_and_hold_summary_matches_asset(close):
    res = run_backtest(close, BuyAndHold().positions(close, 252), cost_bps=0, slippage_bps=0)
    assert res.equity.iloc[-1] == pytest.approx(close.iloc[-1] / close.iloc[0], rel=1e-9)
    assert summary(res.returns, 252)["max_dd"] <= 0.0


def test_deflation_only_ever_lowers_confidence(close):
    rets = close.pct_change().dropna()
    assert deflated_sharpe(rets, 50) < deflated_sharpe(rets, 2) < probabilistic_sharpe(rets)


def test_walk_forward_does_not_use_future_data(close):
    """Appending future bars must not change any earlier out-of-sample decision."""
    cands = default_candidates()
    short = walk_forward(close.iloc[:2600], cands, train_years=5)
    long = walk_forward(close, cands, train_years=5)
    common = short.strategy.held.index
    pd.testing.assert_series_equal(short.strategy.held, long.strategy.held.loc[common])
    assert [f.chosen for f in short.folds[:-1]] == [f.chosen for f in long.folds[:len(short.folds) - 1]]


@pytest.mark.parametrize("seed", range(5))
def test_no_edge_found_in_random_walks(seed):
    """The scorecard must not certify an edge on data where none exists."""
    close = synthetic_prices(seed=100 + seed)["close"]
    card = scorecard(walk_forward(close, default_candidates()))
    assert card["verdict"] == "NO EDGE"
