import numpy as np
import pandas as pd
import pytest

from trading_lab.data import bond_total_return, synthetic_prices
from trading_lab.metrics import consistency
from trading_lab.portfolio import candidates, current_allocation, run_portfolio, with_trend, equal_weight


@pytest.fixture(scope="module")
def closes():
    cols = {name: synthetic_prices(n=600, seed=s, bars_per_year=12)["close"]
            for s, name in enumerate(("stocks", "bonds", "gold"))}
    return pd.DataFrame(cols)


@pytest.mark.parametrize("name", list(candidates()))
def test_rules_never_lever_and_never_look_ahead(closes, name):
    rule = candidates()[name]
    full = rule(closes, 12)
    assert (full.sum(axis=1) <= 1 + 1e-9).all() and (full >= 0).all().all()
    pd.testing.assert_frame_equal(full.iloc[:400], rule(closes.iloc[:400], 12))


def test_leverage_is_rejected(closes):
    w = pd.DataFrame(0.5, index=closes.index, columns=closes.columns)  # 150% gross
    with pytest.raises(ValueError, match="leverage"):
        run_portfolio(closes, w)


def test_portfolio_return_is_weighted_sum():
    closes = pd.DataFrame({"a": [100.0, 110.0, 121.0], "b": [100.0, 100.0, 90.0]})
    w = pd.DataFrame({"a": [0.5] * 3, "b": [0.5] * 3})
    res = run_portfolio(closes, w, cost_bps=0, slippage_bps=0)
    assert res.returns.tolist() == pytest.approx([0.0, 0.05, 0.0])


def test_trend_filter_moves_to_cash_in_downtrend():
    falling = pd.DataFrame({"a": np.linspace(200, 100, 30), "b": np.linspace(100, 200, 30)},
                           index=pd.date_range("2000-01-31", periods=30, freq="ME"))
    alloc = current_allocation(falling, with_trend(equal_weight), 12)
    assert alloc["a"] == 0 and alloc["b"] == 0.5 and alloc["cash"] == 0.5


def test_bond_rebuild_earns_the_yield_when_rates_are_flat():
    tr = bond_total_return(pd.Series([5.0] * 13))
    assert tr.iloc[-1] == pytest.approx((1 + 0.05 / 12) ** 12, rel=1e-9)  # coupons reinvested


def test_bond_loses_when_rates_jump():
    tr = bond_total_return(pd.Series([3.0, 4.0]))
    assert tr.iloc[-1] - 1 == pytest.approx(-0.08, abs=0.01)  # ~8 years duration x 1%


def test_consistency_metrics():
    idx = pd.date_range("2000-01-31", periods=36, freq="ME")
    rets = pd.Series([0.01] * 12 + [-0.02] * 12 + [0.01] * 12, index=idx)
    c = consistency(rets, 12)
    assert c["pos_years"] == pytest.approx(2 / 3)
    assert c["worst_year"] == pytest.approx(0.98**12 - 1)
