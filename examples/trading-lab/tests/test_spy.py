import numpy as np
import pandas as pd
import pytest

from trading_lab import spy_research as sr
from trading_lab.data import synthetic_prices
from trading_lab.paper import read_events
from trading_lab.paper_portfolio import report, step


@pytest.fixture(scope="module")
def df():
    close = synthetic_prices(n=1500, seed=21)["close"]
    rng = np.random.default_rng(0)
    return pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.002, len(close))),
                         "close": close, "vix": 18.0}, index=close.index)


@pytest.mark.parametrize("name", list(sr.HYPOTHESES))
def test_hypotheses_do_not_look_ahead(df, name):
    fn = sr.HYPOTHESES[name]
    if name == "turn of the month":
        pytest.skip("uses the known trading calendar, not future prices")
    full, cut = fn(df), 1000
    pd.testing.assert_series_equal(full.iloc[:cut], fn(df.iloc[:cut]), check_names=False)


def test_turn_of_month_holds_four_days_a_month(df):
    held = sr.turn_of_month(df).shift(1).fillna(0)
    per_month = held.groupby(df.index.to_period("M")).sum().iloc[1:-1]
    assert per_month.between(3, 5).all()


def test_vol_control_never_levers(df):
    w = sr.vol_target(df)
    assert (w <= 1.0 + 1e-12).all() and (w >= 0).all()


def test_evaluate_reports_every_hypothesis(df):
    table = sr.evaluate(df)
    rules = set(table.rule)
    assert set(sr.HYPOTHESES) <= rules and "overnight only" in rules and "buy & hold SPY" in rules


def test_band_rebalancing_trades_only_on_drift(df, tmp_path):
    closes = pd.DataFrame({"SPY": df.close})
    j = tmp_path / "spy.jsonl"
    for t in range(1200, 1300):
        step(closes.iloc[:t + 1], j, "spy volatility control", cash=100_000, band=0.10, max_dd=0.5)
    rebalances = [e for e in read_events(j) if e["type"] == "rebalance"]
    assert 1 <= len(rebalances) < 100
    rep = report(j)
    assert rep["cash_weight"] >= -1e-9 and rep["days"] == 100


def test_save_intraday_appends_without_duplicates(tmp_path, monkeypatch):
    idx = pd.date_range("2026-10-19 09:30", periods=6, freq="5min", tz="America/New_York")
    bars = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 10}, index=idx)
    monkeypatch.setattr(sr, "_yahoo", lambda *a, **k: bars)
    after_close = pd.Timestamp("2026-10-19 17:00", tz="America/New_York")
    assert "6 new" in sr.save_intraday(tmp_path, after_close)
    assert "0 new" in sr.save_intraday(tmp_path, after_close)
    assert len(pd.read_csv(tmp_path / "2026-10.csv")) == 6


def test_save_intraday_skips_unfinished_session(tmp_path, monkeypatch):
    idx = pd.date_range("2026-10-19 09:30", periods=6, freq="5min", tz="America/New_York")
    bars = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 10}, index=idx)
    monkeypatch.setattr(sr, "_yahoo", lambda *a, **k: bars)
    mid_session = pd.Timestamp("2026-10-19 10:30", tz="America/New_York")
    assert "0 new" in sr.save_intraday(tmp_path, mid_session)
