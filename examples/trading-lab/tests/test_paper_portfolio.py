import numpy as np
import pandas as pd
import pytest

from trading_lab.data import synthetic_prices
from trading_lab.paper import read_events
from trading_lab.paper_portfolio import replay, report, resume, step


@pytest.fixture
def closes():
    cols = {n: synthetic_prices(n=800, seed=s)["close"] for s, n in enumerate(("SPY", "IEF", "GLD"))}
    return pd.DataFrame(cols)


def run(closes, journal, start, end, **kw):
    return [step(closes.iloc[:t + 1], journal, **kw) for t in range(start, end)]


def test_daily_runs_rebalance_monthly_and_never_borrow(closes, tmp_path):
    j = tmp_path / "p.jsonl"
    run(closes, j, 550, 800)
    events = read_events(j)
    months = {str(closes.index[t].date())[:7] for t in range(550, 800)}
    assert len([e for e in events if e["type"] == "rebalance"]) == len(months)
    book = replay(events)
    for m in book.marks:
        assert m["cash"] >= -1e-6, "cash went negative: leverage"
    assert report(j)["days"] == 250


def test_same_day_is_a_no_op(closes, tmp_path):
    j = tmp_path / "p.jsonl"
    step(closes, j)
    n = len(read_events(j))
    assert "already processed" in step(closes, j)
    assert len(read_events(j)) == n


def test_circuit_breaker_goes_to_cash_and_stays_halted(tmp_path):
    idx = pd.bdate_range("2020-01-01", periods=400)
    up = np.linspace(100, 200, 300)
    crash = np.linspace(200, 100, 100)
    closes = pd.DataFrame({"SPY": np.r_[up, crash], "IEF": np.r_[up, crash], "GLD": np.r_[up, crash]},
                          index=idx)
    j = tmp_path / "p.jsonl"
    out = run(closes, j, 250, 400, max_dd=0.10)
    assert any("CIRCUIT BREAKER" in line for line in out)
    rep = report(j)
    assert rep["status"].startswith("HALTED")
    assert rep["holdings"] == {} and rep["cash_weight"] == pytest.approx(1.0)
    assert rep["max_drawdown"] > -0.15  # stopped near the limit, not at the bottom
    resume(j, "reviewed")
    assert replay(read_events(j)).halted is None


def test_idle_cash_accrues_interest_per_trading_day(tmp_path):
    import pandas as pd
    from trading_lab import paper_portfolio as pp
    j = tmp_path / "j.jsonl"
    idx = pd.date_range("2026-01-02", periods=300, freq="B")
    closes = pd.DataFrame({"SPY": 100.0}, index=idx)
    pp.append_event(j, {"type": "open", "cash": 1000.0, "rule": "x"})
    pp.append_event(j, {"type": "mark", "date": "2026-10-19", "equity": 1000.0, "cash": 1000.0, "prices": {"SPY": 100.0}})
    pp.append_event(j, {"type": "halt", "date": "2026-10-19", "reason": "test"})
    day = closes.loc[:"2026-10-21"]
    pp.step(day, j, cash_yield=0.0504)
    interest = [e for e in pp.read_events(j) if e["type"] == "interest"]
    assert interest and interest[0]["amount"] == 0.4  # 1000 * 5.04% / 252 * 2 trading days
