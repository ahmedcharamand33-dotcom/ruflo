import pandas as pd
import pytest

from trading_lab.data import synthetic_prices
from trading_lab.paper import read_events, replay, review, step
from trading_lab.strategies import default_candidates


@pytest.fixture
def close():
    return synthetic_prices(n=1600, seed=3)["close"]


def run_days(close, journal, start, end):
    for t in range(start, end):
        step(close.iloc[:t + 1], journal, default_candidates(), train_years=5)


def test_step_is_idempotent(close, tmp_path):
    journal = tmp_path / "j.jsonl"
    step(close, journal, default_candidates(), train_years=5)
    n = len(read_events(journal))
    assert "already processed" in step(close, journal, default_candidates(), train_years=5)
    assert len(read_events(journal)) == n


def test_replay_rebuilds_account_and_tracks_target(close, tmp_path):
    journal = tmp_path / "j.jsonl"
    run_days(close, journal, 1400, 1600)
    acct = replay(read_events(journal))
    assert acct.last_date == str(close.index[-1].date())
    last = acct.marks[-1]
    assert acct.equity(last["price"]) == pytest.approx(last["equity"])
    assert 0.0 <= last["exposure"] <= 1.0 + 1e-9
    # selection refreshes about once per test window, not every bar
    selects = [e for e in read_events(journal) if e["type"] == "select"]
    assert 1 <= len(selects) <= 2


def test_review_halts_when_live_lags_backtest(close, tmp_path):
    journal = tmp_path / "j.jsonl"
    run_days(close, journal, 1500, 1560)
    assert review(journal, pd.Series([0.0, 0.0]), min_bars=1000)["status"] == "WARMING UP"
    optimistic = pd.Series([0.01, 0.012, 0.008] * 50)  # backtest promised ~1%/day
    assert review(journal, optimistic, min_bars=20)["status"] == "HALT"
