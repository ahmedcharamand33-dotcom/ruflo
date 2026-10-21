import numpy as np
import pandas as pd

from trading_lab import readiness
from trading_lab.data import synthetic_prices
from trading_lab.paper_portfolio import step


def test_new_account_is_not_ready(tmp_path):
    close = synthetic_prices(n=1500, seed=4)["close"]
    spy = pd.DataFrame({"open": close, "close": close, "vix": 18.0})
    j = tmp_path / "spy.jsonl"
    for t in range(1400, 1430):
        step(pd.DataFrame({"SPY": close.iloc[:t + 1]}), j, "spy volatility control", band=0.1, max_dd=0.5)
    gates = dict((name, ok) for name, ok, _ in readiness.check(j, spy))
    assert not gates[f"At least {readiness.MIN_TRADING_DAYS} trading days of paper results"]
    assert gates[f"Bot ran on at least {readiness.MIN_COVERAGE:.0%} of trading days"]
    assert "NOT READY" in readiness.render(readiness.check(j, spy))
