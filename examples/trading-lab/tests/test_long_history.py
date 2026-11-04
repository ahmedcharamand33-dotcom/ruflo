import numpy as np
import pandas as pd

from trading_lab import long_history as lh


def test_rules_step_aside_in_a_long_slide_and_cash_earns_interest():
    idx = pd.bdate_range("1990-01-01", periods=900)
    up = np.linspace(100, 200, 600)
    down = np.linspace(200, 80, 300)
    df = pd.DataFrame({"close": np.concatenate([up, down]), "cash": 0.0001}, index=idx)
    r = lh.runs(df)
    slide = slice(idx[650], None)
    hold = (1 + r["buy & hold S&P 500"].loc[slide]).prod()
    trend = (1 + r["trend: above 200-day average"].loc[slide]).prod()
    assert trend > hold  # the trend rule is in cash for most of the slide
    assert r["trend: above 200-day average"].iloc[100] > 0  # flat start: in cash, earning interest
