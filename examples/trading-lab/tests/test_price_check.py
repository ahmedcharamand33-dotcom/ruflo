import pandas as pd

from trading_lab.price_check import compare


def test_flags_only_gaps_above_threshold():
    idx = pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21"])
    yahoo = pd.Series([766.48, 769.05, 780.00], index=idx)
    alpaca = pd.Series([766.48, 769.10, 769.00], index=idx)
    t = compare(yahoo, alpaca, max_gap_bps=5)
    assert list(t.ok) == [True, True, False]
    assert t.loc["2026-10-21", "gap_bps"] > 100


def test_dates_missing_from_either_source_are_skipped():
    yahoo = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-19", "2026-10-20"]))
    alpaca = pd.Series([1.0], index=pd.to_datetime(["2026-10-19"]))
    assert list(compare(yahoo, alpaca).index) == ["2026-10-19"]
