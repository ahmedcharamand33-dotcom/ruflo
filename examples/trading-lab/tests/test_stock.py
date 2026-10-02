import pandas as pd
import pytest

from trading_lab import data
from trading_lab.data import synthetic_prices
from trading_lab.paper_portfolio import report, step

STOOQ_CSV = """Date,Open,High,Low,Close,Volume
2026-09-28,500.1,505.0,498.2,503.4,21000000
2026-09-29,503.0,510.3,502.1,509.9,19000000
2026-09-30,509.5,511.0,500.0,501.2,25000000
"""


def test_parse_stooq():
    df = data.parse_stooq(STOOQ_CSV)
    assert list(df["close"]) == [503.4, 509.9, 501.2]
    assert str(df.index[-1].date()) == "2026-09-30"


def test_parse_stooq_rejects_non_table():
    with pytest.raises(RuntimeError, match="no price table"):
        data.parse_stooq("No data")


def test_load_stock_reports_both_sources_when_blocked(monkeypatch):
    def blocked(_):
        raise ConnectionError("egress blocked")
    monkeypatch.setattr(data, "load_stooq", blocked)
    monkeypatch.setattr(data, "load_yahoo", blocked)
    with pytest.raises(RuntimeError, match="no price source reachable for MSFT"):
        data.load_stock("MSFT")


def test_single_stock_paper_account(tmp_path):
    closes = pd.DataFrame({"MSFT": synthetic_prices(n=700, seed=11)["close"]})
    j = tmp_path / "msft.jsonl"
    for t in range(600, 700):
        step(closes.iloc[:t + 1], j, cash=10_000.0, max_dd=0.25)
    rep = report(j)
    assert rep["benchmark_name"] == "MSFT buy & hold"
    assert rep["cash_weight"] >= -1e-9
    assert set(rep["holdings"]) <= {"MSFT"}
