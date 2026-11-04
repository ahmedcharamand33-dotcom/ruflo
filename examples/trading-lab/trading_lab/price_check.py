"""Cross-check Yahoo's SPY closes against Alpaca's consolidated (SIP) closes.

Yahoo is free and unofficial; if it ever serves a wrong price the bot would
trade on it. Every evening the last ten sessions are compared and new dates
are appended to a log. A gap above ``MAX_GAP_BPS`` marks the day as bad.

Credentials: in the cloud sandbox the network proxy adds Alpaca's headers;
elsewhere ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` are read if set.

    python -m trading_lab price-check
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .data import drop_unfinished_session

MAX_GAP_BPS = 5.0
LOG = Path("paper/price_check.csv")
BARS_URL = "https://data.alpaca.markets/v2/stocks/{sym}/bars"


def alpaca_closes(symbol: str, start: str) -> pd.Series:
    import requests

    headers = {h: os.environ[v] for h, v in (("APCA-API-KEY-ID", "APCA_API_KEY_ID"),
                                             ("APCA-API-SECRET-KEY", "APCA_API_SECRET_KEY")) if v in os.environ}
    r = requests.get(BARS_URL.format(sym=symbol), headers=headers, timeout=20,
                     params={"timeframe": "1Day", "start": start, "feed": "sip", "adjustment": "raw"})
    r.raise_for_status()
    bars = r.json().get("bars") or []
    return pd.Series({pd.Timestamp(b["t"][:10]): float(b["c"]) for b in bars}, dtype=float)


def yahoo_closes(symbol: str) -> pd.Series:
    from .spy_research import _yahoo

    d = _yahoo(symbol, period="15d")
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d["Close"]


def compare(yahoo: pd.Series, alpaca: pd.Series, max_gap_bps: float = MAX_GAP_BPS) -> pd.DataFrame:
    both = pd.DataFrame({"yahoo": yahoo, "alpaca": alpaca}).dropna()
    both["gap_bps"] = ((both.yahoo / both.alpaca - 1.0) * 1e4).round(2)
    both["ok"] = both.gap_bps.abs() <= max_gap_bps
    both.index = both.index.strftime("%Y-%m-%d")
    both.index.name = "date"
    return both.round({"yahoo": 4, "alpaca": 4})


def run(path: Path = LOG, symbol: str = "SPY") -> str:
    yahoo = drop_unfinished_session(yahoo_closes(symbol).to_frame("close")).close
    table = compare(yahoo, alpaca_closes(symbol, str(yahoo.index[0].date())))
    if path.exists():
        old = pd.read_csv(path, index_col="date")
        table = pd.concat([old, table[~table.index.isin(old.index)]]).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path)
    last = table.iloc[-1]
    bad = table[~table.ok.astype(bool)]
    status = "AGREE" if bool(last.ok) else "MISMATCH: do not trade on today's Yahoo price"
    return (f"price-check {table.index[-1]}: Yahoo {last.yahoo} vs Alpaca {last.alpaca} "
            f"({last.gap_bps:+.1f} bp) {status}; {len(table)} days logged, {len(bad)} mismatches")
