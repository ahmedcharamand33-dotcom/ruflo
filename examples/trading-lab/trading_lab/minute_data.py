"""SPY one-minute bars from Alpaca (consolidated SIP feed, split/dividend adjusted).

Only regular-session bars (09:30-15:59 New York time) are kept. One gzip CSV
per year is cached under data/minute/, so later runs only fetch what is new.

    python -m trading_lab minute-data
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from .data import CACHE_DIR

URL = "https://data.alpaca.markets/v2/stocks/{sym}/bars"
FIRST_YEAR = 2016
NY = "America/New_York"


def _headers() -> dict:
    pairs = (("APCA-API-KEY-ID", "APCA_API_KEY_ID"), ("APCA-API-SECRET-KEY", "APCA_API_SECRET_KEY"))
    return {h: os.environ[v] for h, v in pairs if v in os.environ}


def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    import requests

    rows, token = [], None
    while True:
        params = {"timeframe": "1Min", "start": start, "end": end, "feed": "sip",
                  "adjustment": "all", "limit": 10000}
        if token:
            params["page_token"] = token
        for attempt in range(5):
            r = requests.get(URL.format(sym=symbol), headers=_headers(), params=params, timeout=60)
            if r.status_code != 429:
                break
            time.sleep(2 ** attempt)
        r.raise_for_status()
        body = r.json()
        rows += body.get("bars") or []
        token = body.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.t).dt.tz_convert(NY).dt.tz_localize(None)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["open", "high", "low", "close", "volume"]]
    minutes = df.index.hour * 60 + df.index.minute
    return df[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)]


def load(symbol: str = "SPY", refresh_current_year: bool = False) -> pd.DataFrame:
    folder = CACHE_DIR / "minute"
    folder.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=NY)
    frames = []
    for year in range(FIRST_YEAR, now.year + 1):
        f = folder / f"{symbol}-{year}.csv.gz"
        current = year == now.year
        if not f.exists() or (current and refresh_current_year):
            # the free data plan refuses the most recent 15 minutes of consolidated data
            end = ((now - pd.Timedelta(minutes=20)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                   if current else f"{year + 1}-01-01")
            fetch(symbol, f"{year}-01-01", end).to_csv(f, index_label="time")
        frames.append(pd.read_csv(f, index_col="time", parse_dates=True))
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    today_open = df.index.normalize() == now.normalize().tz_localize(None)
    if (now.hour, now.minute) < (16, 15):  # drop an unfinished session
        df = df[~today_open]
    return df


def summary(path: Path | None = None) -> str:
    df = load(refresh_current_year=True)
    days = df.index.normalize().nunique()
    return f"SPY minute bars: {len(df):,} bars, {days:,} sessions, {df.index[0]} to {df.index[-1]}"
