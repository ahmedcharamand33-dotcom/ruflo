"""Price data providers.

Every loader returns a DataFrame indexed by date (ascending, unique) with at
least a ``close`` column. ``close`` should be a *total-return* price where
possible (adjusted for splits and dividends), otherwise backtests understate
buy-and-hold and overstate anything that sits in cash.

Sources (``load(spec)``):
    shiller            Monthly S&P 500 total-return index, 1871-today (GitHub).
    csv:<path>         Any CSV with a date column and close / adj close.
    yahoo:<TICKER>     Daily bars via the optional ``yfinance`` package.
    synthetic[:seed]   Random-walk prices for tests (no edge exists by design).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    if len(df) < 2:
        raise ValueError("need at least 2 valid price rows")
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    date_col = "date" if "date" in raw.columns else raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    df = raw.set_index(date_col)
    df.index.name = "date"
    for name in ("adj close", "adj_close", "adjclose"):
        if name in df.columns:
            df["close"] = df[name]
            break
    if "close" not in df.columns:
        raise ValueError(f"{path}: no 'close' or 'adj close' column")
    return _normalize(df)


def load_shiller(refresh: bool = False) -> pd.DataFrame:
    """Monthly S&P 500 with dividends reinvested.

    Caveat: Shiller prices are *monthly averages* of daily closes, not
    month-end closes. Averaging creates artificial momentum (Working, 1960),
    which flatters trend-following. Evaluate with ``execution_lag=2`` to
    neutralise most of it.
    """
    cache = CACHE_DIR / "shiller_sp500.csv"
    if refresh or not cache.exists():
        import requests

        resp = requests.get(SHILLER_URL, timeout=30)
        resp.raise_for_status()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
    raw = pd.read_csv(io.StringIO(cache.read_text()), parse_dates=["Date"]).set_index("Date")
    price = raw["SP500"].astype(float)
    # Recent rows report dividend 0.0 until Shiller fills them in; carry the last known value.
    div = raw["Dividend"].astype(float).replace(0.0, np.nan).ffill().fillna(0.0)
    ret = (price + div / 12.0) / price.shift(1) - 1.0
    tr = (1.0 + ret.fillna(0.0)).cumprod() * price.iloc[0]
    df = pd.DataFrame({"close": tr, "price": price})
    df.index.name = "date"
    return _normalize(df)


def load_yahoo(ticker: str, start: str = "1990-01-01") -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # optional dependency
        raise RuntimeError("pip install yfinance to use yahoo:<TICKER>") from exc
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        raise RuntimeError(f"yahoo returned no data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [str(c).lower() for c in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    return _normalize(raw)


def synthetic_prices(n: int = 5040, seed: int = 0, drift: float = 0.07,
                     vol: float = 0.18, bars_per_year: int = 252) -> pd.DataFrame:
    """Geometric random walk. Any 'edge' a strategy finds here is luck."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year
    rets = rng.normal((drift - 0.5 * vol**2) * dt, vol * np.sqrt(dt), n)
    idx = pd.bdate_range("2000-01-03", periods=n) if bars_per_year >= 250 else \
        pd.date_range("1900-01-31", periods=n, freq="ME")
    df = pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(rets))}, index=idx)
    df.index.name = "date"
    return df


def load(spec: str) -> pd.DataFrame:
    kind, _, arg = spec.partition(":")
    if kind == "shiller":
        return load_shiller()
    if kind == "csv":
        return load_csv(arg)
    if kind == "yahoo":
        return load_yahoo(arg)
    if kind == "synthetic":
        return synthetic_prices(seed=int(arg or 0))
    raise ValueError(f"unknown data source {spec!r}")


def bars_per_year(index: pd.DatetimeIndex) -> int:
    days = float(np.median(np.diff(index.values).astype("timedelta64[D]").astype(float)))
    if days <= 1.5:
        return 252
    if days <= 8:
        return 52
    return 12
