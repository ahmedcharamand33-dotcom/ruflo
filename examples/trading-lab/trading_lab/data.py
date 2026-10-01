"""Price data providers.

Every loader returns a DataFrame indexed by date (ascending, unique) with at
least a ``close`` column. ``close`` should be a *total-return* price where
possible (adjusted for splits and dividends), otherwise backtests understate
buy-and-hold and overstate anything that sits in cash.

Sources (``load(spec)``):
    shiller            Monthly S&P 500 total-return index, 1871-today (GitHub).
    csv:<path>         Any CSV with a date column and close / adj close.
    yahoo:<TICKER>     Daily bars via the ``yfinance`` package.
    stock:<TICKER>     Daily bars from stooq, falling back to Yahoo.
    synthetic[:seed]   Random-walk prices for tests (no edge exists by design).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

SHILLER_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
GOLD_URL = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _cached(url: str, name: str, refresh: bool = False) -> str:
    cache = CACHE_DIR / name
    if refresh or not cache.exists():
        import requests

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text)
    return cache.read_text()


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"].notna() & (df["close"] > 0)]
    if len(df) < 2:
        raise ValueError("need at least 2 valid price rows")
    return df


def load_csv(path) -> pd.DataFrame:
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
    raw = _shiller_raw(refresh)
    price = raw["SP500"].astype(float)
    # Recent rows report dividend 0.0 until Shiller fills them in; carry the last known value.
    div = raw["Dividend"].astype(float).replace(0.0, np.nan).ffill().fillna(0.0)
    ret = (price + div / 12.0) / price.shift(1) - 1.0
    tr = (1.0 + ret.fillna(0.0)).cumprod() * price.iloc[0]
    df = pd.DataFrame({"close": tr, "price": price})
    df.index.name = "date"
    return _normalize(df)


def _shiller_raw(refresh: bool = False) -> pd.DataFrame:
    text = _cached(SHILLER_URL, "shiller_sp500.csv", refresh)
    return pd.read_csv(io.StringIO(text), parse_dates=["Date"]).set_index("Date")


def bond_total_return(yields_pct: pd.Series, maturity: float = 10.0) -> pd.Series:
    """Total-return index of a constant-maturity par bond rebuilt from monthly yields.

    Each month: buy a par bond paying the current yield, a month later revalue it
    at the new yield with one month less to run (semi-annual coupons).
    """
    y = yields_pct.astype(float).replace(0.0, np.nan).ffill() / 100.0
    prev, cur = y.shift(1), y
    n = 2.0 * (maturity - 1.0 / 12.0)
    disc = (1.0 + cur / 2.0) ** (-n)
    price = prev / cur * (1.0 - disc) + disc
    ret = (price - 1.0 + prev / 12.0).fillna(0.0)
    return (1.0 + ret).cumprod()


def load_history(refresh: bool = False) -> pd.DataFrame:
    """Monthly total-return indexes for US stocks, 10y Treasuries and gold, 1871-today.

    Caveats: all three are monthly averages (use execution_lag=2); gold was
    pegged until 1971, so it behaves like cash before then.
    """
    raw = _shiller_raw(refresh)
    gold = pd.read_csv(io.StringIO(_cached(GOLD_URL, "gold_monthly.csv", refresh)))
    gold = pd.Series(gold["Price"].astype(float).values,
                     index=pd.to_datetime(gold["Date"]), name="gold")
    df = pd.DataFrame({
        "stocks": load_shiller(refresh)["close"],
        "bonds": bond_total_return(raw["Long Interest Rate"]),
    }).join(gold, how="inner").dropna()
    df.index.name = "date"
    return df / df.iloc[0]


def parse_stooq(text: str) -> pd.DataFrame:
    if not text.lstrip().lower().startswith("date"):
        raise RuntimeError(f"stooq returned no price table: {text.strip()[:80]!r}")
    return load_csv(io.StringIO(text))


def load_stooq(ticker: str) -> pd.DataFrame:
    """Daily US stock bars from stooq (split-adjusted; dividends may be excluded)."""
    import requests

    resp = requests.get(STOOQ_URL.format(symbol=f"{ticker.lower()}.us"), timeout=30)
    resp.raise_for_status()
    return parse_stooq(resp.text)


def load_stock(ticker: str) -> pd.DataFrame:
    """Daily bars for one US stock: stooq first, Yahoo (yfinance) as fallback."""
    try:
        return load_stooq(ticker)
    except Exception as stooq_err:
        try:
            return load_yahoo(ticker)
        except Exception as yahoo_err:
            raise RuntimeError(f"no price source reachable for {ticker}: "
                               f"stooq: {stooq_err}; yahoo: {yahoo_err}") from yahoo_err


def load_universe(spec: str) -> pd.DataFrame:
    """Closes by column: 'history', 'stock:MSFT[,AAPL]' or 'yahoo:SPY,IEF,GLD'."""
    kind, _, arg = spec.partition(":")
    if kind == "history":
        return load_history()
    if kind == "stock":
        return pd.DataFrame({t: load_stock(t)["close"] for t in arg.split(",")}).dropna()
    if kind == "yahoo":
        cols = {t: load_yahoo(t)["close"] for t in arg.split(",")}
        return pd.DataFrame(cols).dropna()
    raise ValueError(f"unknown universe {spec!r}")


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
    if kind == "stock":
        return load_stock(arg)
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
