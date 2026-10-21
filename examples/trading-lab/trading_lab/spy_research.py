"""SPY research: pre-registered hypotheses about how SPY moves, tested honestly.

The hypothesis list is fixed before looking at results and every one is
reported, winners and losers, so the deflated Sharpe charge (n_trials) is
real. Each is fit on nothing (textbook parameters) and judged on two
periods: discovery 1994-2012 and out-of-sample 2013-today.

    python -m trading_lab spy-research
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import CACHE_DIR, drop_unfinished_session
from .metrics import deflated_sharpe, sharpe, summary

COST_BPS = 1.0       # half the bid-ask spread plus a little, per side
SLIPPAGE_BPS = 1.0
SPLIT = "2013-01-01"


def load_spy(refresh: bool = False) -> pd.DataFrame:
    """Daily SPY OHLC (dividend-adjusted) joined with the VIX close."""
    cache = CACHE_DIR / "spy_research.csv"
    if refresh or not cache.exists():
        import yfinance as yf

        frames = {}
        for t in ("SPY", "^VIX"):
            d = yf.download(t, start="1993-01-01", auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.index = pd.to_datetime(d.index).tz_localize(None)
            frames[t] = d
        spy = frames["SPY"].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        spy["vix"] = frames["^VIX"]["Close"].reindex(spy.index).ffill()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        spy.to_csv(cache)
    df = pd.read_csv(cache, index_col=0, parse_dates=True)
    return drop_unfinished_session(df.dropna(subset=["close"]))


def rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


# ---- the five pre-registered hypotheses (targets: exposure decided at the close) ----

def trend_200(df):
    sma = df.close.rolling(200).mean()
    return (df.close > sma).astype(float).where(sma.notna(), 0.0)


def rsi2_dip(df):
    """Buy a 2-day oversold dip in an uptrend; sell when price closes above its 5-day average."""
    up = df.close > df.close.rolling(200).mean()
    entry = (rsi(df.close, 2) < 10) & up
    exit_ = df.close > df.close.rolling(5).mean()
    state = np.where(entry, 1.0, np.where(exit_, 0.0, np.nan))
    return pd.Series(state, index=df.index).ffill().fillna(0.0)


def turn_of_month(df):
    """Hold from the last trading day of a month through the 3rd trading day of the next."""
    month = df.index.to_period("M")
    pos_in_month = pd.Series(1, index=df.index).groupby(month).cumcount()
    from_end = pd.Series(1, index=df.index).groupby(month).cumcount(ascending=False)
    # exposure held over the *next* bar, so mark the day before each wanted holding day
    want = ((pos_in_month <= 2) | (from_end == 0)).astype(float)
    return want.shift(-1).fillna(0.0)


def vol_target(df, target=0.15):
    vol = df.close.pct_change().rolling(20).std() * np.sqrt(252)
    return (target / vol).clip(upper=1.0).fillna(0.0)


HYPOTHESES = {
    "trend: above 200-day average": trend_200,
    "dip: RSI(2)<10 in uptrend": rsi2_dip,
    "turn of the month": turn_of_month,
    "volatility control (15%)": vol_target,
}


def overnight_only(df) -> pd.Series:
    """Buy at the close, sell at the next open, every day: two trades a day."""
    gap = df.open / df.close.shift(1) - 1.0
    return gap.fillna(0.0) - 2 * (COST_BPS + SLIPPAGE_BPS) / 1e4


def facts(df) -> dict:
    """Descriptive behaviour of SPY: no trading rules, just what it does."""
    r = df.close.pct_change().dropna()
    overnight = (df.open / df.close.shift(1) - 1).dropna()
    intraday = (df.close / df.open - 1).dropna()
    big_down = r[r < -0.03]
    after_big_down = r.shift(-1).reindex(big_down.index).dropna()
    by_dow = r.groupby(r.index.dayofweek).mean() * 1e4
    return {
        "years": round(len(r) / 252, 1),
        "up_days": float((r > 0).mean()),
        "avg_day_bps": float(r.mean() * 1e4),
        "growth_overnight": float((1 + overnight).prod() - 1),
        "growth_intraday": float((1 + intraday).prod() - 1),
        "days_down_3pct": int(len(big_down)),
        "next_day_after_down_3pct_bps": float(after_big_down.mean() * 1e4),
        "share_of_big_days_when_vix_above_25": float(
            (df.vix.reindex(r[r.abs() > 0.03].index) > 25).mean()),
        "avg_day_bps_by_weekday": {d: round(float(v), 1) for d, v in
                                  zip(["Mon", "Tue", "Wed", "Thu", "Fri"], by_dow.values)},
    }


def evaluate(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """One row per hypothesis per period, against buy-and-hold over the same days."""
    n_trials = len(HYPOTHESES) + 1  # + overnight_only
    bh = run_backtest(df.close, pd.Series(1.0, index=df.index), COST_BPS, SLIPPAGE_BPS, lag).returns
    series = {name: run_backtest(df.close, fn(df), COST_BPS, SLIPPAGE_BPS, lag) for name, fn in HYPOTHESES.items()}
    rows = []
    for period, sl in (("1994-2012", slice("1994-01-01", "2012-12-31")), ("2013-now", slice(SPLIT, None))):
        b = bh.loc[sl]
        if len(b) < 252:  # not enough data in this period to say anything
            continue
        bs = summary(b, 252)
        rows.append({"rule": "buy & hold SPY", "period": period, "cagr": bs["cagr"], "sharpe": bs["sharpe"],
                     "max_dd": bs["max_dd"], "in_market": 1.0, "trades_yr": 0.0, "p_beats_bh": np.nan})
        for name, res in list(series.items()) + [("overnight only", None)]:
            if res is None:
                r, held = overnight_only(df).loc[sl], pd.Series(1.0, index=b.index)
            else:
                r, held = res.returns.loc[sl], res.held.loc[sl]
            s = summary(r, 252, held)
            entries = int(((held > 0) & (held.shift(1).fillna(0) == 0)).sum()) if res is not None else 252 * s["years"]
            rows.append({"rule": name, "period": period, "cagr": s["cagr"], "sharpe": s["sharpe"],
                         "max_dd": s["max_dd"], "in_market": float((held > 0).mean()),
                         "trades_yr": entries / s["years"],
                         "p_beats_bh": deflated_sharpe(r, n_trials, sharpe(b, 1))})
    return pd.DataFrame(rows)


def render(table: pd.DataFrame, f: dict) -> str:
    out = ["== SPY: what it does (1993-today) ==",
           f"Up days: {f['up_days']:.0%}. Average day: {f['avg_day_bps']:+.1f} bp.",
           f"All growth from close to next open (overnight): {f['growth_overnight']:+.0%}; "
           f"open to close (daytime): {f['growth_intraday']:+.0%}.",
           f"Days down more than 3%: {f['days_down_3pct']}; average next day: "
           f"{f['next_day_after_down_3pct_bps']:+.0f} bp. "
           f"{f['share_of_big_days_when_vix_above_25']:.0%} of 3%+ moves happened with VIX above 25.",
           "Average day by weekday (bp): " + ", ".join(f"{k} {v:+.1f}" for k, v in f["avg_day_bps_by_weekday"].items()),
           ""]
    for period, g in table.groupby("period", sort=False):
        out.append(f"-- {period} --")
        out.append(f"{'rule':32}{'CAGR':>7}{'Sharpe':>8}{'worst drop':>11}{'in mkt':>8}{'entries/yr':>11}{'P(beat B&H)':>13}")
        for _, r in g.iterrows():
            p = "" if np.isnan(r.p_beats_bh) else f"{r.p_beats_bh:.0%}"
            out.append(f"{r.rule:32}{r.cagr:>7.1%}{r.sharpe:>8.2f}{r.max_dd:>11.0%}{r.in_market:>8.0%}"
                       f"{r.trades_yr:>11.0f}{p:>13}")
        out.append("")
    return "\n".join(out)


def _vol_control_weights(closes: pd.DataFrame, bpy: int) -> pd.DataFrame:
    """Live form of the volatility-control hypothesis, one weight per column (single asset)."""
    return pd.DataFrame({c: vol_target(pd.DataFrame({"close": closes[c]})) for c in closes.columns})


# Rules the paper engine can run. Only pre-registered hypotheses that held up go here.
LIVE_RULES = {"spy volatility control": _vol_control_weights}


def observation(df: pd.DataFrame) -> dict:
    """Today's SPY state in plain numbers, appended to the daily log for later study."""
    c = df.close
    sma200 = c.rolling(200).mean().iloc[-1]
    vol20 = c.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    return {
        "date": str(df.index[-1].date()), "open": round(float(df.open.iloc[-1]), 4),
        "close": round(float(c.iloc[-1]), 4), "day_return": round(float(c.pct_change().iloc[-1]), 6),
        "overnight_gap": round(float(df.open.iloc[-1] / c.iloc[-2] - 1), 6),
        "vix": round(float(df.vix.iloc[-1]), 2), "vol20": round(float(vol20), 4),
        "pct_above_200d": round(float(c.iloc[-1] / sma200 - 1), 4),
        "rsi2": round(float(rsi(c, 2).iloc[-1]), 1),
        "vol_control_exposure": round(float(vol_target(df).iloc[-1]), 3),
        "trend_200_signal": int(trend_200(df).iloc[-1]),
        "dip_signal": int(rsi2_dip(df).iloc[-1]),
    }


def _yahoo(ticker: str, **kw) -> pd.DataFrame:
    import yfinance as yf

    d = yf.download(ticker, auto_adjust=False, progress=False, **kw)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d


def vix3m_close(date: str) -> float | None:
    """3-month VIX on ``date``. VIX above VIX3M (ratio > 1) marks acute stress."""
    try:
        v = _yahoo("^VIX3M", period="10d")["Close"]
        v.index = pd.to_datetime(v.index).tz_localize(None)
        return round(float(v.loc[:date].iloc[-1]), 2)
    except Exception:
        return None


def save_intraday(folder: Path, now: pd.Timestamp | None = None) -> str:
    """Keep every completed 5-minute SPY bar Yahoo still has (it only serves ~60 days).

    Bars are appended to one CSV per month, de-duplicated by timestamp, so running
    daily builds a permanent intraday history that would otherwise disappear.
    """
    now = now or pd.Timestamp.now(tz="America/New_York")
    bars = _yahoo("SPY", period="60d", interval="5m", prepost=False)
    if bars.empty:
        return "intraday: no bars returned"
    bars.index = pd.to_datetime(bars.index).tz_convert("America/New_York")
    if (now.hour, now.minute) < (16, 15):  # today's session is not finished
        bars = bars[bars.index.date < now.date()]
    bars = bars[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower).round(4)
    folder.mkdir(parents=True, exist_ok=True)
    added = 0
    for month, chunk in bars.groupby(bars.index.strftime("%Y-%m")):
        f = folder / f"{month}.csv"
        chunk = chunk.copy()
        chunk.index = chunk.index.strftime("%Y-%m-%d %H:%M")
        chunk.index.name = "time"
        if f.exists():
            old = pd.read_csv(f, index_col="time")
            merged = pd.concat([old, chunk[~chunk.index.isin(old.index)]]).sort_index()
            added += len(merged) - len(old)
        else:
            merged, added = chunk, added + len(chunk)
        merged.to_csv(f)
    return f"intraday: {added} new 5-minute bars saved"


def append_observation(path: Path, refresh: bool = True) -> str:
    """Add today's row to the SPY observation log (idempotent per date)."""
    row = observation(load_spy(refresh=refresh))
    row["vix3m"] = vix3m_close(row["date"])
    row["vix_term_ratio"] = round(row["vix"] / row["vix3m"], 3) if row["vix3m"] else None
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        log = pd.read_csv(path)
        if row["date"] in set(log["date"].astype(str)):
            return f"{row['date']}: already logged"
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    else:
        log = pd.DataFrame([row])
    log.to_csv(path, index=False)
    return f"{row['date']}: logged (close {row['close']}, VIX {row['vix']}, exposure {row['vol_control_exposure']:.0%})"
