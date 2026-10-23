"""The SPY rules on 98 years of S&P 500 daily data (1928-today).

SPY only exists since 1993, so the rules have never been tested on 1929, the
1930s, the 1970s or the 1987 crash. Nothing here is tuned: the rules are the
ones already pre-registered in spy_research.py, run unchanged.

Data: S&P 500 daily closes (^GSPC, price only) plus Shiller's monthly dividend
yield spread over trading days, so holding earns dividends. Cash earns the
13-week T-bill yield from 1960; before that no rate is available and cash
earns 0%, which understates the rules that sit in cash.

    python -m trading_lab long-history
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .data import CACHE_DIR, _shiller_raw, drop_unfinished_session
from .metrics import deflated_sharpe, sharpe, summary
from .spy_research import COST_BPS, SLIPPAGE_BPS, _yahoo, tbill_yield, trend_200, vol_target

N_TRIALS = 6  # the SPY hypothesis family these rules were drawn from

CRISES = [
    ("1929 crash and Depression", "1929-09-01", "1932-12-31"),
    ("1937-38 crash", "1937-03-01", "1938-12-31"),
    ("1973-74 bear market", "1973-01-01", "1974-12-31"),
    ("1987 crash", "1987-08-01", "1987-12-31"),
    ("2000-02 dot-com bust", "2000-03-01", "2002-12-31"),
    ("2008 financial crisis", "2007-10-01", "2009-06-30"),
    ("2020 covid crash", "2020-02-01", "2020-04-30"),
    ("2022 bear market", "2022-01-01", "2022-12-31"),
]

RULES = {"trend: above 200-day average": trend_200, "volatility control (15%)": vol_target}


def load(refresh: bool = False) -> pd.DataFrame:
    cache = CACHE_DIR / "sp500_daily.csv"
    if refresh or not cache.exists():
        d = _yahoo("^GSPC", start="1927-12-01")
        d.index = pd.to_datetime(d.index).tz_localize(None)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        d[["Close"]].rename(columns={"Close": "price"}).to_csv(cache)
    df = drop_unfinished_session(pd.read_csv(cache, index_col=0, parse_dates=True).rename(columns={"price": "close"}))
    raw = _shiller_raw()
    div = raw["Dividend"].astype(float).replace(0.0, np.nan).ffill()
    yld = (div / raw["SP500"].astype(float)).reindex(df.index, method="ffill").fillna(0.0)
    ret = df.close.pct_change().fillna(0.0) + yld / 252.0
    out = pd.DataFrame({"close": (1.0 + ret).cumprod() * df.close.iloc[0]}, index=df.index)
    out["cash"] = (tbill_yield().reindex(out.index, method="ffill") / 252.0).fillna(0.0)
    return out


def runs(df: pd.DataFrame) -> dict[str, pd.Series]:
    hold = run_backtest(df.close, pd.Series(1.0, index=df.index), COST_BPS, SLIPPAGE_BPS).returns
    out = {"buy & hold S&P 500": hold}
    for name, fn in RULES.items():
        out[name] = run_backtest(df.close, fn(df), COST_BPS, SLIPPAGE_BPS, cash_rate=df.cash).returns
    return out


def _worst_year(r: pd.Series) -> float:
    return float((1.0 + r).groupby(r.index.year).prod().sub(1.0).min())


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = runs(df)
    hold = series["buy & hold S&P 500"]
    rows = []
    for period, sl in (("1928-1992 (before SPY)", slice("1928-06-01", "1992-12-31")),
                       ("1993-today", slice("1993-01-01", None)), ("all 98 years", slice("1928-06-01", None))):
        b = hold.loc[sl]
        for name, r in series.items():
            r = r.loc[sl]
            s = summary(r, 252)
            rows.append({"period": period, "rule": name, "cagr": s["cagr"], "sharpe": s["sharpe"],
                         "max_dd": s["max_dd"], "worst_year": _worst_year(r),
                         "p_beats_hold": np.nan if r is b or name.startswith("buy") else
                         deflated_sharpe(r, N_TRIALS, sharpe(b, 1))})
    crises = []
    for label, a, z in CRISES:
        row = {"crisis": label}
        for name, r in series.items():
            row[name] = float((1.0 + r.loc[a:z]).prod() - 1.0)
        crises.append(row)
    return pd.DataFrame(rows), pd.DataFrame(crises)


def render(table: pd.DataFrame, crises: pd.DataFrame) -> str:
    out = ["== The SPY rules on S&P 500 daily data, 1928-today (dividends included) =="]
    for period, g in table.groupby("period", sort=False):
        out += [f"-- {period} --",
                f"{'rule':32}{'CAGR':>7}{'Sharpe':>8}{'worst drop':>11}{'worst year':>11}{'P(beat hold)':>14}"]
        for _, r in g.iterrows():
            p = "" if np.isnan(r.p_beats_hold) else f"{r.p_beats_hold:.0%}"
            out.append(f"{r.rule:32}{r.cagr:>7.1%}{r.sharpe:>8.2f}{r.max_dd:>11.0%}{r.worst_year:>11.0%}{p:>14}")
        out.append("")
    names = [c for c in crises.columns if c != "crisis"]
    out.append("-- Total return through each crisis --")
    out.append(f"{'crisis':28}" + "".join(f"{n.split(':')[0].split(' (')[0][:16]:>18}" for n in names))
    for _, r in crises.iterrows():
        out.append(f"{r.crisis:28}" + "".join(f"{r[n]:>18.0%}" for n in names))
    out.append("\nCash earns 0% before 1960 (no T-bill series), which understates the rules in the 1930s-50s.")
    return "\n".join(out)
