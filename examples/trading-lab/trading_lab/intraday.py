"""Day-trading SPY: published intraday rules, written down before testing.

Every rule is flat at the close (nothing held overnight), trades at most 1x
the account (long or short, never borrowed beyond that) and is taken as its
source describes it, with no parameter tuning. Decisions use bars up to t and
fill at the open of bar t+1; every change of position pays costs per side.

Discovery 2016-2022, judged only on 2023-today. The luck haircut charges
these five rules plus the five daily SPY hypotheses already tested (10 trials).

    python -m trading_lab intraday
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import deflated_sharpe, summary

COST_BPS = 1.0       # per side, same as the daily SPY research
SLIPPAGE_BPS = 1.0
CHEAP_BPS = 0.25     # sensitivity only: roughly SPY's half-spread, no slippage
SPLIT = "2023-01-01"
N_TRIALS = 10
SESSION = 390        # one-minute bars in a full session


def _days(bars: pd.DataFrame):
    """Yield (date, open, high, low, close, volume, previous close) per session."""
    prev = np.nan
    for day, g in bars.groupby(bars.index.normalize(), sort=True):
        if len(g) < 60:  # broken or tiny session
            continue
        yield day, g.open.values, g.high.values, g.low.values, g.close.values, g.volume.values, prev
        prev = g.close.values[-1]


def _vector_pnl(o, c, pos, cost):
    """Return for a position path: pos[i] is held from open[i] to open[i+1] (last bar to close)."""
    nxt = np.append(o[1:], c[-1])
    gross = float(np.sum(pos * (nxt / o - 1.0)))
    turns = np.abs(np.diff(np.concatenate([[0.0], pos, [0.0]]))).sum()
    return gross - turns * cost, int((np.diff(np.concatenate([[0.0], pos])) != 0).sum() > 0)


# --- the rules ---------------------------------------------------------------
# Each returns a position path (one value per bar, decided from earlier bars only).

def intraday_momentum(o, h, l, c, v, prev, ctx):
    """Gao, Han, Li & Zhou (2018, J. Financial Economics), "Market intraday momentum":
    the return from yesterday's close to 10:00 predicts the last half hour. Trade 15:30-close."""
    pos = np.zeros(len(o))
    if len(o) < SESSION or np.isnan(prev):
        return pos
    pos[360:] = np.sign(c[29] / prev - 1.0)
    return pos


def opening_range_breakout(o, h, l, c, v, prev, ctx):
    """Crabel (1990) / Zarattini & Aziz (2023), 5-minute opening range: go the way the first
    5-minute bar closed, stop at the other end of that bar, otherwise hold to the close."""
    pos = np.zeros(len(o))
    side = np.sign(c[4] - o[0])
    if side == 0:
        return pos
    stop = l[:5].min() if side > 0 else h[:5].max()
    pos[5:] = side
    hit = np.where(l[5:] <= stop)[0] if side > 0 else np.where(h[5:] >= stop)[0]
    if hit.size:
        k = 5 + hit[0]
        ctx["stop_fill"] = (k, stop, "stop")  # exit inside bar k at the stop price
        pos[k + 1:] = 0.0
    return pos


def gap_fade(o, h, l, c, v, prev, ctx):
    """Floor-trader gap fill (Raschke & Connors, "Street Smarts", 1995): after a 0.25%-1% gap,
    trade back toward yesterday's close from 09:31; take profit there, else exit at the close."""
    pos = np.zeros(len(o))
    if np.isnan(prev):
        return pos
    gap = o[0] / prev - 1.0
    if not 0.0025 <= abs(gap) <= 0.01:
        return pos
    side = -np.sign(gap)
    pos[1:] = side
    hit = np.where(h[1:] >= prev)[0] if side > 0 else np.where(l[1:] <= prev)[0]
    if hit.size:
        k = 1 + hit[0]
        ctx["stop_fill"] = (k, prev, "target")
        pos[k + 1:] = 0.0
    return pos


def vwap_trend(o, h, l, c, v, prev, ctx):
    """Zarattini & Aziz (2023), VWAP trend: long above the day's VWAP, short below,
    re-checked every minute."""
    vwap = np.cumsum((h + l + c) / 3.0 * v) / np.maximum(np.cumsum(v), 1.0)
    pos = np.zeros(len(o))
    pos[1:] = np.sign(c[:-1] - vwap[:-1])
    return pos


def noise_breakout(o, h, l, c, v, prev, ctx):
    """Zarattini, Aziz & Barbon (2024), "Beat the Market" (SPY): a 14-day "noise area" around
    the open; at each half hour go long above it (and above VWAP), short below, else flat."""
    sigma = ctx.get("sigma")
    pos = np.zeros(len(o))
    if sigma is None or np.isnan(prev) or len(o) < SESSION:
        return pos
    upper = max(o[0], prev) * (1.0 + sigma)
    lower = min(o[0], prev) * (1.0 - sigma)
    vwap = np.cumsum((h + l + c) / 3.0 * v) / np.maximum(np.cumsum(v), 1.0)
    state = 0.0
    for k in range(29, len(o) - 1):
        if (k + 1) % 30 == 0:  # decisions at 10:00, 10:30, ... using the bar that just closed
            if c[k] > max(upper[k], vwap[k]):
                state = 1.0
            elif c[k] < min(lower[k], vwap[k]):
                state = -1.0
            else:
                state = 0.0
        pos[k + 1] = state
    return pos


RULES = {
    "intraday momentum (last 30 min)": intraday_momentum,
    "opening-range breakout (5 min)": opening_range_breakout,
    "gap fade (0.25-1% gaps)": gap_fade,
    "VWAP trend": vwap_trend,
    "noise-area breakout": noise_breakout,
}


def _noise(bars: pd.DataFrame) -> dict:
    """Per day: average |close/open - 1| at each minute over the previous 14 sessions."""
    day = bars.index.normalize()
    k = (bars.index.hour * 60 + bars.index.minute) - 570
    first = bars.groupby(day).open.transform("first")
    move = (bars.close / first - 1.0).abs()
    table = pd.DataFrame({"day": day, "k": k, "m": move.values}).pivot(index="day", columns="k", values="m")
    table = table.reindex(columns=range(SESSION))
    sigma = table.rolling(14, min_periods=14).mean().shift(1)
    return {d: row.values for d, row in sigma.iterrows() if not row.isna().all()}


def daily_returns(bars: pd.DataFrame, cost_bps: float = COST_BPS + SLIPPAGE_BPS) -> pd.DataFrame:
    """One column per rule (plus 'hold during the day'), one row per session, net of costs."""
    cost = cost_bps / 1e4
    noise = _noise(bars)
    out, trades = {}, {}
    for day, o, h, l, c, v, prev in _days(bars):
        row, trow = {}, {}
        hold, _ = _vector_pnl(o, c, np.ones(len(o)), cost)
        row["hold during the day"] = hold
        for name, rule in RULES.items():
            ctx = {"sigma": noise.get(day)}
            pos = rule(o, h, l, c, v, prev, ctx)
            if "stop_fill" in ctx:  # bar k exits at a price level, not at the next open
                k, price, kind = ctx["stop_fill"]
                side = pos[k]
                if kind == "stop":  # a stop gapped through fills at the worse open price
                    price = min(price, o[k]) if side > 0 else max(price, o[k])
                pos_k = pos.copy()
                pos_k[k:] = 0.0
                r, t = _vector_pnl(o, c, pos_k, cost)  # already charges the exit at bar k
                r += side * (price / o[k] - 1.0)
            else:
                r, t = _vector_pnl(o, c, pos, cost)
            row[name], trow[name] = r, int(np.abs(np.diff(np.concatenate([[0.0], pos, [0.0]]))).sum() / 2)
        out[day], trades[day] = row, trow
    res = pd.DataFrame.from_dict(out, orient="index")
    res.attrs["trades"] = pd.DataFrame.from_dict(trades, orient="index")
    return res


def evaluate(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, bps in (("realistic costs", COST_BPS + SLIPPAGE_BPS), ("spread only", CHEAP_BPS)):
        daily = daily_returns(bars, bps)
        trades = daily.attrs["trades"]
        for period, sl in (("2016-2022", slice(None, "2022-12-31")), ("2023-now", slice(SPLIT, None))):
            for name in daily.columns:
                r = daily[name].loc[sl]
                s = summary(r, 252)
                active = r[r != 0]
                n = trades[name].loc[sl].sum() if name in trades else len(r)
                rows.append({"costs": label, "period": period, "rule": name, "cagr": s["cagr"],
                             "sharpe": s["sharpe"], "max_dd": s["max_dd"],
                             "trades_yr": n / s["years"], "win_rate": float((active > 0).mean()) if len(active) else 0.0,
                             "p_real": deflated_sharpe(r, N_TRIALS) if name in RULES else np.nan})
    return pd.DataFrame(rows)


def render(table: pd.DataFrame) -> str:
    out = []
    for (costs, period), g in table.groupby(["costs", "period"], sort=False):
        out.append(f"-- {period}, {costs} --")
        out.append(f"{'rule':34}{'CAGR':>8}{'Sharpe':>8}{'worst drop':>11}{'trades/yr':>10}{'win %':>7}{'P(real)':>9}")
        for _, r in g.iterrows():
            p = "" if np.isnan(r.p_real) else f"{r.p_real:.0%}"
            out.append(f"{r.rule:34}{r.cagr:>8.1%}{r.sharpe:>8.2f}{r.max_dd:>11.0%}{r.trades_yr:>10.0f}"
                       f"{r.win_rate:>7.0%}{p:>9}")
        out.append("")
    out.append("P(real) = chance the rule truly makes money after costs, charged for 10 rules tried.")
    return "\n".join(out)


FORWARD_START = "2026-10-21"  # rules frozen on this date; only later sessions are logged


def forward_log(path, bars: pd.DataFrame) -> str:
    """Append each new session's result for every rule (both cost levels) to a CSV.

    The rules are frozen, so every logged day is out-of-sample by construction.
    """
    from pathlib import Path

    path = Path(path)
    recent = bars.loc[bars.index.normalize() >= bars.index.normalize().unique()[-40]]
    frames = []
    for label, bps in (("realistic", COST_BPS + SLIPPAGE_BPS), ("spread_only", CHEAP_BPS)):
        d = daily_returns(recent, bps).loc[FORWARD_START:]
        d.attrs = {}  # the trade counts are not logged
        frames.append(d.add_suffix(f" [{label}]"))
    new = pd.concat(frames, axis=1).round(6)
    new.index = new.index.strftime("%Y-%m-%d")
    new.index.name = "date"
    if path.exists():
        old = pd.read_csv(path, index_col="date")
        new = pd.concat([old, new[~new.index.isin(old.index)]]).sort_index()
    if new.empty:
        return "intraday forward test: no sessions since the rules were frozen yet"
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path)
    best = "noise-area breakout [spread_only]"
    total = float((1.0 + new[best]).prod() - 1.0)
    return (f"intraday forward test: {len(new)} sessions logged through {new.index[-1]}; "
            f"noise-area breakout (spread-only costs) {new[best].iloc[-1]:+.2%} today, {total:+.2%} since start")
