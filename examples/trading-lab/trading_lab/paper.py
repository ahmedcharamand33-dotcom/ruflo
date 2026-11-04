"""Paper trading with an append-only event journal.

Run ``step`` once per bar (e.g. daily after the close, from cron). It:
  1. replays the journal to rebuild the account (event sourcing),
  2. re-selects the strategy when the current pick is older than one test
     window, using the same walk-forward rule as the backtest,
  3. trades to the chosen strategy's target exposure at the latest close,
  4. records a mark-to-market so live results can be audited later.

``review`` is the feedback loop: it compares live results with what the
backtest led us to expect, and says HALT when the gap is too large to be bad
luck. A live system that underperforms its backtest is the most common
real-world failure; catching it early is worth more than any new indicator.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .data import bars_per_year as infer_bpy
from .strategies import Strategy
from .walkforward import select

MIN_TRADE_FRACTION = 1e-4  # ignore rebalances smaller than 0.01% of equity


@dataclass
class Account:
    cash: float = 0.0
    shares: float = 0.0
    last_date: str | None = None
    strategy: str | None = None
    selected_on: str | None = None
    fills: list[dict] = field(default_factory=list)
    marks: list[dict] = field(default_factory=list)

    def equity(self, price: float) -> float:
        return self.cash + self.shares * price


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def replay(events: list[dict]) -> Account:
    acct = Account()
    for e in events:
        kind = e["type"]
        if kind == "open":
            acct.cash = e["cash"]
        elif kind == "select":
            acct.strategy, acct.selected_on = e["strategy"], e["date"]
        elif kind == "fill":
            acct.shares += e["shares"]
            acct.cash -= e["shares"] * e["price"] + e["cost"]
            acct.fills.append(e)
        elif kind == "mark":
            acct.last_date = e["date"]
            acct.marks.append(e)
    return acct


def step(close: pd.Series, journal: Path, candidates: list[Strategy], cash: float = 100_000.0,
         train_years: float = 10, test_years: float = 1, cost_bps: float = 5.0,
         slippage_bps: float = 5.0) -> str:
    """Process the latest bar in ``close``. Idempotent: re-running on the same bar is a no-op."""
    events = read_events(journal)
    if not events:
        append_event(journal, {"type": "open", "cash": cash})
        events = read_events(journal)
    acct = replay(events)
    date = str(close.index[-1].date())
    if acct.last_date is not None and date <= acct.last_date:
        return f"{date}: already processed (last bar {acct.last_date})"

    bpy = infer_bpy(close.index)
    train = int(train_years * bpy)
    if len(close) < train + 2:
        raise ValueError(f"need {train + 2} bars of history, have {len(close)}")
    targets = {c.label: c.positions(close, bpy) for c in candidates}

    stale = acct.selected_on is None or acct.strategy not in targets or \
        (pd.Timestamp(date) - pd.Timestamp(acct.selected_on)).days >= test_years * 365.25
    if stale:
        standalone = {lbl: run_backtest(close, t, cost_bps, slippage_bps) for lbl, t in targets.items()}
        best, score = select(standalone, len(close), train, bpy)
        append_event(journal, {"type": "select", "date": date, "strategy": best,
                               "train_sharpe": round(score, 4), "previous": acct.strategy})
        acct.strategy, acct.selected_on = best, date

    price = float(close.iloc[-1])
    target = float(targets[acct.strategy].iloc[-1])
    equity = acct.equity(price)
    # Size so exposure hits the target *after* paying costs (never borrow to pay commission).
    rate = (cost_bps + slippage_bps) / 1e4
    gap = target * equity - acct.shares * price
    delta = gap / (price * (1.0 + target * rate if gap > 0 else 1.0 - target * rate))
    msg = f"{date}: {acct.strategy} target {target:.0%} @ {price:,.2f}"
    if abs(delta * price) > equity * MIN_TRADE_FRACTION:
        cost = abs(delta * price) * rate
        append_event(journal, {"type": "fill", "date": date, "shares": delta, "price": price,
                               "cost": cost, "target": target, "strategy": acct.strategy})
        acct.shares += delta
        acct.cash -= delta * price + cost
        msg += f", traded {delta:+,.4f} sh (cost {cost:,.2f})"
    equity = acct.equity(price)
    append_event(journal, {"type": "mark", "date": date, "price": price, "equity": equity,
                           "exposure": acct.shares * price / equity if equity else 0.0})
    return msg + f", equity {equity:,.2f}"


def review(journal: Path, expected_returns: pd.Series, min_bars: int = 20,
           halt_z: float = -2.0) -> dict:
    """Compare live per-bar returns with the backtest's out-of-sample distribution."""
    acct = replay(read_events(journal))
    equity = pd.Series([m["equity"] for m in acct.marks], dtype=float)
    live = equity.pct_change().dropna()
    out = {"bars": len(live), "fills": len(acct.fills), "strategy": acct.strategy,
           "equity": float(equity.iloc[-1]) if len(equity) else None, "status": "WARMING UP"}
    if len(live) < min_bars:
        return out
    mu, sd = expected_returns.mean(), expected_returns.std()
    z = (live.mean() - mu) / (sd / math.sqrt(len(live))) if sd else 0.0
    out.update(live_mean=float(live.mean()), expected_mean=float(mu), z=float(z),
               status="HALT" if z < halt_z else "OK")
    return out
