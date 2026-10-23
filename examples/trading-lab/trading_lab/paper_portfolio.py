"""Hands-off paper trading for a multi-asset portfolio rule.

Designed to run unattended once per market day (e.g. a scheduled CI job):

  * marks the book to market every run,
  * rebalances to the rule's target weights on the first run of each month,
  * never levers: buys are sized so cash stays >= 0 after costs,
  * circuit breaker: if equity falls ``max_dd`` below its peak, it sells
    everything and HALTs until someone deliberately resumes it.

The journal is append-only JSONL; the book is rebuilt by replaying it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import bars_per_year
from .paper import append_event, read_events
from .portfolio import candidates


def all_rules() -> dict:
    """Portfolio rules plus single-asset live rules (e.g. SPY volatility control)."""
    from .spy_research import LIVE_RULES
    return {**candidates(), **LIVE_RULES}

DEFAULT_RULE = "equal weight + trend"
MIN_TRADE = 1.0  # dollars


@dataclass
class Book:
    cash: float = 0.0
    shares: dict = field(default_factory=dict)
    last_date: str | None = None
    rebalanced: str | None = None   # YYYY-MM of last rebalance
    halted: str | None = None       # reason, while halted
    marks: list[dict] = field(default_factory=list)
    fills: int = 0

    def equity(self, prices: pd.Series) -> float:
        return self.cash + sum(q * float(prices[a]) for a, q in self.shares.items())


def replay(events: list[dict]) -> Book:
    book = Book()
    for e in events:
        kind = e["type"]
        if kind == "open":
            book.cash = e["cash"]
        elif kind == "fill":
            book.shares[e["asset"]] = book.shares.get(e["asset"], 0.0) + e["shares"]
            book.cash -= e["shares"] * e["price"] + e["cost"]
            book.fills += 1
        elif kind == "interest":
            book.cash += e["amount"]
        elif kind == "rebalance":
            book.rebalanced = e["date"][:7]
        elif kind == "halt":
            book.halted = e["reason"]
        elif kind == "resume":
            book.halted = None
        elif kind == "mark":
            book.last_date = e["date"]
            book.marks.append(e)
    return book


def _trade_to(book: Book, journal: Path, date: str, prices: pd.Series, weights: pd.Series,
              rate: float, why: str) -> list[str]:
    equity = book.equity(prices)
    current = {a: book.shares.get(a, 0.0) * float(prices[a]) for a in prices.index}
    est_cost = sum(abs(weights.get(a, 0.0) * equity - v) for a, v in current.items()) * rate
    budget = equity - est_cost  # pay costs out of the allocation, not with borrowed money
    notes = []
    # sells first so buys are funded by them
    orders = sorted(prices.index, key=lambda a: weights.get(a, 0.0) * budget - current[a])
    for a in orders:
        delta_value = weights.get(a, 0.0) * budget - current[a]
        if abs(delta_value) < MIN_TRADE:
            continue
        px = float(prices[a])
        qty, cost = delta_value / px, abs(delta_value) * rate
        append_event(journal, {"type": "fill", "date": date, "asset": a, "shares": qty,
                               "price": px, "cost": cost, "reason": why})
        book.shares[a] = book.shares.get(a, 0.0) + qty
        book.cash -= qty * px + cost
        notes.append(f"{a} {'+' if qty > 0 else ''}{delta_value:,.0f}")
    return notes


def step(closes: pd.DataFrame, journal: Path, rule: str = DEFAULT_RULE, cash: float = 100_000.0,
         cost_bps: float = 5.0, slippage_bps: float = 5.0, max_dd: float = 0.15,
         band: float | None = None, cash_yield: float | None = None) -> str:
    """One unattended step. Rebalances on the first run of each month, or, with ``band``,
    on any run where a holding has drifted more than ``band`` from its target weight.
    With ``cash_yield`` (annual T-bill yield), idle cash earns interest for each
    trading day since the last mark, as it would parked in a T-bill fund."""
    events = read_events(journal)
    if not events:
        append_event(journal, {"type": "open", "cash": cash, "rule": rule})
        events = read_events(journal)
    book = replay(events)
    date = str(closes.index[-1].date())
    if book.last_date is not None and date <= book.last_date:
        return f"{date}: already processed"

    if cash_yield and book.last_date and book.cash > 0:
        days = int(np.busday_count(book.last_date, date))
        amount = round(book.cash * cash_yield / 252.0 * days, 2)
        if amount > 0:
            append_event(journal, {"type": "interest", "date": date, "amount": amount, "yield": round(cash_yield, 5)})
            book.cash += amount
    prices = closes.iloc[-1]
    rate = (cost_bps + slippage_bps) / 1e4
    equity = book.equity(prices)
    peak = max([m["equity"] for m in book.marks] + [equity])
    msg = [f"{date}: equity {equity:,.2f}"]

    if book.halted:
        msg.append(f"HALTED ({book.halted}); holding cash")
    elif equity < peak * (1.0 - max_dd):
        reason = f"drawdown {equity / peak - 1:.1%} breached -{max_dd:.0%} limit"
        msg += ["CIRCUIT BREAKER: " + reason] + _trade_to(book, journal, date, prices, pd.Series(dtype=float), rate, "halt")
        append_event(journal, {"type": "halt", "date": date, "reason": reason})
    elif book.rebalanced != date[:7] or (band is not None and _drift(book, prices, closes, rule) > band):
        weights = all_rules()[rule](closes, bars_per_year(closes.index)).iloc[-1]
        notes = _trade_to(book, journal, date, prices, weights, rate, "rebalance")
        append_event(journal, {"type": "rebalance", "date": date,
                               "weights": {a: round(float(w), 4) for a, w in weights.items()}})
        msg.append("rebalanced to " + ", ".join(f"{a} {w:.0%}" for a, w in weights.items()) +
                   (f" ({'; '.join(notes)})" if notes else " (no trades needed)"))

    equity = book.equity(prices)
    append_event(journal, {"type": "mark", "date": date, "equity": equity, "cash": book.cash,
                           "prices": {a: float(p) for a, p in prices.items()}})
    return " | ".join(msg)


def _drift(book: Book, prices: pd.Series, closes: pd.DataFrame, rule: str) -> float:
    """Largest gap between a holding's current weight and the rule's target today."""
    equity = book.equity(prices)
    if equity <= 0:
        return 0.0
    target = all_rules()[rule](closes, bars_per_year(closes.index)).iloc[-1]
    return max(abs(float(target.get(a, 0.0)) - book.shares.get(a, 0.0) * float(prices[a]) / equity)
               for a in prices.index)


def resume(journal: Path, note: str) -> None:
    append_event(journal, {"type": "resume", "note": note})


def report(journal: Path) -> dict:
    """Paper results so far, next to 60/40 of the first two assets (or buy & hold of a single one)."""
    events = read_events(journal)
    book = replay(events)
    if not book.marks:
        return {"status": "no data yet"}
    eq = pd.Series([m["equity"] for m in book.marks],
                   index=pd.to_datetime([m["date"] for m in book.marks]))
    px = pd.DataFrame([m["prices"] for m in book.marks], index=eq.index)
    rets = px.pct_change().fillna(0.0)
    if px.shape[1] >= 2:
        a, b = px.columns[:2]
        bench, bench_name = (1.0 + 0.6 * rets[a] + 0.4 * rets[b]).cumprod(), "60/40"
    else:
        a = px.columns[0]
        bench, bench_name = (1.0 + rets[a]).cumprod(), f"{a} buy & hold"
    start = events[0]["cash"]
    last = book.marks[-1]
    return {
        "status": f"HALTED: {book.halted}" if book.halted else "running",
        "since": str(eq.index[0].date()), "as_of": str(eq.index[-1].date()), "days": len(eq),
        "equity": round(eq.iloc[-1], 2),
        "return": eq.iloc[-1] / start - 1.0,
        "benchmark": float(bench.iloc[-1] - 1.0),
        "benchmark_name": bench_name,
        "max_drawdown": float((eq / eq.cummax().clip(lower=start) - 1.0).min()),
        "holdings": {k: round(q * last["prices"][k] / last["equity"], 3)
                     for k, q in book.shares.items() if abs(q) > 1e-9},
        "cash_weight": round(last["cash"] / last["equity"], 3),
        "fills": book.fills,
    }
