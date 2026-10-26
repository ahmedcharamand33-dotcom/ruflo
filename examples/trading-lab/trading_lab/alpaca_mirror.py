"""Mirror the SPY paper journal into Alpaca's paper account, and measure real fills.

Each evening the journal's target SPY weight is applied to Alpaca's own equity.
If the account is off by more than ``THRESHOLD`` of equity, a market order is
queued for the next open. The next evening each filled order is compared with
(a) that morning's official open (pure trading cost) and (b) the close the
journal assumed (the full real-world gap, overnight move included).

PAPER ONLY: the base URL is fixed to Alpaca's paper endpoint. Credentials are
added by the network proxy, or read from APCA_API_KEY_ID / APCA_API_SECRET_KEY.

    python -m trading_lab alpaca-sync
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .paper import append_event, read_events

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
THRESHOLD = 0.02      # rebalance only when the account is >2% of equity off target
CASH_BUFFER = 0.995   # never ask for 100% of equity: a gap up at the open would be refused
LOG = Path("paper/alpaca.jsonl")
SYMBOL = "SPY"


def plan_order(weight: float, equity: float, position_value: float, price: float,
               threshold: float = THRESHOLD) -> dict | None:
    """Order that moves the account to ``weight`` of equity, or None if close enough."""
    target = max(0.0, min(1.0, weight)) * equity * CASH_BUFFER
    gap = target - position_value
    if abs(gap) <= threshold * equity:
        return None
    if gap > 0:
        return {"symbol": SYMBOL, "side": "buy", "notional": f"{gap:.2f}"}
    qty = min(position_value, -gap) / price
    return {"symbol": SYMBOL, "side": "sell", "qty": f"{qty:.6f}"}


def slippage_bps(side: str, fill: float, reference: float) -> float:
    """Cost of a fill against a reference price, positive = paid more than the reference."""
    sign = 1.0 if side == "buy" else -1.0
    return round(sign * (fill / reference - 1.0) * 1e4, 2)


def journal_target(journal: Path) -> tuple[float, str, float] | None:
    """(SPY weight, date, assumed close) from the journal's latest rebalance and mark."""
    events = read_events(journal)
    rebal = [e for e in events if e["type"] == "rebalance"]
    marks = [e for e in events if e["type"] == "mark"]
    halted = any(e["type"] == "halt" for e in events) and not (
        events and [e for e in events if e["type"] in ("halt", "resume")][-1]["type"] == "resume")
    if not marks:
        return None
    weight = 0.0 if halted or not rebal else float(rebal[-1]["weights"].get(SYMBOL, 0.0))
    return weight, marks[-1]["date"], float(marks[-1]["prices"][SYMBOL])


class Alpaca:
    def __init__(self):
        import requests

        self.s = requests.Session()
        for header, var in (("APCA-API-KEY-ID", "APCA_API_KEY_ID"), ("APCA-API-SECRET-KEY", "APCA_API_SECRET_KEY")):
            if var in os.environ:
                self.s.headers[header] = os.environ[var]

    def _get(self, url, **params):
        r = self.s.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def account(self) -> dict:
        return self._get(f"{PAPER_URL}/v2/account")

    def position_value(self) -> float:
        for p in self._get(f"{PAPER_URL}/v2/positions"):
            if p["symbol"] == SYMBOL:
                return float(p["market_value"])
        return 0.0

    def open_orders(self) -> list:
        return self._get(f"{PAPER_URL}/v2/orders", status="open", symbols=SYMBOL)

    def order(self, order_id: str) -> dict:
        return self._get(f"{PAPER_URL}/v2/orders/{order_id}")

    def day_open(self, date: str) -> float | None:
        import pandas as pd

        # the free data plan refuses the most recent 15 minutes, so never ask past now - 20 min
        end = min(pd.Timestamp(date, tz="UTC") + pd.Timedelta(days=1), pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20))
        bars = self._get(f"{DATA_URL}/v2/stocks/{SYMBOL}/bars", timeframe="1Day", start=date,
                         end=end.strftime("%Y-%m-%dT%H:%M:%SZ"), feed="sip", adjustment="raw").get("bars") or []
        return float(bars[0]["o"]) if bars else None

    def submit(self, order: dict) -> dict:
        body = {**order, "type": "market", "time_in_force": "day"}
        r = self.s.post(f"{PAPER_URL}/v2/orders", data=json.dumps(body), timeout=20)
        r.raise_for_status()
        return r.json()


def _reconcile(api: Alpaca, log: Path) -> list[str]:
    """Record fills for orders submitted earlier and not yet reconciled."""
    events = read_events(log)
    done = {e["order_id"] for e in events if e["type"] in ("fill", "unfilled")}
    notes = []
    for sub in (e for e in events if e["type"] == "submit" and e["order_id"] not in done):
        o = api.order(sub["order_id"])
        if o["status"] == "filled":
            price, day = float(o["filled_avg_price"]), o["filled_at"][:10]
            day_open = api.day_open(day)
            ev = {"type": "fill", "order_id": o["id"], "date": day, "side": o["side"],
                  "qty": float(o["filled_qty"]), "price": price, "assumed_close": sub["assumed_close"],
                  "vs_assumed_bps": slippage_bps(o["side"], price, sub["assumed_close"]),
                  "open": day_open, "vs_open_bps": slippage_bps(o["side"], price, day_open) if day_open else None}
            append_event(log, ev)
            notes.append(f"filled {o['side']} {ev['qty']:.4f} SPY at {price:.2f} "
                         f"({ev['vs_open_bps']:+.1f} bp vs open, {ev['vs_assumed_bps']:+.1f} bp vs assumed close)")
        elif o["status"] in ("canceled", "expired", "rejected", "done_for_day"):
            append_event(log, {"type": "unfilled", "order_id": o["id"], "status": o["status"]})
            notes.append(f"order {o['id'][:8]} ended {o['status']} without a fill")
    return notes


def sync(journal: Path = Path("paper/spy.jsonl"), log: Path = LOG, api: Alpaca | None = None) -> str:
    api = api or Alpaca()
    notes = _reconcile(api, log)
    target = journal_target(journal)
    if target is None:
        return "alpaca: journal has no marks yet"
    weight, date, close = target
    acct = api.account()
    if not str(acct.get("account_number", "")).startswith("PA"):  # Alpaca paper accounts are numbered PA...
        raise RuntimeError("refusing to trade: this is not an Alpaca paper account")
    equity, held = float(acct["equity"]), api.position_value()
    if api.open_orders():
        notes.append("an order is already waiting for the open; nothing new submitted")
    else:
        order = plan_order(weight, equity, held, close)
        if order:
            placed = api.submit(order)
            append_event(log, {"type": "submit", "order_id": placed["id"], "date": date, **order,
                               "target_weight": weight, "assumed_close": close})
            notes.append(f"queued {order['side']} {order.get('notional') or order.get('qty')} SPY for the next open "
                         f"(target {weight:.0%} of ${equity:,.0f})")
    append_event(log, {"type": "mark", "date": date, "equity": equity, "spy_value": held, "target_weight": weight})
    have = held / equity if equity else 0.0
    return f"alpaca {date}: equity ${equity:,.2f}, SPY {have:.0%} (target {weight:.0%}). " + "; ".join(notes)
