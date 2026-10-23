import json

import pytest

from trading_lab import alpaca_mirror as am


def test_no_order_when_within_threshold():
    assert am.plan_order(1.0, 100_000, 98_000, 700.0) is None


def test_buy_uses_notional_and_keeps_a_cash_buffer():
    o = am.plan_order(1.0, 100_000, 0.0, 700.0)
    assert o["side"] == "buy" and float(o["notional"]) == pytest.approx(99_500, abs=0.01)


def test_sell_never_exceeds_position_and_weight_is_capped():
    o = am.plan_order(0.0, 100_000, 50_000, 500.0)
    assert o == {"symbol": "SPY", "side": "sell", "qty": "100.000000"}
    assert float(am.plan_order(3.0, 100_000, 0.0, 700.0)["notional"]) <= 100_000  # no leverage


def test_slippage_sign():
    assert am.slippage_bps("buy", 101.0, 100.0) == 100.0     # paid more: cost
    assert am.slippage_bps("sell", 101.0, 100.0) == -100.0   # sold higher: gain


def _journal(tmp_path, events):
    p = tmp_path / "j.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events))
    return p


def test_halted_journal_targets_zero(tmp_path):
    j = _journal(tmp_path, [
        {"type": "rebalance", "date": "2026-10-20", "weights": {"SPY": 1.0}},
        {"type": "mark", "date": "2026-10-20", "equity": 1, "prices": {"SPY": 700.0}},
        {"type": "halt", "date": "2026-10-21", "reason": "stop"},
        {"type": "mark", "date": "2026-10-21", "equity": 1, "prices": {"SPY": 690.0}},
    ])
    assert am.journal_target(j) == (0.0, "2026-10-21", 690.0)


class FakeApi:
    def __init__(self, number="PA123"):
        self.number, self.submitted, self.orders = number, [], {}

    def account(self):
        return {"account_number": self.number, "equity": "100000"}

    def position_value(self):
        return 0.0

    def open_orders(self):
        return []

    def submit(self, order):
        self.submitted.append(order)
        return {"id": "ord-1"}

    def order(self, order_id):
        return {"id": order_id, "status": "filled", "filled_avg_price": "702.1", "filled_at": "2026-10-23T13:30:01Z",
                "side": "buy", "filled_qty": "141.7"}

    def day_open(self, date):
        return 702.0


def test_sync_submits_then_reconciles_fill(tmp_path):
    j = _journal(tmp_path, [
        {"type": "rebalance", "date": "2026-10-22", "weights": {"SPY": 1.0}},
        {"type": "mark", "date": "2026-10-22", "equity": 1, "prices": {"SPY": 700.0}},
    ])
    log, api = tmp_path / "a.jsonl", FakeApi()
    am.sync(j, log, api)
    assert api.submitted[0]["side"] == "buy"
    msg = am.sync(j, log, api)
    fill = [e for e in am.read_events(log) if e["type"] == "fill"][0]
    assert fill["vs_open_bps"] == pytest.approx(1.42, abs=0.01) and fill["vs_assumed_bps"] == pytest.approx(30.0)
    assert "filled buy" in msg


def test_refuses_a_non_paper_account(tmp_path):
    j = _journal(tmp_path, [{"type": "mark", "date": "2026-10-22", "equity": 1, "prices": {"SPY": 700.0}}])
    with pytest.raises(RuntimeError):
        am.sync(j, tmp_path / "a.jsonl", FakeApi(number="LIVE1"))
