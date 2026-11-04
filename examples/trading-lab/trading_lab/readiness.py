"""Go-live gates for trading SPY with real money.

The measurable gates are computed from the paper journal and SPY history, so
the answer cannot be talked into a "yes". The personal gates are listed for the
owner; no code can check them.

    python -m trading_lab readiness
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .paper import read_events
from .spy_research import COST_BPS, SLIPPAGE_BPS, SPLIT, LIVE_SPY_SIGNAL, cash_rate, load_spy

MIN_TRADING_DAYS = 126      # about six months
MIN_COVERAGE = 0.95         # share of trading days the bot actually ran
MAX_SHORTFALL_Z = -2.0      # live returns this far below the backtest = something is wrong

PERSONAL_GATES = [
    "You have written down what you will do if the account falls 20% (the answer should be: nothing).",
    "The money is not needed for at least 5 years.",
    "You start with at most a quarter of the amount you eventually intend, for the first 3 months.",
    "You have a low-cost broker account that allows fractional shares or the rule's share counts.",
    "You know short-term gains are taxed as income, and have asked how that applies to you.",
]


ANSWERS = Path("paper/personal_gates.json")  # the owner's own words, keyed by gate number


def answers(path: Path = ANSWERS) -> dict[int, dict]:
    return {int(k): v for k, v in json.loads(path.read_text()).items()} if path.exists() else {}


def _marks(journal: Path) -> pd.Series:
    marks = [e for e in read_events(journal) if e["type"] == "mark"]
    return pd.Series([m["equity"] for m in marks], index=pd.to_datetime([m["date"] for m in marks]))


def check(journal: Path = Path("paper/spy.jsonl"), spy: pd.DataFrame | None = None) -> list[tuple[str, bool, str]]:
    spy = spy if spy is not None else load_spy()
    equity = _marks(journal)
    if equity.empty:
        return [("Paper account exists", False, "no paper results yet")]
    trading_days = spy.loc[equity.index[0]:equity.index[-1]].index
    days = len(equity)
    coverage = days / max(1, len(trading_days))

    expected = run_backtest(spy.close, LIVE_SPY_SIGNAL(spy), COST_BPS, SLIPPAGE_BPS,
                            cash_rate=cash_rate(spy.index)).returns.loc[SPLIT:]
    live = equity.pct_change().dropna()
    if len(live) >= 20 and expected.std() > 0:
        z = (live.mean() - expected.mean()) / (expected.std() / math.sqrt(len(live)))
        tracking = (z >= MAX_SHORTFALL_Z, f"live vs expected z = {z:+.1f} (needs above {MAX_SHORTFALL_Z:+.0f})")
    else:
        tracking = (False, f"only {len(live)} days of returns; need 20+ to judge")
    halts = [e for e in read_events(journal) if e["type"] == "halt"]
    return [
        (f"At least {MIN_TRADING_DAYS} trading days of paper results", days >= MIN_TRADING_DAYS,
         f"{days} so far"),
        (f"Bot ran on at least {MIN_COVERAGE:.0%} of trading days", coverage >= MIN_COVERAGE,
         f"{coverage:.0%} of {len(trading_days)} days"),
        ("Paper results are in line with the tested rule", *tracking),
        ("No safety-stop event left unexplained", not halts,
         f"{len(halts)} safety stops" if halts else "none"),
    ]


def render(gates: list[tuple[str, bool, str]], answered: dict[int, dict] | None = None) -> str:
    answered = answers() if answered is None else answered
    out = ["== Real-money readiness (SPY) =="]
    out += [f"  [{'PASS' if ok else 'NOT YET'}] {name}: {detail}" for name, ok, detail in gates]
    out.append("Personal gates (only you can check these):")
    for i, g in enumerate(PERSONAL_GATES):
        a = answered.get(i)
        out.append(f"  [x] {g}\n      {a['date']}: \"{a['answer']}\"" if a else f"  [ ] {g}")
    verdict = "READY to start small" if all(ok for _, ok, _ in gates) else "NOT READY: keep paper trading"
    out.append(f"VERDICT: {verdict}")
    return "\n".join(out)
