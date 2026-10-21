"""Export everything the dashboard page shows as one JSON file (SPY account first).

    python -m trading_lab dashboard
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .metrics import consistency, summary
from .paper import read_events
from .paper_portfolio import report
from .readiness import check
from .spy_research import COST_BPS, SLIPPAGE_BPS, SPLIT, facts, load_spy, vol_target

SPY_JOURNAL = Path("paper/spy.jsonl")
OBSERVATIONS = Path("paper/spy_observations.csv")

# Side accounts shown under the SPY account: title, benchmark label, safety stop.
EXTRA_ACCOUNTS = [
    {"journal": Path("paper/etf.jsonl"), "title": "Three-asset portfolio ($100,000: SPY, bonds, gold)",
     "bench": "60/40 stocks/bonds", "stop": "−15%"},
]


def _rounded(rep: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rep.items()}


def _backtest(spy: pd.DataFrame, since: str = SPLIT) -> dict:
    """The live rule against simply holding SPY, over the out-of-sample years."""
    runs = {
        "Just holding SPY": run_backtest(spy.close, pd.Series(1.0, index=spy.index), COST_BPS, SLIPPAGE_BPS),
        "Volatility control (this account's rule)": run_backtest(spy.close, vol_target(spy), COST_BPS, SLIPPAGE_BPS),
    }
    curves, stats, dates = {}, {}, []
    for name, res in runs.items():
        r = res.returns.loc[since:]
        monthly = (1.0 + r).cumprod().resample("ME").last()  # keep the page small
        curves[name] = [round(float(v), 4) for v in monthly.values]
        dates = [d.strftime("%Y-%m") for d in monthly.index]
        stats[name] = {**{k: round(float(v), 4) for k, v in summary(r, 252).items()},
                       **{k: round(float(v), 4) for k, v in consistency(r, 252).items()}}
    return {"since": dates[0], "dates": dates, "curves": curves, "stats": stats, "ours": list(runs)[1]}


def _today() -> dict | None:
    if not OBSERVATIONS.exists():
        return None
    row = pd.read_csv(OBSERVATIONS).iloc[-1]
    return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()}


def build(journal: Path = SPY_JOURNAL, extra: list[dict] | None = None) -> dict:
    spy = load_spy()
    events = read_events(journal)
    return {
        "generated": str(date.today()),
        "max_dd_limit": 0.25,
        "paper": _rounded(report(journal)),
        "marks": [{"date": e["date"], "equity": round(e["equity"], 2)} for e in events if e["type"] == "mark"],
        "activity": [e for e in events if e["type"] in ("rebalance", "fill", "halt", "resume")][-40:],
        "backtest": _backtest(spy),
        "facts": facts(spy),
        "today": _today(),
        "readiness": [{"name": n, "ok": bool(ok), "detail": d} for n, ok, d in check(journal, spy)],
        "extra": [{**{k: v for k, v in a.items() if k != "journal"}, "rep": _rounded(report(a["journal"]))}
                  for a in (EXTRA_ACCOUNTS if extra is None else extra) if a["journal"].exists()],
    }


def write(out: Path, journal: Path = SPY_JOURNAL) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(journal), indent=1))
