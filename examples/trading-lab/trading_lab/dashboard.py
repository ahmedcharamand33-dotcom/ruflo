"""Export everything the dashboard page shows as one JSON file.

    python -m trading_lab dashboard --journal paper/history.jsonl --out dashboard/data.json
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import portfolio as pf
from .data import load_history
from .metrics import consistency, summary
from .paper import read_events
from .paper_portfolio import DEFAULT_RULE, report

SHOWN = ("stocks only", "60/40", DEFAULT_RULE)


def build(journal: Path, since: str = "1972-01-01", lag: int = 2) -> dict:
    closes = load_history()
    runs = pf.run_all(closes, 12, execution_lag=lag)
    curves, stats = {}, {}
    for name in SHOWN:
        r = runs[name].result.returns.loc[since:]
        eq = (1.0 + r).cumprod()
        curves[name] = [round(float(v), 4) for v in eq.values]
        stats[name] = {**{k: round(float(v), 4) for k, v in summary(r, 12).items()},
                       **{k: round(float(v), 4) for k, v in consistency(r, 12).items()}}
    dates = [d.strftime("%Y-%m") for d in runs[SHOWN[0]].result.returns.loc[since:].index]

    events = read_events(journal)
    activity = [e for e in events if e["type"] in ("rebalance", "fill", "halt", "resume")]
    marks = [{"date": e["date"], "equity": round(e["equity"], 2)} for e in events if e["type"] == "mark"]
    rep = report(journal)
    rep = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rep.items()}
    return {
        "generated": str(date.today()),
        "rule": DEFAULT_RULE,
        "max_dd_limit": 0.15,
        "paper": rep,
        "marks": marks,
        "activity": activity[-40:],
        "backtest": {"since": dates[0], "dates": dates, "curves": curves, "stats": stats},
    }


def write(journal: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(journal), indent=1))
