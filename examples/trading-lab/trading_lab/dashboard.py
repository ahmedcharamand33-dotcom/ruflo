"""Export everything the dashboard page shows as one JSON file.

    python -m trading_lab dashboard --journal paper/etf.jsonl --msft-journal paper/msft.jsonl
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from . import portfolio as pf
from .data import bars_per_year, load_universe
from .metrics import consistency, summary
from .paper import read_events
from .paper_portfolio import DEFAULT_RULE, report

# The page colours holdings by asset class; map tickers onto those classes.
ASSET_CLASS = {"SPY": "stocks", "IEF": "bonds", "GLD": "gold"}


def _backtest(universe: str, since: str, lag: int) -> dict:
    closes = load_universe(universe)
    bpy = bars_per_year(closes.index)
    runs = pf.run_all(closes, bpy, execution_lag=lag)
    shown = {f"{closes.columns[0]} only": "stocks only", "60/40": "60/40", DEFAULT_RULE: DEFAULT_RULE}
    curves, stats, dates = {}, {}, []
    for name, key in shown.items():
        r = runs[name].result.returns.loc[since:]
        monthly = (1.0 + r).cumprod().resample("ME").last()  # keep the page small
        curves[key] = [round(float(v), 4) for v in monthly.values]
        dates = [d.strftime("%Y-%m") for d in monthly.index]
        stats[key] = {**{k: round(float(v), 4) for k, v in summary(r, bpy).items()},
                      **{k: round(float(v), 4) for k, v in consistency(r, bpy).items()}}
    return {"since": dates[0], "dates": dates, "curves": curves, "stats": stats}


def _rounded(rep: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in rep.items()}


def build(journal: Path, universe: str = "stock:SPY,IEF,GLD", since: str = "2008-01-01",
          lag: int = 1, msft_journal: Path | None = None) -> dict:
    events = read_events(journal)
    rep = _rounded(report(journal))
    if "holdings" in rep:
        rep["holdings"] = {ASSET_CLASS.get(k, k): v for k, v in rep["holdings"].items()}
    out = {
        "generated": str(date.today()),
        "rule": DEFAULT_RULE,
        "max_dd_limit": 0.15,
        "paper": rep,
        "marks": [{"date": e["date"], "equity": round(e["equity"], 2)} for e in events if e["type"] == "mark"],
        "activity": [e for e in events if e["type"] in ("rebalance", "fill", "halt", "resume")][-40:],
        "backtest": _backtest(universe, since, lag),
    }
    if msft_journal is not None and msft_journal.exists():
        out["msft"] = _rounded(report(msft_journal))
    return out


def write(journal: Path, out: Path, msft_journal: Path | None = None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(journal, msft_journal=msft_journal), indent=1))
