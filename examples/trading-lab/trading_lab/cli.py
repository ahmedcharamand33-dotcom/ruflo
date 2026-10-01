"""Command line: python -m trading_lab <command> [options]

    evaluate     walk-forward backtest + scorecard (is there an edge?)
    paper-step   process the latest bar into the paper-trading journal
    review       compare paper results with backtest expectations (kill switch)
    portfolio    multi-asset test (stocks/bonds/gold) + today's target allocation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import bars_per_year, load, load_universe
from .paper import review, step
from . import portfolio as pf
from .scorecard import portfolio_table, render, scorecard
from .strategies import default_candidates
from .walkforward import walk_forward


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="shiller", help="shiller | csv:<path> | yahoo:<TICKER> | synthetic[:seed]")
    p.add_argument("--train-years", type=float, default=10)
    p.add_argument("--test-years", type=float, default=1)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trading_lab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    _common(ev)
    ev.add_argument("--lag", type=int, default=1, help="bars between signal and holding (use 2 for shiller)")
    for name in ("paper-step", "review"):
        p = sub.add_parser(name)
        _common(p)
        p.add_argument("--journal", type=Path, required=True)
        p.add_argument("--cash", type=float, default=100_000.0)
    po = sub.add_parser("portfolio")
    po.add_argument("--universe", default="history", help="history | yahoo:SPY,IEF,GLD")
    po.add_argument("--lag", type=int, default=2)
    po.add_argument("--rule", default="equal weight + trend", choices=list(pf.candidates()))
    args = ap.parse_args(argv)

    if args.cmd == "portfolio":
        return _portfolio(args)
    close = load(args.source)["close"]
    cost = dict(cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)
    window = dict(train_years=args.train_years, test_years=args.test_years)

    if args.cmd == "evaluate":
        wf = walk_forward(close, default_candidates(), execution_lag=args.lag, **window, **cost)
        print(render(scorecard(wf), f"{args.source} (lag {args.lag})"))
    elif args.cmd == "paper-step":
        print(step(close, args.journal, default_candidates(), cash=args.cash, **window, **cost))
    else:
        wf = walk_forward(close, default_candidates(), **window, **cost)
        for k, v in review(args.journal, wf.strategy.returns).items():
            print(f"{k:>14}: {v}")
    return 0


def _portfolio(args: argparse.Namespace) -> int:
    closes = load_universe(args.universe)
    bpy = bars_per_year(closes.index)
    runs = pf.run_all(closes, bpy, execution_lag=args.lag)
    start = closes.index[pf.warmup_bars(bpy) + args.lag]
    n = len(pf.candidates())
    periods = [(str(start.date()), None)] + [(p, None) for p in ("1972-01-01", "2008-01-01")
                                              if closes.index[0] < pd.Timestamp(p)]
    for a, b in periods:
        print(portfolio_table(runs, bpy, a, b, n_trials=n) + "\n")
    alloc = pf.current_allocation(closes, pf.candidates()[args.rule], bpy)
    print(f"Target allocation today ({args.rule}, data to {closes.index[-1].date()}):")
    for asset, w in alloc.items():
        print(f"  {asset:8}{w:>7.1%}")
    return 0
