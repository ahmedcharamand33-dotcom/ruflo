"""Walk-forward selection: the system's learning loop.

At the start of every test window the system looks back over a rolling
training window, scores each candidate strategy on what *already happened*,
and trades the winner over the next window, which it has never seen. Only
these out-of-sample windows are stitched together and graded, so the
reported performance is what you would have experienced live (minus
execution surprises).

This is how the bot "learns from its mistakes": approaches that stop
working lose their rank and stop being traded. It cannot learn from the
future, and it cannot memorise the past, which is the point.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .data import bars_per_year as infer_bpy
from .metrics import sharpe
from .strategies import BuyAndHold, Strategy


@dataclass
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: str
    train_sharpe: float


@dataclass
class WalkForwardResult:
    strategy: BacktestResult        # out-of-sample only
    benchmark: BacktestResult       # buy-and-hold over the same window
    folds: list[Fold]
    n_trials: int                   # candidates tried (the deflation charge)
    bars_per_year: int
    targets: dict[str, pd.Series]


def _slice(res: BacktestResult, start: int) -> BacktestResult:
    idx = res.returns.index[start:]
    trades = [t for t in res.trades if t.entry >= idx[0]]
    return BacktestResult(res.returns.iloc[start:], res.held.iloc[start:], res.costs.iloc[start:], trades)


def select(standalone: dict[str, BacktestResult], end: int, train: int, bpy: int) -> tuple[str, float]:
    """Best candidate by Sharpe over bars [end-train, end). Ties go to the earliest listed."""
    scores = {lbl: sharpe(bt.returns.iloc[end - train:end], bpy) for lbl, bt in standalone.items()}
    best = max(scores, key=scores.get)
    return best, scores[best]


def walk_forward(close: pd.Series, candidates: list[Strategy], train_years: float = 10,
                 test_years: float = 1, cost_bps: float = 5.0, slippage_bps: float = 5.0,
                 execution_lag: int = 1, bpy: int | None = None) -> WalkForwardResult:
    bpy = bpy or infer_bpy(close.index)
    train, test = int(train_years * bpy), max(1, int(test_years * bpy))
    if len(close) <= train + execution_lag + test:
        raise ValueError(f"need more than {train + test} bars, have {len(close)}")
    kw = dict(cost_bps=cost_bps, slippage_bps=slippage_bps, execution_lag=execution_lag)

    targets = {c.label: c.positions(close, bpy) for c in candidates}
    standalone = {lbl: run_backtest(close, t, **kw) for lbl, t in targets.items()}

    stitched = pd.Series(0.0, index=close.index)
    folds: list[Fold] = []
    i = train
    while i < len(close):
        j = min(i + test, len(close))
        best, score = select(standalone, i, train, bpy)
        stitched.iloc[i:j] = targets[best].iloc[i:j].values
        folds.append(Fold(close.index[i], close.index[j - 1], best, score))
        i = j

    oos = train + execution_lag
    bench = run_backtest(close, BuyAndHold().positions(close, bpy), **kw)
    return WalkForwardResult(
        strategy=_slice(run_backtest(close, stitched, **kw), oos),
        benchmark=_slice(bench, oos),
        folds=folds,
        n_trials=len(candidates),
        bars_per_year=bpy,
        targets=targets,
    )
