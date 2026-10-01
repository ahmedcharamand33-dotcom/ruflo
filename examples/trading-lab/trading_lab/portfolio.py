"""Multi-asset portfolios: the route to *consistent* returns without leverage.

A single asset's return stream is lumpy no matter how clever the signal.
Combining assets that fall at different times (stocks, bonds, gold), sizing
each by its risk, and stepping aside when an asset's trend breaks is the
best-documented way to smooth the ride.

Every rule here keeps total exposure <= 100% (no leverage). Weights not
allocated sit in cash, which earns 0% in these tests (a conservative choice).

The rules use textbook parameters (10-month trend, 12-month volatility) that
were fixed in advance, not tuned here. They were popularised around 2007
(Faber, "A Quantitative Approach to Tactical Asset Allocation"), so 2008
onwards is a genuine after-publication test.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .strategies import months_to_bars

Rule = Callable[[pd.DataFrame, int], pd.DataFrame]


def _fixed(weights: dict[str, float]) -> Rule:
    def rule(closes, bpy):
        return pd.DataFrame({c: weights.get(c, 0.0) for c in closes.columns}, index=closes.index)
    return rule


def equal_weight(closes, bpy):
    return pd.DataFrame(1.0 / closes.shape[1], index=closes.index, columns=closes.columns)


def inverse_vol(closes, bpy):
    """Risk parity (naive): each asset gets weight proportional to 1 / its recent volatility."""
    vol = closes.pct_change().rolling(months_to_bars(12, bpy)).std()
    inv = 1.0 / vol
    return inv.div(inv.sum(axis=1), axis=0).fillna(0.0)


def uptrend(closes, bpy):
    """1 where an asset is above its 10-month average, else 0."""
    sma = closes.rolling(months_to_bars(10, bpy)).mean()
    return (closes > sma).astype(float).where(sma.notna(), 0.0)


def with_trend(base: Rule) -> Rule:
    """Hold each asset only while it is in an uptrend; its share goes to cash otherwise."""
    def rule(closes, bpy):
        return base(closes, bpy) * uptrend(closes, bpy)
    return rule


def benchmarks(columns) -> dict[str, Rule]:
    first = columns[0]
    out = {f"{first} only": _fixed({first: 1.0})}
    if len(columns) >= 2:
        out["60/40"] = _fixed({columns[0]: 0.6, columns[1]: 0.4})
    return out


def candidates() -> dict[str, Rule]:
    """Every rule we consider. Their count is charged against us by the deflated Sharpe."""
    return {
        "equal weight": equal_weight,
        "risk parity": inverse_vol,
        "equal weight + trend": with_trend(equal_weight),
        "risk parity + trend": with_trend(inverse_vol),
    }


def run_portfolio(closes: pd.DataFrame, weights: pd.DataFrame, cost_bps: float = 5.0,
                  slippage_bps: float = 5.0, execution_lag: int = 1) -> BacktestResult:
    """Rebalance to target weights every bar. ``held`` reports gross exposure (0..1)."""
    if execution_lag < 1:
        raise ValueError("execution_lag must be >= 1 (lag 0 is look-ahead)")
    weights = weights.reindex(closes.index).fillna(0.0)
    gross = weights.sum(axis=1)
    if (gross > 1.0 + 1e-9).any() or (weights < -1e-12).any().any():
        raise ValueError("weights must be long-only and sum to <= 1 (no leverage)")
    rets = closes.pct_change().fillna(0.0)
    held = weights.shift(execution_lag).fillna(0.0)
    traded = held.diff().abs().sum(axis=1)
    traded.iloc[0] = held.iloc[0].abs().sum()
    costs = traded * (cost_bps + slippage_bps) / 1e4
    net = (held * rets).sum(axis=1) - costs
    return BacktestResult(returns=net, held=held.sum(axis=1), costs=costs, trades=[])


@dataclass
class PortfolioRun:
    name: str
    result: BacktestResult
    weights: pd.DataFrame


def run_all(closes: pd.DataFrame, bpy: int, **kw) -> dict[str, PortfolioRun]:
    rules = {**benchmarks(list(closes.columns)), **candidates()}
    out = {}
    for name, rule in rules.items():
        w = rule(closes, bpy)
        out[name] = PortfolioRun(name, run_portfolio(closes, w, **kw), w)
    return out


def current_allocation(closes: pd.DataFrame, rule: Rule, bpy: int) -> pd.Series:
    """Target weights for the latest bar, plus cash."""
    w = rule(closes, bpy).iloc[-1]
    return pd.concat([w, pd.Series({"cash": max(0.0, 1.0 - w.sum())})]).round(4)


def warmup_bars(bpy: int) -> int:
    return int(np.ceil(months_to_bars(12, bpy))) + 1
