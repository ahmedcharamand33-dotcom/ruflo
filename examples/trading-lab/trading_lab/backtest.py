"""Vectorised single-asset backtester.

Timing convention (the most common source of fake profits):
    exposure decided at the close of bar t is held from bar t+lag onward.
    lag=1 assumes you can trade at the same close the signal was computed on
    (market-on-close). lag=2 is the conservative choice for averaged data.

Costs are charged on every change in held exposure: commission + slippage
in basis points of traded notional.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    entry: pd.Timestamp
    exit: pd.Timestamp
    side: float
    bars: int
    ret: float


@dataclass
class BacktestResult:
    returns: pd.Series      # net per-bar strategy returns
    held: pd.Series         # exposure actually held each bar
    costs: pd.Series        # per-bar cost drag
    trades: list[Trade]

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def run_backtest(close: pd.Series, target: pd.Series, cost_bps: float = 5.0,
                 slippage_bps: float = 5.0, execution_lag: int = 1) -> BacktestResult:
    if execution_lag < 1:
        raise ValueError("execution_lag must be >= 1 (lag 0 is look-ahead)")
    asset_ret = close.pct_change().fillna(0.0)
    target = target.reindex(close.index).fillna(0.0).clip(-1.0, 1.0)
    held = target.shift(execution_lag).fillna(0.0)
    costs = held.diff().abs().fillna(held.abs()) * (cost_bps + slippage_bps) / 1e4
    net = held * asset_ret - costs
    return BacktestResult(returns=net, held=held, costs=costs, trades=extract_trades(held, net))


def extract_trades(held: pd.Series, net: pd.Series) -> list[Trade]:
    """Round trips: a trade opens when exposure leaves 0 (or flips sign) and closes when it returns."""
    trades: list[Trade] = []
    side, start, growth, bars = 0.0, None, 1.0, 0
    sign = held.apply(lambda x: 0.0 if abs(x) < 1e-12 else (1.0 if x > 0 else -1.0))
    for ts, s, r in zip(held.index, sign.values, net.values):
        if side != 0.0 and s != side:
            trades.append(Trade(start, ts, side, bars, growth - 1.0))
            side, growth, bars = 0.0, 1.0, 0
        if s != 0.0 and side == 0.0:
            side, start = s, ts
        if side != 0.0:
            growth *= 1.0 + r
            bars += 1
    if side != 0.0:
        trades.append(Trade(start, held.index[-1], side, bars, growth - 1.0))
    return trades
