"""Trading strategies.

A strategy maps a close-price series to a *target exposure* per bar, in
[-1, 1] (1 = fully long, 0 = cash). The exposure at bar t may only use data up
to and including bar t; the backtester decides when it is actually held.

Lookbacks are given in months and converted to bars, so the same parameter
grid means the same thing on daily and monthly data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def months_to_bars(months: float, bars_per_year: int) -> int:
    return max(1, int(round(months * bars_per_year / 12.0)))


@dataclass(frozen=True)
class Strategy:
    name: str = "base"
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"

    def positions(self, close: pd.Series, bars_per_year: int) -> pd.Series:
        raise NotImplementedError


@dataclass(frozen=True)
class BuyAndHold(Strategy):
    name: str = "buy_hold"

    def positions(self, close, bars_per_year):
        return pd.Series(1.0, index=close.index)


@dataclass(frozen=True)
class SmaTrend(Strategy):
    """Long while the fast moving average is above the slow one (fast=0: price vs SMA)."""
    name: str = "sma_trend"

    def positions(self, close, bars_per_year):
        slow = months_to_bars(self.params["slow"], bars_per_year)
        fast_m = self.params.get("fast", 0)
        fast = close if fast_m == 0 else close.rolling(months_to_bars(fast_m, bars_per_year)).mean()
        sma = close.rolling(slow).mean()
        return (fast > sma).astype(float).where(sma.notna(), 0.0)


@dataclass(frozen=True)
class Momentum(Strategy):
    """Time-series momentum: long when the trailing return is positive."""
    name: str = "momentum"

    def positions(self, close, bars_per_year):
        lb = months_to_bars(self.params["lookback"], bars_per_year)
        past = close.pct_change(lb)
        return (past > 0).astype(float).where(past.notna(), 0.0)


@dataclass(frozen=True)
class MeanReversion(Strategy):
    """Buy when price is stretched below its average, exit when it reverts."""
    name: str = "mean_rev"

    def positions(self, close, bars_per_year):
        lb = max(3, months_to_bars(self.params["lookback"], bars_per_year))
        mean = close.rolling(lb).mean()
        std = close.rolling(lb).std()
        z = (close - mean) / std
        state = np.where(z < -self.params["entry_z"], 1.0, np.where(z > 0, 0.0, np.nan))
        return pd.Series(state, index=close.index).ffill().fillna(0.0)


def default_candidates() -> list[Strategy]:
    """The small, fixed menu the walk-forward chooses from.

    Kept deliberately short: every extra candidate is another lottery ticket,
    and the deflated Sharpe ratio charges for each one.
    """
    cands: list[Strategy] = [BuyAndHold()]
    for slow in (6, 10, 12):
        cands.append(SmaTrend(params={"fast": 0, "slow": slow}))
    for lb in (3, 6, 12):
        cands.append(Momentum(params={"lookback": lb}))
    for lb, z in ((1, 1.0), (3, 1.5)):
        cands.append(MeanReversion(params={"lookback": lb, "entry_z": z}))
    return cands
