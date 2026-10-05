"""Performance metrics, including statistics that correct for luck.

probabilistic_sharpe  P(true Sharpe > benchmark) given sample length,
                      skew and fat tails (Bailey & Lopez de Prado, 2012).
deflated_sharpe       Same, but the benchmark is the Sharpe you'd expect
                      from the *best of N* useless strategies (2014). This is
                      the number that tells you whether you found an edge or
                      just tried enough things.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649


def sharpe(returns: pd.Series, bars_per_year: int) -> float:
    sd = returns.std()
    return 0.0 if not sd or np.isnan(sd) else float(returns.mean() / sd * math.sqrt(bars_per_year))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def summary(returns: pd.Series, bars_per_year: int, held: pd.Series | None = None,
            n_trades: int | None = None) -> dict:
    n = len(returns)
    growth = float((1.0 + returns).prod())
    years = n / bars_per_year
    downside = returns[returns < 0].std()
    out = {
        "years": round(years, 1),
        "cagr": growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else -1.0,
        "vol": float(returns.std() * math.sqrt(bars_per_year)),
        "sharpe": sharpe(returns, bars_per_year),
        "sortino": float(returns.mean() / downside * math.sqrt(bars_per_year)) if downside else 0.0,
        "max_dd": max_drawdown(returns),
    }
    out["calmar"] = out["cagr"] / abs(out["max_dd"]) if out["max_dd"] < 0 else 0.0
    if held is not None:
        out["exposure"] = float((held.abs() > 1e-12).mean())
        out["turnover_yr"] = float(held.diff().abs().sum() / years) if years else 0.0
    if n_trades is not None:
        out["trades"] = n_trades
    return out


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Inputs are per-bar (not annualised) Sharpe ratios."""
    n = len(returns)
    sd = returns.std()
    if n < 3 or not sd:
        return 0.0
    sr = returns.mean() / sd
    skew = float(returns.skew())
    kurt = float(returns.kurt()) + 3.0  # pandas gives excess kurtosis
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return 0.0
    return _N.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, n_obs: int) -> float:
    """Per-bar Sharpe that the best of N skill-less trials would show by chance.

    Uses the sampling variance of a Sharpe estimate, 1/(n_obs-1), as the
    spread of the trials (they share one asset, so their noise is if anything
    smaller). Estimating it from the trials themselves would mistake real
    differences between strategies for luck.
    """
    if n_trials < 2 or n_obs < 3:
        return 0.0
    sd = math.sqrt(1.0 / (n_obs - 1))
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return sd * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_benchmark: float = 0.0) -> float:
    """P(true per-bar Sharpe > benchmark), after charging for picking the best of n_trials."""
    return probabilistic_sharpe(returns, sr_benchmark + expected_max_sharpe(n_trials, len(returns)))


def consistency(returns: pd.Series, bars_per_year: int) -> dict:
    """How smooth the ride was: the numbers that decide whether you stick with a strategy."""
    equity = (1.0 + returns).cumprod()
    yearly = (1.0 + returns).groupby(returns.index.year).prod() - 1.0
    rolling = equity.pct_change(bars_per_year).dropna()
    under = equity < equity.cummax()
    runs = under.groupby((~under).cumsum()).sum()
    return {
        "pos_years": float((yearly > 0).mean()),
        "worst_year": float(yearly.min()),
        "pos_12m": float((rolling > 0).mean()) if len(rolling) else 0.0,
        "underwater_yrs": float(runs.max() / bars_per_year) if len(runs) else 0.0,
    }
