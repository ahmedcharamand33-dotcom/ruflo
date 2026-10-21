import numpy as np
import pandas as pd

from trading_lab import intraday


def _session(day="2024-03-04", seed=0, n=390):
    rng = np.random.default_rng(seed)
    close = 500 * np.exp(np.cumsum(rng.normal(0, 4e-4, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range(f"{day} 09:30", periods=n, freq="1min")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.02,
                         "low": np.minimum(open_, close) - 0.02, "close": close,
                         "volume": rng.integers(1e4, 1e5, n).astype(float)}, index=idx)


def test_rules_never_use_future_bars():
    d = _session()
    o, h, l, c, v = (d[k].values for k in ("open", "high", "low", "close", "volume"))
    sigma = np.full(390, 0.002)
    for name, rule in intraday.RULES.items():
        full = rule(o, h, l, c, v, 499.0, {"sigma": sigma})
        for k in (40, 200, 370):
            c2, h2, l2 = c.copy(), h.copy(), l.copy()
            c2[k:] *= 1.05; h2[k:] *= 1.05; l2[k:] *= 1.05  # change only the future
            later = rule(o, h2, l2, c2, v, 499.0, {"sigma": sigma})
            assert np.array_equal(full[:k + 1], later[:k + 1]), (name, k)


def test_flat_at_close_and_never_above_1x():
    d = _session()
    o, h, l, c, v = (d[k].values for k in ("open", "high", "low", "close", "volume"))
    for rule in intraday.RULES.values():
        pos = rule(o, h, l, c, v, 499.0, {"sigma": np.full(390, 0.002)})
        assert np.abs(pos).max() <= 1.0


def test_costs_make_a_flat_market_lose():
    d = pd.concat([_session("2024-03-04", 1), _session("2024-03-05", 2)])
    for col in ("open", "high", "low", "close"):
        d[col] = 500.0
    r = intraday.daily_returns(d, cost_bps=2.0)
    assert (r.drop(columns="hold during the day") <= 1e-12).all().all()
    assert np.isclose(r["hold during the day"].iloc[0], -4e-4)
