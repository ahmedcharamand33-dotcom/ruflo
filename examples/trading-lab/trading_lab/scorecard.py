"""Turn a walk-forward run into a verdict a human can act on.

    EDGE        Better risk-adjusted returns than buy-and-hold, survives the
                multiple-testing haircut, AND the excess return is significant.
    RISK EDGE   Better risk-adjusted and survives the haircut, but does not
                make provably more money than just holding. Often still useful.
    NO EDGE     Keep researching. Do not trade this with real money.
"""
from __future__ import annotations

from collections import Counter

from .metrics import consistency, deflated_sharpe, sharpe, summary
from .walkforward import WalkForwardResult

CONFIDENCE = 0.95
MIN_TRADES = 20


def scorecard(wf: WalkForwardResult) -> dict:
    bpy = wf.bars_per_year
    s, b = wf.strategy, wf.benchmark
    strat = summary(s.returns, bpy, s.held, len(s.trades))
    bench = summary(b.returns, bpy, b.held)
    n_trials = wf.n_trials
    # Hurdle: buy-and-hold's Sharpe plus what the luckiest of N tries would add by chance.
    dsr = deflated_sharpe(s.returns, n_trials, sharpe(b.returns, 1))
    dsr_active = deflated_sharpe(s.returns - b.returns, n_trials)

    checks = [
        ("Out-of-sample Sharpe beats buy-and-hold", strat["sharpe"] > bench["sharpe"]),
        (f"P(Sharpe beats buy&hold, deflated for {n_trials} trials) >= {CONFIDENCE:.0%}",
         dsr >= CONFIDENCE),
        ("Max drawdown no worse than buy-and-hold", strat["max_dd"] >= bench["max_dd"]),
        (f"At least {MIN_TRADES} completed trades", len(s.trades) >= MIN_TRADES),
    ]
    core = all(ok for _, ok in checks)
    if core and dsr_active >= CONFIDENCE:
        verdict = "EDGE"
    elif core:
        verdict = "RISK EDGE"
    else:
        verdict = "NO EDGE"
    wins = [t.ret for t in s.trades if t.ret > 0]
    return {
        "verdict": verdict,
        "checks": checks,
        "strategy": strat,
        "benchmark": bench,
        "deflated_sharpe": dsr,
        "deflated_sharpe_vs_benchmark": dsr_active,
        "win_rate": len(wins) / len(s.trades) if s.trades else 0.0,
        "chosen": Counter(f.chosen for f in wf.folds).most_common(),
        "period": (s.returns.index[0].date(), s.returns.index[-1].date()),
    }


def render(card: dict, title: str = "") -> str:
    s, b = card["strategy"], card["benchmark"]
    rows = [
        ("CAGR", "cagr", "{:+.2%}"), ("Volatility", "vol", "{:.2%}"),
        ("Sharpe", "sharpe", "{:.2f}"), ("Sortino", "sortino", "{:.2f}"),
        ("Max drawdown", "max_dd", "{:.1%}"), ("Calmar", "calmar", "{:.2f}"),
    ]
    out = [f"== {title or 'Scorecard'} ==",
           f"Out-of-sample: {card['period'][0]} -> {card['period'][1]} ({s['years']} yrs)",
           f"{'':16}{'strategy':>12}{'buy&hold':>12}"]
    out += [f"{name:16}{fmt.format(s[k]):>12}{fmt.format(b[k]):>12}" for name, k, fmt in rows]
    out += [f"{'Exposure':16}{s['exposure']:>12.0%}{b['exposure']:>12.0%}",
            f"{'Trades':16}{s['trades']:>12}   win rate {card['win_rate']:.0%}",
            f"P(Sharpe edge is real, not luck):        {card['deflated_sharpe']:.1%}",
            f"P(makes more money than buy&hold):       {card['deflated_sharpe_vs_benchmark']:.1%}",
            "Checks:"]
    out += [f"  [{'PASS' if ok else 'FAIL'}] {label}" for label, ok in card["checks"]]
    out.append("Strategies chosen (windows): " +
               ", ".join(f"{lbl} x{n}" for lbl, n in card["chosen"][:5]))
    out.append(f"VERDICT: {card['verdict']}")
    return "\n".join(out)


def portfolio_table(runs: dict, bpy: int, start: str | None = None, end: str | None = None,
                    hurdle: str = "60/40", n_trials: int = 1) -> str:
    """One row per portfolio: return, risk, consistency, and P(Sharpe beats the hurdle)."""
    sl = slice(start, end)
    base = runs[hurdle].result.returns.loc[sl]
    head = (f"{'portfolio':22}{'CAGR':>7}{'vol':>7}{'Sharpe':>7}{'maxDD':>7}{'+yrs':>6}"
            f"{'worst yr':>9}{'underwater':>11}{'P(beat ' + hurdle + ')':>16}")
    rows = [head]
    for name, run in runs.items():
        r = run.result.returns.loc[sl]
        s, c = summary(r, bpy), consistency(r, bpy)
        p = "" if name in (hurdle,) or name.endswith(" only") else \
            f"{deflated_sharpe(r, n_trials, sharpe(base, 1)):.0%}"
        rows.append(f"{name:22}{s['cagr']:>7.1%}{s['vol']:>7.1%}{s['sharpe']:>7.2f}{s['max_dd']:>7.0%}"
                    f"{c['pos_years']:>6.0%}{c['worst_year']:>9.0%}{c['underwater_yrs']:>9.1f}y{p:>16}")
    idx = base.index
    return f"-- {idx[0].date()} -> {idx[-1].date()} ({len(idx) / bpy:.0f} yrs) --\n" + "\n".join(rows)
