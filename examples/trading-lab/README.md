# Trading Lab

A research and paper-trading system built to answer one question honestly:
**does this strategy have a real edge, or did we just get lucky?**

It does not promise profits. It promises not to lie to you about them.

## Quick start

```bash
cd examples/trading-lab
pip install -r requirements.txt          # pandas, numpy, requests, pytest (+ yfinance optional)
python -m pytest -q                      # 24 integrity tests

# Is there an edge? (155 years of S&P 500, out-of-sample only)
python -m trading_lab evaluate --source shiller --lag 2

# Daily data on your own machine (Yahoo is reachable there)
python -m trading_lab evaluate --source yahoo:SPY

# Paper trade: run once per day after the close (cron), then audit
python -m trading_lab paper-step --source yahoo:SPY --journal journal/spy.jsonl
python -m trading_lab review     --source yahoo:SPY --journal journal/spy.jsonl
```

## How it works

| Stage | File | What it guards against |
|---|---|---|
| Data | `data.py` | Price-only data that ignores dividends; averaged prices |
| Strategies | `strategies.py` | Look-ahead: a signal at bar *t* only sees data up to *t* (tested) |
| Backtest | `backtest.py` | Same-bar execution, free trading: commission and slippage are charged on every change |
| Walk-forward | `walkforward.py` | Overfitting: pick the strategy on the past, grade it only on the unseen future |
| Scorecard | `scorecard.py` | Data mining: a deflated Sharpe ratio charges for every strategy tried |
| Paper trading | `paper.py` | Silent decay: an append-only journal and a kill switch (`HALT`) when live results fall significantly below the backtest |

**The learning loop.** Every year (configurable), the system re-ranks a small,
fixed menu of strategies (trend, momentum, mean-reversion, buy-and-hold) on the
last 10 years and trades the winner. Strategies that stop working lose their
rank. The menu is kept small on purpose: each extra strategy is another lottery
ticket, and the scorecard charges for it.

## Verdicts

- **EDGE**: beats buy-and-hold on Sharpe and drawdown, survives the luck
  haircut, and makes significantly more money. Candidate for real capital.
- **RISK EDGE**: better risk-adjusted and survives the haircut, but doesn't
  make provably more money.
- **NO EDGE**: keep researching. Do not trade it.

## First results (2026-10-01, Shiller monthly S&P 500, 1881-2026 out-of-sample)

| Timing | Sharpe (strat / B&H) | Max DD (strat / B&H) | CAGR (strat / B&H) | P(edge is real) | Verdict |
|---|---|---|---|---|---|
| lag 1 (optimistic) | 0.93 / 0.70 | -48% / -82% | 9.5% / 9.4% | 86% | NO EDGE |
| lag 2 (honest for averaged data) | 0.74 / 0.70 | -52% / -82% | 7.5% / 9.4% | 17% | NO EDGE |
| random walk (control) | — | — | — | <1% | NO EDGE |

Reading: trend-following roughly **halved the worst crash** but gave up return.
Most of the apparent Sharpe edge comes from Shiller's monthly *averaged* prices,
which manufacture momentum. Cash earns 0% here, which slightly understates
strategies that sit out. The next step is to re-run on daily total-return data.

## Known limits

- Single asset, long/flat only. No shorting, leverage or portfolio sizing yet.
- Paper fills happen at the latest close; real fills will differ (see `review`).
- `yahoo:` needs `yfinance`, which this repo's cloud sandbox cannot reach.
