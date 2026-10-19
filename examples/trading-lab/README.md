# Trading Lab

A research and paper-trading system built to answer one question honestly:
**does this strategy have a real edge, or did we just get lucky?**

It does not promise profits. It promises not to lie to you about them.

## Quick start

```bash
cd examples/trading-lab
pip install -r requirements.txt          # pandas, numpy, requests, pytest (+ yfinance optional)
python -m pytest -q                      # integrity tests

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

## Consistent returns: multi-asset portfolios (`portfolio.py`)

```bash
python -m trading_lab portfolio                          # stocks / 10y Treasuries / gold, 1872-2026
python -m trading_lab portfolio --universe yahoo:SPY,IEF,GLD --lag 1   # daily ETFs, your machine
```

No leverage: every rule keeps exposure at or below 100%, and the engine rejects anything else.
The trend and volatility windows (10 and 12 months) are textbook values fixed in advance, not tuned.

| Since 2008 (after the rules were published) | CAGR | Sharpe | Max DD | Worst year | P(beats 60/40) |
|---|---|---|---|---|---|
| Stocks only | 11.3% | 0.88 | -45% | -39% | |
| 60/40 | 8.1% | 1.05 | -26% | -19% | |
| **Equal weight + trend** | 6.4% | **1.27** | **-6%** | **-4%** | 46% |

Over 1972-2026: CAGR 8.1% vs 11.2% for stocks, max DD -11% vs -49%, positive in 87% of years.
The trade-off is plain: roughly a third less return for about a tenth of the crash risk. The higher
Sharpe is consistent across periods but not yet statistically proven against 60/40. Cash earns
0% in these tests, which understates the trend rules (they hold cash about a third of the time).

## Autopilot (hands-off paper trading)

Live paper accounts, run weekly by a scheduled Claude routine (nothing to do on your side):

| Journal | Account | Data | Safety stop |
|---|---|---|---|
| `paper/etf.jsonl` | $100k, "equal weight + trend" on SPY / IEF / GLD (opened 2026-10-16) | daily closes | -15% |
| `paper/msft.jsonl` | $10k, trend rule on Microsoft (opened 2026-10-16) | daily closes | -25% |
| `paper/history.jsonl` | retired 2026-10-19: the same rule on monthly-average index data | monthly | -15% |

```bash
python -m trading_lab autopilot --universe stock:SPY,IEF,GLD --journal paper/etf.jsonl
python -m trading_lab autopilot --universe stock:MSFT --journal paper/msft.jsonl --cash 10000 --max-dd 0.25
python -m trading_lab dashboard --journal paper/etf.jsonl --msft-journal paper/msft.jsonl
```

- Rebalances once per month; never borrows (cash stays >= 0 after costs). Cash earns 0% in the paper
  account; in a real account it would sit in T-bills.
- Circuit breaker: sells everything and HALTs at the stop until a deliberate `--resume "<reason>"`.
- On real daily ETF prices (2008-2026) the rule earned 5.4-5.7%/yr with a -10 to -12% worst drawdown
  and a -7 to -8% worst year: the most consistent option tested, but its Sharpe ratio does not beat
  60/40 or plain equal weight. The monthly-average index test (Sharpe 1.27) overstated it.
- Paper only. Connecting real money is the owner's decision, after paper results earn it.

## Single-stock sleeve (MSFT)

A separate $10k paper account runs the same trend rule on one stock: hold MSFT while it is
above its 10-month average, otherwise cash. Its safety stop is -25%, because one stock swings
about twice as much as the portfolio. MSFT was chosen for its 40-year history, liquidity and
diversified business. Its backtest is flattered by hindsight (we know it won), so it is judged
mainly on whether the rule cuts the crashes.

```bash
python -m trading_lab evaluate --source stock:MSFT
python -m trading_lab autopilot --universe stock:MSFT --journal paper/msft.jsonl --cash 10000 --max-dd 0.25
```

Needs network access to `stooq.com` or Yahoo (`query1/query2.finance.yahoo.com`, `fc.yahoo.com`).
On Microsoft since 2008 the trend rule cut the worst year from -44% to -25% and the worst drawdown
from -56% to -41%, but earned 11.3%/yr against 17.6% for simply holding (Sharpe 0.62 vs 0.71).

## Known limits

- Single asset, long/flat only. No shorting, leverage or portfolio sizing yet.
- Paper fills happen at the latest close; real fills will differ (see `review`).
- `yahoo:` needs `yfinance`, which this repo's cloud sandbox cannot reach.
