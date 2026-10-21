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
| `paper/spy.jsonl` | **$100k, SPY volatility control (opened 2026-10-20) — the focus** | daily closes | -25% |
| `paper/msft.jsonl` | retired 2026-10-21 (owner chose to focus on SPY): trend rule on Microsoft | daily closes | -25% |
| `paper/history.jsonl` | retired 2026-10-19: the same rule on monthly-average index data | monthly | -15% |

```bash
python -m trading_lab autopilot --universe stock:SPY,IEF,GLD --journal paper/etf.jsonl
python -m trading_lab dashboard            # SPY account first; side accounts in dashboard.EXTRA_ACCOUNTS
```

- Rebalances once per month; never borrows (cash stays >= 0 after costs). Cash earns 0% in the paper
  account; in a real account it would sit in T-bills.
- Circuit breaker: sells everything and HALTs at the stop until a deliberate `--resume "<reason>"`.
- On real daily ETF prices (2008-2026) the rule earned 5.4-5.7%/yr with a -10 to -12% worst drawdown
  and a -7 to -8% worst year: the most consistent option tested, but its Sharpe ratio does not beat
  60/40 or plain equal weight. The monthly-average index test (Sharpe 1.27) overstated it.
- Paper only. Connecting real money is the owner's decision, after paper results earn it.

## SPY research and SPY account

`spy_research.py` tests five hypotheses about SPY, fixed before looking at results and all
reported (1994-2012 discovery, 2013-today out-of-sample, 2 bp per trade side, deflated for 5 trials):

| 2013-today | CAGR | Sharpe | Worst drop | P(beats holding SPY) |
|---|---|---|---|---|
| Buy & hold SPY | 15.0% | 0.92 | -34% | |
| Trend (above 200-day average) | 11.7% | 1.01 | -20% | 20% |
| Buy the dip (RSI(2) < 10 in uptrend) | 4.4% | 0.75 | -11% | 4% |
| Turn of the month | 2.2% | 0.34 | -10% | 0% |
| Volatility control (15%) | 13.0% | 1.03 | -19% | 22% |
| Overnight only | -1.0% | -0.04 | -38% | 0% |

Facts: nearly all of SPY's growth since 1993 came overnight (+2,433%) rather than during the day
(+26%), but trading it costs more than it earns. The dip rule was strong before 2013 (Sharpe 0.85,
published 2008) and faded after. Nothing beats holding SPY with confidence; trend and volatility
control keep most of the return with roughly half the worst drop.

`paper/spy.jsonl` runs volatility control live ($100k, rebalance when the weight drifts 10 points,
-25% safety stop). `paper/spy_observations.csv` logs SPY's state every trading day (price, overnight
gap, VIX and VIX3M, trend, 2-day RSI, every rule's signal) and `paper/spy_intraday/` keeps every
5-minute bar (Yahoo only serves the last 60 days, so this history exists nowhere else).

### Real-money gates (`python -m trading_lab readiness`)

Measured from the journal, so they cannot be argued into a pass: at least 126 trading days of paper
results; the bot ran on at least 95% of trading days; live returns not significantly below the tested
rule (z above -2); no unexplained safety stop. Plus five personal gates only the owner can tick: a
written plan for a 20% drop, money not needed for 5 years, starting with at most a quarter of the
intended amount, a low-cost broker, and understanding how short-term gains are taxed.

```bash
python -m trading_lab spy-research --refresh
python -m trading_lab autopilot --universe stock:SPY --journal paper/spy.jsonl --rule "spy volatility control" \
    --band 0.10 --cash 100000 --max-dd 0.25 --cost-bps 1 --slippage-bps 1
python -m trading_lab spy-log
python -m trading_lab readiness
```

## Day trading SPY (`intraday.py`, `minute_data.py`)

Five published day-trading rules, frozen on 2026-10-21 before testing, run on 1.06 million SPY
one-minute bars from Alpaca (2016-today). All are flat at the close, at most 1x long or short, filled
at the next bar's open. Discovery 2016-2022; judged on 2023-today; charged for 10 trials in total.

| 2023-today | CAGR at 2 bp/side | CAGR at 0.25 bp/side | P(real), 0.25 bp |
|---|---|---|---|
| Hold SPY during the day only | -3.2% | 5.7% | |
| Intraday momentum (Gao et al. 2018) | -8.4% | -0.1% | 6% |
| 5-minute opening-range breakout (Crabel; Zarattini & Aziz 2023) | -10.5% | -2.6% | 1% |
| Gap fade, 0.25-1% gaps (Raschke & Connors 1995) | -5.6% | -1.6% | 2% |
| VWAP trend (Zarattini & Aziz 2023) | -82.1% | -17.2% | 0% |
| Noise-area breakout (Zarattini, Aziz & Barbon 2024) | -5.4% | 2.5% | 26% |

Costs decide everything: at 1 bp per side every rule loses. Only the noise-area breakout makes money at
near-zero costs, and even then less than simply holding SPY during the day, with no statistical proof.
`python -m trading_lab intraday-log` adds each new session's result for every rule to
`paper/intraday_forward.csv`, a forward test no backtest can fake. No new day-trading rules are added
because of these results.

```bash
python -m trading_lab intraday        # full report (downloads minute bars to data/minute/ on first run)
python -m trading_lab intraday-log    # forward test, once per evening
```

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
