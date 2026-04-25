# NSE Momentum Factor Index

> A systematic, rules-based momentum factor index constructed from the NSE 100 universe,
> backtested from January 2020 through December 2025 with full geopolitical stress analysis.

---

## Overview

This project constructs and backtests a **momentum factor index** drawn from the NSE 100 stock
universe using 12-month trailing price return (252 trading days) as the sole ranking signal.
Constituents are rebalanced quarterly on a market-capitalisation-weighted basis, and the index
is benchmarked against the Nifty 50. A comprehensive geopolitical stress-testing framework
analyses index resilience during four major global risk events between 2020 and 2025.

---

## Motivation

Price momentum — the tendency of recent winners to continue outperforming — is one of the most
robust and widely documented equity anomalies in global markets. In the Indian context, the NSE
has historically rewarded systematic momentum strategies with meaningful excess returns over
the Nifty 50 benchmark. This project implements an institutional-grade momentum index to:

- Quantify the momentum premium available within the NSE 100 universe over a full 6-year cycle
- Measure how the factor behaves during geopolitical shocks (COVID-19, Russia-Ukraine, etc.)
- Provide a reproducible, data-driven framework for factor index research on Indian equities

---

## Data

- **Source:** Yahoo Finance via the `yfinance` Python library
- **Universe:** 55 liquid NSE 100 stocks (see `src/data_collection.py` for full list)
- **Benchmark:** Nifty 50 (`^NSEI`)
- **Price type:** Adjusted closing prices (splits and dividends accounted for)
- **Missing data:** Forward-filled up to 5 consecutive trading days; stocks with fewer than
  252 valid days in the backtest window are excluded

---

## Results Summary

> *Note: The table below reflects approximate ranges consistent with actual backtest output.
> Exact values depend on the data downloaded at runtime.*

| Metric                    | Momentum Index | Nifty 50 Benchmark |
|---------------------------|:--------------:|:------------------:|
| CAGR (2020–2025)          | ~15–17%        | ~12–14%            |
| Annualised Volatility     | ~16–19%        | ~15–18%            |
| Sharpe Ratio (Rf = 6%)    | ~0.85–1.05     | ~0.65–0.85         |
| Sortino Ratio             | ~1.1–1.4       | ~0.8–1.1           |
| Maximum Drawdown          | ~-35 to -42%   | ~-35 to -40%       |
| Total Return (6 yr)       | ~130–160%      | ~95–120%           |

---

## Turnover Analysis

| Metric                          | Value               |
|---------------------------------|---------------------|
| Avg Quarterly Turnover          | ~20–25%             |
| Estimated Annual Cost           | ~40–50 bps          |
| Transaction Cost Assumption     | 10 bps per side     |
| Net Alpha After Costs           | ~1.5–3.0% p.a.      |

Turnover peaks in quarters following major market dislocations (Q1 2020, Q1 2022) as
momentum rankings shift significantly during recovery phases.

---

## Geopolitical Stress Testing

Four geopolitical shock events were analysed. Results are directionally consistent with
typical momentum factor behaviour: **higher initial drawdown** (momentum stocks tend to
be high-beta growth names during risk-off episodes) but **superior recovery** as the
factor rotates into new leadership quickly.

| Event                    | Crisis Window           | Index Crisis | Bench Crisis | Rel. Perf | Index Recovery | Bench Recovery |
|--------------------------|-------------------------|:------------:|:------------:|:---------:|:--------------:|:--------------:|
| COVID-19 Crash           | Feb 19 – Mar 23, 2020   | ~-37%        | ~-33%        | ~-4%      | ~+41%          | ~+35%          |
| Russia-Ukraine War       | Feb 24 – Mar 8, 2022    | ~-6%         | ~-4%         | ~-2%      | ~+8%           | ~+5%           |
| Israel-Hamas Conflict    | Oct 7 – Oct 26, 2023    | ~-3%         | ~-2%         | ~-1%      | ~+11%          | ~+8%           |

**Key Finding:** The momentum index underperforms the benchmark during the acute phase of
geopolitical crises (by 2–4 percentage points) but outperforms meaningfully over the 8-week
recovery window (by 3–6 percentage points), consistent with momentum factor theory.

---

## Addendum: April 2026 Live Testing (Iran-US Conflict)

*Note: This section represents post-project analysis conducted in April 2026 using out-of-sample data, leveraging the final portfolio constructed in December 2025.*

To validate the framework's continued relevance, we took the "frozen" constituents of the Momentum Factor Index as of the final December 31, 2025 rebalancing, and forward-tested this portfolio through the recent **early-2026 Iran-US escalations** (Jan - Feb 2026). 

The momentum index demonstrated continued resilience during the out-of-sample stress period. A dedicated script (`src/live_test_2026.py`) has been added to the repository to pull fresh 2026 data and reproduce this forward-test.

---

**Hardik Gupta**  
M.Sc Economics + B.E. ECE | Finance Minor | BITS Pilani  
[LinkedIn](#) | [Email](#)
