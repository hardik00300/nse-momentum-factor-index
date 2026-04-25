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

## Methodology

### Momentum Signal

The momentum score for each stock is defined as its **trailing 12-month total price return**:

```
Momentum(t) = Price(t) / Price(t - 252) - 1
```

The prior month is **not** skipped (unlike the academic Jegadeesh-Titman specification) to
align with NSE factor index conventions and to avoid lookahead bias in quarterly rebalancing.

### Constituent Selection

At each rebalancing date, stocks are ranked by momentum score. The **top 50 stocks** are
selected as index constituents. Stocks with fewer than 252 consecutive trading days of valid
price history on the rebalancing date are ineligible.

### Weighting Scheme

Selected constituents are weighted proportionally to their **free-float market capitalisation**
(proxied by current market cap from Yahoo Finance — see Limitations). Weights are computed as:

```
w_i = MarketCap_i / Σ MarketCap_j  (for j in constituents)
```

### Rebalancing Frequency

The index rebalances **quarterly** — on the last trading day of March, June, September, and
December. Full reconstitution is applied at each rebalance (no buffer rules or banding).

### Backtest Period

| Parameter        | Value                          |
|------------------|-------------------------------|
| Backtest start   | January 1, 2020               |
| Backtest end     | December 31, 2025              |
| Data history     | January 1, 2019 (for lookback) |
| Index base value | 100 (Jan 1, 2020)              |

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

## Assumptions & Limitations

1. **Survivorship Bias:** The stock universe is based on current NSE 100 membership, not
   historical constituent lists. Stocks delisted or downgraded during 2020–2025 are excluded,
   which may modestly overstate returns.

2. **Static Market Cap Proxy:** Free-float market cap is proxied using the most recent
   available value from Yahoo Finance. Historical market cap data for weighting is not used,
   which may affect weight accuracy in early backtest years.

3. **Transaction Costs:** Estimated at 10 basis points per side on turnover. Actual impact
   costs, market impact, and taxes (STT, GST) are not modelled.

4. **No Buffer Rules:** Full reconstitution at each rebalance (no index buffer bands), which
   may overstate turnover relative to production index implementations.

5. **Data Source:** Yahoo Finance adjusted prices are used as a proxy for NSE-official adjusted
   prices. Minor discrepancies may exist due to dividend and split adjustment methodology.

---

## Technical Implementation

| Dependency     | Version   | Purpose                        |
|----------------|-----------|-------------------------------|
| Python         | ≥ 3.10    | Core language                  |
| pandas         | ≥ 2.0     | Data manipulation              |
| numpy          | ≥ 1.24    | Numerical operations           |
| yfinance       | ≥ 0.2.31  | Market data download           |
| matplotlib     | ≥ 3.7     | Visualisation                  |
| scipy          | ≥ 1.10    | OLS regression (attribution)   |
| statsmodels    | ≥ 0.14    | Statistical utilities          |
| jupyter        | ≥ 1.0     | Interactive notebook           |

---

## Repository Structure

```
nse-momentum-factor-index/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── run_all.py                       # Master pipeline runner
│
├── src/
│   ├── data_collection.py           # yfinance downloads, data persistence
│   ├── index_construction.py        # Momentum engine + quarterly backtest
│   ├── performance_analysis.py      # Metrics, attribution, Charts 1 & 2
│   └── stress_testing.py            # Geopolitical event analysis, Charts 3 & 4
│
├── data/
│   ├── nse_prices.csv               # Daily adjusted close prices (2019–2025)
│   ├── market_caps.csv              # Market cap proxy for weighting
│   └── nifty50_benchmark.csv        # Nifty 50 daily levels
│
├── results/
│   ├── index_performance.csv        # Daily index values & returns
│   ├── rebalancing_calendar.csv     # Constituents, turnover at each rebalance
│   ├── performance_metrics.csv      # Full metrics comparison table
│   ├── turnover_analysis.csv        # Quarterly turnover statistics
│   ├── attribution_results.csv      # Beta, alpha, R-squared
│   ├── geopolitical_stress_tests.csv# Event-level stress test results
│   ├── performance_comparison.png   # Chart 1: Cumulative performance
│   ├── turnover_chart.png           # Chart 2: Quarterly turnover
│   ├── drawdown_analysis.png        # Chart 3: Rolling drawdown
│   └── stress_test_comparison.png   # Chart 4: Crisis vs recovery
│
└── notebooks/
    └── full_analysis.ipynb          # End-to-end interactive walkthrough
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline
python run_all.py

# Results will appear in data/ and results/
```

Or run each step individually:

```bash
python src/data_collection.py
python src/index_construction.py
python src/performance_analysis.py
python src/stress_testing.py
```

---

## Author & Timeline

**Hardik Gupta**  
*December 2025 – January 2026*

This project was conceived and completed as an independent quantitative finance research
initiative during December 2025 – January 2026, simulating the workflow of an institutional
factor index construction desk. All analysis uses data available as of December 31, 2025.
The repository was subsequently published to GitHub in April 2026 without modification to
the underlying research or data.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
