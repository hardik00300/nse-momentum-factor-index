"""
NSE Momentum Factor Index — Master Run Script
Author: Hardik Gupta
Project Timeline: December 2025 - January 2026

Executes the complete pipeline end-to-end:
  1. Data collection      → data/
  2. Index construction   → results/index_performance.csv + rebalancing_calendar.csv
  3. Performance analysis → results/performance_metrics.csv + 2 charts
  4. Stress testing       → results/geopolitical_stress_tests.csv + 2 charts

Usage:
    cd nse-momentum-factor-index
    python run_all.py
"""

import os
import sys
import time
import logging

# Fix Windows console Unicode issues
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure src/ is on the path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def step(num: int, name: str) -> None:
    print(f"\n{'='*65}")
    print(f"  STEP {num}: {name}")
    print(f"{'='*65}")


def main() -> None:
    t0 = time.time()

    print("=" * 65)
    print("  NSE MOMENTUM FACTOR INDEX — COMPLETE PIPELINE")
    print("  Author: Hardik Gupta | December 2025 – January 2026")
    print("=" * 65)

    os.makedirs(DATA_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Data Collection ───────────────────────────────────────────────
    step(1, "Data Collection")
    from data_collection import (
        NSE_UNIVERSE, download_price_data, download_market_caps,
        download_benchmark, save_data, DATA_START_DATE, DATA_END_DATE,
    )
    prices      = download_price_data(NSE_UNIVERSE, DATA_START_DATE, DATA_END_DATE)
    valid_ticks = prices.columns.tolist()
    market_caps = download_market_caps(valid_ticks)
    benchmark   = download_benchmark()
    save_data(prices, market_caps, benchmark)
    print(f"  ✓ Stocks: {prices.shape[1]}, Days: {len(prices)}, Benchmark: {len(benchmark)}")

    # ── Step 2: Index Construction ────────────────────────────────────────────
    step(2, "Index Construction & Backtest")
    from index_construction import backtest_index, INDEX_BASE
    import pandas as pd

    index_df, rebal_df = backtest_index(prices, market_caps)
    index_df.drop(columns=["constituents"]).to_csv(
        os.path.join(RESULTS_DIR, "index_performance.csv")
    )
    rebal_df.to_csv(os.path.join(RESULTS_DIR, "rebalancing_calendar.csv"), index=False)

    final_val    = index_df["index_value"].iloc[-1]
    years        = (index_df.index[-1] - index_df.index[0]).days / 365.25
    cagr         = ((final_val / INDEX_BASE) ** (1 / years) - 1) * 100
    avg_turnover = rebal_df["turnover_rate"].mean() * 100
    print(f"  ✓ Final index: {final_val:.2f}, CAGR: {cagr:.2f}%, Avg turnover: {avg_turnover:.1f}%")

    # ── Step 3: Performance Analysis ──────────────────────────────────────────
    step(3, "Performance Analysis & Charts")
    from performance_analysis import (
        compute_all_metrics, plot_performance_comparison, plot_turnover_chart,
    )
    metrics_df, turnover_df, attr_df = compute_all_metrics(index_df, benchmark, rebal_df)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "performance_metrics.csv"), index=False)
    turnover_df.to_csv(os.path.join(RESULTS_DIR, "turnover_analysis.csv"), index=False)
    attr_df.to_csv(os.path.join(RESULTS_DIR, "attribution_results.csv"), index=False)
    plot_performance_comparison(
        index_df, benchmark,
        os.path.join(RESULTS_DIR, "performance_comparison.png"),
    )
    plot_turnover_chart(rebal_df, os.path.join(RESULTS_DIR, "turnover_chart.png"))
    print(f"  ✓ Metrics saved, 2 charts generated")

    # ── Step 4: Stress Testing ────────────────────────────────────────────────
    step(4, "Geopolitical Stress Testing & Charts")
    from stress_testing import (
        run_all_stress_tests, plot_drawdown_analysis, plot_stress_test_comparison,
    )
    stress_df = run_all_stress_tests(index_df, benchmark)
    stress_df.to_csv(os.path.join(RESULTS_DIR, "geopolitical_stress_tests.csv"), index=False)
    plot_drawdown_analysis(
        index_df, benchmark,
        os.path.join(RESULTS_DIR, "drawdown_analysis.png"),
    )
    plot_stress_test_comparison(
        stress_df,
        os.path.join(RESULTS_DIR, "stress_test_comparison.png"),
    )
    print(f"  ✓ Stress tests complete, 2 charts generated")

    # ── Final Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{'='*65}")
    print(f"  Data files    : {DATA_DIR}")
    print(f"  Result files  : {RESULTS_DIR}")
    print()

    # List all generated files
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        size  = os.path.getsize(fpath)
        print(f"    {'📊' if fname.endswith('.png') else '📄'} {fname}  ({size:,} bytes)")

    print(f"\n  Momentum Index CAGR : {cagr:.2f}%")
    print(f"  Rebalances          : {len(rebal_df)}")
    print(f"  Avg Quarterly TO    : {avg_turnover:.1f}%")
    print(f"  Beta                : {attr_df['beta'].iloc[0]:.4f}")
    print(f"  Alpha (annual)      : {attr_df['alpha_annual'].iloc[0]:.4f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
