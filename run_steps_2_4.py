"""
NSE Momentum Factor Index — Steps 2-4 Runner (skip re-download)
Runs index construction, performance analysis, and stress testing
using already-downloaded data in data/.
"""
import os, sys, time, logging, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SRC_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, SRC_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import pandas as pd

os.makedirs(RESULTS_DIR, exist_ok=True)

t0 = time.time()
print("=" * 65)
print("NSE MOMENTUM FACTOR INDEX - Steps 2-4")
print("=" * 65)

# ── Load saved data ───────────────────────────────────────────────────────────
print("\nLoading saved data from data/...")
from data_collection import load_data
prices, market_caps, benchmark = load_data()
print(f"  Prices: {prices.shape}, Market caps: {len(market_caps)}, Benchmark: {len(benchmark)}")

# ── Step 2: Index Construction & Backtest ────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: Index Construction & Backtest")
print("=" * 65)
from index_construction import backtest_index, INDEX_BASE

index_df, rebal_df = backtest_index(prices, market_caps)
index_df.drop(columns=["constituents"]).to_csv(os.path.join(RESULTS_DIR, "index_performance.csv"))
rebal_df.to_csv(os.path.join(RESULTS_DIR, "rebalancing_calendar.csv"), index=False)

final_val    = index_df["index_value"].iloc[-1]
years        = (index_df.index[-1] - index_df.index[0]).days / 365.25
cagr         = ((final_val / INDEX_BASE) ** (1 / years) - 1) * 100
avg_turnover = rebal_df["turnover_rate"].mean() * 100
print(f"  Final index  : {final_val:.2f}")
print(f"  CAGR         : {cagr:.2f}%")
print(f"  Avg turnover : {avg_turnover:.1f}%")
print(f"  Rebalances   : {len(rebal_df)}")

# ── Step 3: Performance Analysis & Charts ────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: Performance Analysis & Charts")
print("=" * 65)
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
print(f"  Beta     : {attr_df['beta'].iloc[0]:.4f}")
print(f"  Alpha/yr : {attr_df['alpha_annual'].iloc[0]:.4f}")
print(f"  R-sq     : {attr_df['r_squared'].iloc[0]:.4f}")
print("  Charts 1 & 2 saved.")

# ── Step 4: Stress Testing & Charts ──────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4: Geopolitical Stress Testing & Charts")
print("=" * 65)
from stress_testing import (
    run_all_stress_tests, plot_drawdown_analysis, plot_stress_test_comparison,
)
stress_df = run_all_stress_tests(index_df, benchmark)
stress_df.to_csv(os.path.join(RESULTS_DIR, "geopolitical_stress_tests.csv"), index=False)
plot_drawdown_analysis(index_df, benchmark, os.path.join(RESULTS_DIR, "drawdown_analysis.png"))
plot_stress_test_comparison(stress_df, os.path.join(RESULTS_DIR, "stress_test_comparison.png"))
print("  Charts 3 & 4 saved.")

# ── Final Summary ──────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print("\n" + "=" * 65)
print(f"PIPELINE COMPLETE ({elapsed:.1f}s)")
print("=" * 65)
print("\nGenerated files:")
for fname in sorted(os.listdir(RESULTS_DIR)):
    fpath = os.path.join(RESULTS_DIR, fname)
    size  = os.path.getsize(fpath)
    tag   = "[PNG]" if fname.endswith(".png") else "[CSV]"
    print(f"  {tag} {fname}  ({size:,} bytes)")

print(f"\nMomentum Index CAGR : {cagr:.2f}%")
print(f"Avg Quarterly TO    : {avg_turnover:.1f}%")
print(f"Beta                : {attr_df['beta'].iloc[0]:.4f}")
print(f"Alpha (annual)      : {attr_df['alpha_annual'].iloc[0]:.4f}")

# Print metrics table
print("\n--- Performance Metrics ---")
print(metrics_df.to_string(index=False))

# Print stress test results
print("\n--- Stress Test Results ---")
disp_cols = ["event", "index_crisis_return", "benchmark_crisis_return",
             "crisis_relative_perf", "index_recovery_return", "benchmark_recovery_return"]
disp = stress_df[disp_cols].copy()
for col in disp_cols[1:]:
    disp[col] = disp[col].apply(
        lambda v: f"{v:.2%}" if v is not None and str(v) != "None" else "N/A"
    )
print(disp.to_string(index=False))
print("=" * 65)
print("Done. All results in results/")
