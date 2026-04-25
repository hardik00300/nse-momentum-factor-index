"""
NSE Momentum Factor Index — Performance Analysis Module
Author: Hardik Gupta
Project Timeline: December 2025 - January 2026

Computes all performance metrics (CAGR, Sharpe, Sortino, max drawdown),
turnover analysis, factor attribution (beta/alpha/R²), and generates
two publication-quality charts:
  - performance_comparison.png  (cumulative returns)
  - turnover_chart.png          (quarterly turnover bar chart)

Run after index_construction.py.
"""

import os
import sys
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

RISK_FREE_RATE: float = 0.06          # 6% annual, proxy for Indian T-bill
TRADING_DAYS:   int   = 252
TRANSACTION_BPS: float = 10.0         # 10 bps per unit turnover

# ── Chart Style ───────────────────────────────────────────────────────────────
STYLE = {
    "bg":       "#0d1117",
    "surface":  "#161b22",
    "border":   "#30363d",
    "accent1":  "#58a6ff",   # Blue — momentum index
    "accent2":  "#f0883e",   # Orange — Nifty 50
    "accent3":  "#3fb950",   # Green
    "accent4":  "#ff7b72",   # Red
    "text":     "#e6edf3",
    "subtext":  "#8b949e",
    "grid":     "#21262d",
}


def _apply_dark_style(fig: plt.Figure, axes) -> None:
    """Apply consistent dark theme to figure and axes."""
    fig.patch.set_facecolor(STYLE["bg"])
    ax_list = axes if hasattr(axes, "__iter__") else [axes]
    for ax in ax_list:
        ax.set_facecolor(STYLE["surface"])
        ax.tick_params(colors=STYLE["subtext"], labelsize=9)
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["border"])
        ax.grid(color=STYLE["grid"], linewidth=0.5, alpha=0.8)


# ══════════════════════════════════════════════════════════════════════════════
# Performance Metric Functions
# ══════════════════════════════════════════════════════════════════════════════

def calculate_cagr(index_series: pd.Series) -> float:
    """Calculate Compound Annual Growth Rate from an index level series.

    CAGR = (Final / Initial)^(1/Years) - 1

    Args:
        index_series: Series of index levels with DatetimeIndex.

    Returns:
        CAGR as a decimal (e.g. 0.15 for 15%).
    """
    years = (index_series.index[-1] - index_series.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (index_series.iloc[-1] / index_series.iloc[0]) ** (1 / years) - 1


def calculate_annualized_vol(daily_returns: pd.Series) -> float:
    """Calculate annualised volatility from daily returns.

    Vol = daily_std × sqrt(252)

    Args:
        daily_returns: Series of daily return values.

    Returns:
        Annualised volatility as a decimal.
    """
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def calculate_sharpe(daily_returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Calculate annualised Sharpe ratio.

    Sharpe = (Ann_Return - Rf) / Ann_Vol

    Args:
        daily_returns: Series of daily returns.
        rf: Annual risk-free rate (default 6%).

    Returns:
        Sharpe ratio (annualised).
    """
    ann_ret = daily_returns.mean() * TRADING_DAYS
    ann_vol = calculate_annualized_vol(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - rf) / ann_vol


def calculate_sortino(daily_returns: pd.Series, rf: float = RISK_FREE_RATE) -> float:
    """Calculate Sortino ratio using downside deviation.

    Sortino = (Ann_Return - Rf) / Downside_Dev

    Args:
        daily_returns: Series of daily returns.
        rf: Annual risk-free rate.

    Returns:
        Sortino ratio (annualised).
    """
    ann_ret  = daily_returns.mean() * TRADING_DAYS
    daily_rf = rf / TRADING_DAYS
    downside = daily_returns[daily_returns < daily_rf] - daily_rf
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)
    if downside_dev == 0:
        return 0.0
    return (ann_ret - rf) / downside_dev


def calculate_max_drawdown(index_series: pd.Series) -> float:
    """Calculate maximum peak-to-trough drawdown.

    MDD = max((cumulative_value / running_max) - 1)

    Args:
        index_series: Series of cumulative index levels.

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.35 for -35%).
    """
    running_max = index_series.cummax()
    drawdown    = (index_series / running_max) - 1.0
    return drawdown.min()


def calculate_drawdown_series(index_series: pd.Series) -> pd.Series:
    """Return the rolling drawdown time-series.

    Args:
        index_series: Series of cumulative index levels.

    Returns:
        Series of drawdown values (negative), same index as input.
    """
    running_max = index_series.cummax()
    return (index_series / running_max) - 1.0


def calculate_beta_alpha(
    index_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict[str, float]:
    """OLS regression of index returns on benchmark returns.

    Returns market beta, annualised factor alpha, and R-squared.

    Args:
        index_returns: Daily index return Series.
        benchmark_returns: Daily benchmark return Series.

    Returns:
        Dict with keys: beta, alpha_annual, r_squared, p_value.
    """
    aligned = pd.concat([index_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["idx", "bench"]

    slope, intercept, r_value, p_value, _ = stats.linregress(
        aligned["bench"], aligned["idx"]
    )
    return {
        "beta":         round(slope, 4),
        "alpha_daily":  round(intercept, 6),
        "alpha_annual": round(intercept * TRADING_DAYS, 4),
        "r_squared":    round(r_value ** 2, 4),
        "p_value":      round(p_value, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full Metrics Report
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    index_df: pd.DataFrame,
    benchmark: pd.Series,
    rebal_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute all performance metrics, turnover stats, and attribution.

    Args:
        index_df: DataFrame from backtest with 'index_value' and 'daily_return'.
        benchmark: Nifty 50 level Series.
        rebal_df: Rebalancing calendar DataFrame.

    Returns:
        Tuple of (metrics_df, turnover_df, attribution_df).
    """
    idx_series  = index_df["index_value"]
    idx_returns = index_df["daily_return"]

    # Align benchmark to backtest period
    bench_bt = benchmark.reindex(idx_series.index).ffill()
    bench_returns = bench_bt.pct_change().fillna(0.0)

    # ── Momentum Index Metrics ────────────────────────────────────────────────
    idx_cagr    = calculate_cagr(idx_series)
    idx_vol     = calculate_annualized_vol(idx_returns)
    idx_sharpe  = calculate_sharpe(idx_returns)
    idx_sortino = calculate_sortino(idx_returns)
    idx_mdd     = calculate_max_drawdown(idx_series)

    # ── Benchmark Metrics ─────────────────────────────────────────────────────
    bench_base   = bench_bt.iloc[0]
    bench_series = (bench_bt / bench_base) * 100
    ben_cagr    = calculate_cagr(bench_series)
    ben_vol     = calculate_annualized_vol(bench_returns)
    ben_sharpe  = calculate_sharpe(bench_returns)
    ben_sortino = calculate_sortino(bench_returns)
    ben_mdd     = calculate_max_drawdown(bench_series)

    # ── Turnover & Cost ───────────────────────────────────────────────────────
    avg_turnover   = rebal_df["turnover_rate"].mean() if not rebal_df.empty else 0
    annual_cost    = avg_turnover * 4 * (TRANSACTION_BPS / 10000)  # 4 quarters
    net_alpha      = idx_cagr - ben_cagr - annual_cost

    # ── Attribution ───────────────────────────────────────────────────────────
    attribution = calculate_beta_alpha(idx_returns, bench_returns)

    # ── Build metrics DataFrame ───────────────────────────────────────────────
    metrics = pd.DataFrame({
        "Metric": [
            "CAGR", "Annualised Volatility", "Sharpe Ratio", "Sortino Ratio",
            "Max Drawdown", "Total Return",
            "Avg Quarterly Turnover", "Est. Annual Transaction Cost",
            "Net Alpha vs Benchmark",
        ],
        "Momentum Index": [
            f"{idx_cagr:.2%}", f"{idx_vol:.2%}", f"{idx_sharpe:.3f}", f"{idx_sortino:.3f}",
            f"{idx_mdd:.2%}", f"{(idx_series.iloc[-1]/idx_series.iloc[0]-1):.2%}",
            f"{avg_turnover:.2%}", f"{annual_cost:.4%}", f"{net_alpha:.2%}",
        ],
        "Nifty 50 Benchmark": [
            f"{ben_cagr:.2%}", f"{ben_vol:.2%}", f"{ben_sharpe:.3f}", f"{ben_sortino:.3f}",
            f"{ben_mdd:.2%}", f"{(bench_series.iloc[-1]/bench_series.iloc[0]-1):.2%}",
            "—", "—", "—",
        ],
    })

    # ── Turnover DataFrame ────────────────────────────────────────────────────
    turnover_df = rebal_df[["rebal_date", "turnover_rate", "stocks_added", "stocks_dropped"]].copy()
    turnover_df["cost_bps"] = (turnover_df["turnover_rate"] * TRANSACTION_BPS).round(2)

    # ── Attribution DataFrame ─────────────────────────────────────────────────
    attr_df = pd.DataFrame([{
        "beta":         attribution["beta"],
        "alpha_annual": attribution["alpha_annual"],
        "r_squared":    attribution["r_squared"],
        "p_value":      attribution["p_value"],
        "risk_free_rate": RISK_FREE_RATE,
        "avg_quarterly_turnover": avg_turnover,
        "est_annual_cost": annual_cost,
    }])

    logger.info(f"Metrics computed — Index CAGR: {idx_cagr:.2%}, Nifty CAGR: {ben_cagr:.2%}")
    logger.info(f"Sharpe: {idx_sharpe:.3f}, Max DD: {idx_mdd:.2%}, Avg Turnover: {avg_turnover:.2%}")

    return metrics, turnover_df, attr_df


# ══════════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_performance_comparison(
    index_df: pd.DataFrame,
    benchmark: pd.Series,
    out_path: str,
) -> None:
    """Chart 1 — Cumulative performance: Momentum Index vs Nifty 50 (base 100).

    Args:
        index_df: Backtest output with 'index_value'.
        benchmark: Nifty 50 level Series.
        out_path: File path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    _apply_dark_style(fig, ax)

    idx_series  = index_df["index_value"]
    bench_bt    = benchmark.reindex(idx_series.index).ffill()
    bench_norm  = (bench_bt / bench_bt.iloc[0]) * 100

    ax.plot(idx_series.index, idx_series,   color=STYLE["accent1"], lw=2.2,
            label="NSE Momentum Factor Index", zorder=3)
    ax.plot(bench_norm.index, bench_norm,    color=STYLE["accent2"], lw=1.8,
            linestyle="--", label="Nifty 50 Benchmark", zorder=2)
    ax.fill_between(idx_series.index, idx_series, bench_norm,
                    where=(idx_series >= bench_norm),
                    alpha=0.08, color=STYLE["accent3"], zorder=1)
    ax.fill_between(idx_series.index, idx_series, bench_norm,
                    where=(idx_series < bench_norm),
                    alpha=0.08, color=STYLE["accent4"], zorder=1)

    # Annotations
    final_idx  = idx_series.iloc[-1]
    final_ben  = bench_norm.iloc[-1]
    yrs = (idx_series.index[-1] - idx_series.index[0]).days / 365.25
    cagr_idx = ((final_idx / 100) ** (1 / yrs) - 1) * 100
    cagr_ben = ((final_ben / 100) ** (1 / yrs) - 1) * 100
    ax.annotate(f"Index: {final_idx:.1f}\n(CAGR {cagr_idx:.1f}%)",
                xy=(idx_series.index[-1], final_idx),
                xytext=(-100, 10), textcoords="offset points",
                color=STYLE["accent1"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=STYLE["accent1"], lw=1))
    ax.annotate(f"Nifty 50: {final_ben:.1f}\n(CAGR {cagr_ben:.1f}%)",
                xy=(bench_norm.index[-1], final_ben),
                xytext=(-120, -30), textcoords="offset points",
                color=STYLE["accent2"], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=STYLE["accent2"], lw=1))

    ax.set_title("NSE Momentum Factor Index vs Nifty 50\nCumulative Performance (Base 100 = Jan 1, 2020)",
                 fontsize=14, fontweight="bold", pad=18, color=STYLE["text"])
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Index Level (Base 100)", fontsize=11)
    ax.legend(loc="upper left", facecolor=STYLE["surface"],
              edgecolor=STYLE["border"], labelcolor=STYLE["text"], fontsize=10)
    ax.set_xlim(idx_series.index[0], idx_series.index[-1])

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_turnover_chart(rebal_df: pd.DataFrame, out_path: str) -> None:
    """Chart 2 — Quarterly turnover bar chart.

    Args:
        rebal_df: Rebalancing calendar DataFrame.
        out_path: File path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    _apply_dark_style(fig, ax)

    dates    = [str(d) for d in rebal_df["rebal_date"]]
    turnover = rebal_df["turnover_rate"].values * 100
    avg_t    = turnover.mean()

    colors = [STYLE["accent1"] if t <= avg_t else STYLE["accent4"] for t in turnover]
    bars = ax.bar(dates, turnover, color=colors, width=0.6, edgecolor=STYLE["border"],
                  linewidth=0.5, zorder=3)

    ax.axhline(avg_t, color=STYLE["accent3"], lw=2, linestyle="--",
               label=f"Average: {avg_t:.1f}%", zorder=4)

    # Value labels on bars
    for bar, val in zip(bars, turnover):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom",
                color=STYLE["text"], fontsize=7, fontweight="bold")

    ax.set_title("Quarterly Portfolio Turnover Rate\nNSE Momentum Factor Index (2020–2025)",
                 fontsize=13, fontweight="bold", pad=15, color=STYLE["text"])
    ax.set_xlabel("Rebalancing Date", fontsize=11)
    ax.set_ylabel("Turnover Rate (%)", fontsize=11)
    ax.legend(facecolor=STYLE["surface"], edgecolor=STYLE["border"],
              labelcolor=STYLE["text"], fontsize=10)
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(0, max(turnover) * 1.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    plt.close()
    logger.info(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from data_collection import load_data

    print("=" * 65)
    print("NSE Momentum Factor Index — Performance Analysis")
    print("Author: Hardik Gupta | December 2025 – January 2026")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load raw data
    _, market_caps, benchmark = load_data()

    # Load backtest results
    index_df = pd.read_csv(
        os.path.join(RESULTS_DIR, "index_performance.csv"),
        index_col=0, parse_dates=True,
    )
    rebal_df = pd.read_csv(os.path.join(RESULTS_DIR, "rebalancing_calendar.csv"))

    # Compute metrics
    metrics_df, turnover_df, attr_df = compute_all_metrics(index_df, benchmark, rebal_df)

    # Save CSVs
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "performance_metrics.csv"), index=False)
    turnover_df.to_csv(os.path.join(RESULTS_DIR, "turnover_analysis.csv"), index=False)
    attr_df.to_csv(os.path.join(RESULTS_DIR, "attribution_results.csv"), index=False)

    # Generate charts
    plot_performance_comparison(
        index_df, benchmark,
        os.path.join(RESULTS_DIR, "performance_comparison.png"),
    )
    plot_turnover_chart(
        rebal_df,
        os.path.join(RESULTS_DIR, "turnover_chart.png"),
    )

    # Print summary table
    print("\n── Performance Metrics ───────────────────────────────────────")
    print(metrics_df.to_string(index=False))
    print(f"\n── Attribution ──────────────────────────────────────────────")
    print(f"  Beta      : {attr_df['beta'].iloc[0]:.4f}")
    print(f"  Alpha/yr  : {attr_df['alpha_annual'].iloc[0]:.4f}")
    print(f"  R-squared : {attr_df['r_squared'].iloc[0]:.4f}")
    print("─" * 65)
    print("✓ Done. Run stress_testing.py next.")
