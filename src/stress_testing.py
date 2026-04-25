"""
NSE Momentum Factor Index — Geopolitical Stress Testing Module
Author: Hardik Gupta
Project Timeline: December 2025 - January 2026

Analyses index behaviour during four geopolitical shock events:
  1. COVID-19 Crash         (Feb–Mar 2020)
  2. Russia-Ukraine War     (Feb–Mar 2022)
  3. Israel-Hamas Conflict  (Oct 2023)
  4. Iran-US Escalation     (Jun 2025) — post-project live test

Generates two publication-quality charts:
  - drawdown_analysis.png        (rolling drawdown with shaded event windows)
  - stress_test_comparison.png   (grouped bar chart: crisis vs recovery)

Run after performance_analysis.py.
"""

import os
import sys
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# ── Chart Style (consistent with performance_analysis.py) ─────────────────────
STYLE = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "border":  "#30363d",
    "accent1": "#58a6ff",
    "accent2": "#f0883e",
    "accent3": "#3fb950",
    "accent4": "#ff7b72",
    "accent5": "#d2a8ff",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
    "grid":    "#21262d",
}

# ── Geopolitical Event Definitions ────────────────────────────────────────────
EVENTS: Dict[str, dict] = {
    "COVID-19 Crash": {
        "crisis_start":   "2020-02-19",
        "crisis_end":     "2020-03-23",
        "recovery_start": "2020-03-24",
        "recovery_end":   "2020-05-18",
        "color":          "#ff7b72",
        "description":    "Global pandemic; NSE circuit breakers triggered.",
    },
    "Russia-Ukraine War": {
        "crisis_start":   "2022-02-24",
        "crisis_end":     "2022-03-08",
        "recovery_start": "2022-03-09",
        "recovery_end":   "2022-05-03",
        "color":          "#f0883e",
        "description":    "Russian invasion of Ukraine; commodity shock.",
    },
    "Israel-Hamas Conflict": {
        "crisis_start":   "2023-10-07",
        "crisis_end":     "2023-10-26",
        "recovery_start": "2023-10-27",
        "recovery_end":   "2023-12-21",
        "color":          "#d2a8ff",
        "description":    "Middle-East conflict; oil price volatility.",
    },
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


def _window_return(series: pd.Series, start: str, end: str) -> float:
    """Compute total return of a level/index series over a date window.

    Args:
        series: Cumulative index level Series with DatetimeIndex.
        start: Window start date string 'YYYY-MM-DD'.
        end: Window end date string 'YYYY-MM-DD'.

    Returns:
        Total return as a decimal. Returns NaN if window has no data.
    """
    mask = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
    window = series[mask]
    if len(window) < 2:
        return float("nan")
    return (window.iloc[-1] / window.iloc[0]) - 1.0


def _window_max_drawdown(series: pd.Series, start: str, end: str) -> float:
    """Compute maximum drawdown within a specific date window.

    Args:
        series: Cumulative index level Series with DatetimeIndex.
        start: Window start date string.
        end: Window end date string.

    Returns:
        Maximum drawdown as a negative decimal.
    """
    mask   = (series.index >= pd.Timestamp(start)) & (series.index <= pd.Timestamp(end))
    window = series[mask]
    if window.empty:
        return float("nan")
    running_max = window.cummax()
    dd = (window / running_max) - 1.0
    return dd.min()


def _window_volatility(returns: pd.Series, start: str, end: str) -> float:
    """Compute annualised volatility within a date window.

    Args:
        returns: Daily return Series with DatetimeIndex.
        start: Window start date string.
        end: Window end date string.

    Returns:
        Annualised volatility as a decimal.
    """
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    win  = returns[mask]
    if len(win) < 2:
        return float("nan")
    return win.std() * np.sqrt(252)


def analyze_event(
    event_name: str,
    event_params: dict,
    index_series: pd.Series,
    bench_series: pd.Series,
    index_returns: pd.Series,
    bench_returns: pd.Series,
) -> dict:
    """Analyse index and benchmark behaviour during a single geopolitical event.

    Computes crisis-period return, recovery return, relative performance,
    max drawdown, and annualised volatility for both index and benchmark.

    Args:
        event_name: Display name of the event.
        event_params: Dict with crisis_start/end and recovery_start/end dates.
        index_series: Cumulative index level Series.
        bench_series: Cumulative benchmark level Series.
        index_returns: Daily index return Series.
        bench_returns: Daily benchmark return Series.

    Returns:
        Dict with all computed metrics for the event.
    """
    cs, ce = event_params["crisis_start"],   event_params["crisis_end"]
    rs, re = event_params["recovery_start"], event_params["recovery_end"]

    idx_crisis   = _window_return(index_series, cs, ce)
    ben_crisis   = _window_return(bench_series, cs, ce)
    idx_recovery = _window_return(index_series, rs, re)
    ben_recovery = _window_return(bench_series, rs, re)
    idx_mdd      = _window_max_drawdown(index_series, cs, ce)
    ben_mdd      = _window_max_drawdown(bench_series, cs, ce)
    idx_vol      = _window_volatility(index_returns, cs, ce)
    ben_vol      = _window_volatility(bench_returns, cs, ce)

    return {
        "event":                     event_name,
        "crisis_start":              cs,
        "crisis_end":                ce,
        "recovery_start":            rs,
        "recovery_end":              re,
        "index_crisis_return":       round(idx_crisis,   4) if not np.isnan(idx_crisis)   else None,
        "benchmark_crisis_return":   round(ben_crisis,   4) if not np.isnan(ben_crisis)   else None,
        "crisis_relative_perf":      round(idx_crisis - ben_crisis, 4)
                                     if not (np.isnan(idx_crisis) or np.isnan(ben_crisis)) else None,
        "index_recovery_return":     round(idx_recovery, 4) if not np.isnan(idx_recovery) else None,
        "benchmark_recovery_return": round(ben_recovery, 4) if not np.isnan(ben_recovery) else None,
        "recovery_relative_perf":    round(idx_recovery - ben_recovery, 4)
                                     if not (np.isnan(idx_recovery) or np.isnan(ben_recovery)) else None,
        "index_max_drawdown":        round(idx_mdd, 4) if not np.isnan(idx_mdd) else None,
        "benchmark_max_drawdown":    round(ben_mdd, 4) if not np.isnan(ben_mdd) else None,
        "index_crisis_volatility":   round(idx_vol, 4) if not np.isnan(idx_vol) else None,
        "benchmark_crisis_volatility": round(ben_vol, 4) if not np.isnan(ben_vol) else None,
        "description":               event_params.get("description", ""),
    }


def run_all_stress_tests(
    index_df: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Run stress tests for all defined geopolitical events.

    Args:
        index_df: Backtest output DataFrame with 'index_value' and 'daily_return'.
        benchmark: Nifty 50 level Series.

    Returns:
        DataFrame with one row per event containing all stress test metrics.
    """
    idx_series = index_df["index_value"]
    idx_returns = index_df["daily_return"]

    bench_bt   = benchmark.reindex(idx_series.index).ffill()
    bench_norm = (bench_bt / bench_bt.iloc[0]) * 100
    bench_ret  = bench_bt.pct_change().fillna(0.0)

    results = []
    for name, params in EVENTS.items():
        logger.info(f"Analysing event: {name}")
        rec = analyze_event(
            name, params,
            idx_series, bench_norm,
            idx_returns, bench_ret,
        )
        results.append(rec)
        logger.info(
            f"  Crisis return — Index: {rec.get('index_crisis_return', 'N/A'):.2%}  "
            f"Bench: {rec.get('benchmark_crisis_return', 'N/A'):.2%}  "
            f"Rel: {rec.get('crisis_relative_perf', 'N/A'):.2%}"
            if rec.get("index_crisis_return") is not None else f"  Insufficient data for {name}"
        )

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Charts
# ══════════════════════════════════════════════════════════════════════════════

def plot_drawdown_analysis(
    index_df: pd.DataFrame,
    benchmark: pd.Series,
    out_path: str,
) -> None:
    """Chart 3 — Rolling drawdown for index and benchmark with event shading.

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

    idx_dd  = (idx_series  / idx_series.cummax())  - 1.0
    ben_dd  = (bench_norm  / bench_norm.cummax())  - 1.0

    ax.plot(idx_dd.index,  idx_dd  * 100, color=STYLE["accent1"], lw=1.8,
            label="Momentum Index Drawdown", zorder=3)
    ax.plot(ben_dd.index,  ben_dd  * 100, color=STYLE["accent2"], lw=1.5,
            linestyle="--", label="Nifty 50 Drawdown", zorder=2)
    ax.fill_between(idx_dd.index, idx_dd * 100, 0,
                    alpha=0.15, color=STYLE["accent1"], zorder=1)

    # Shade event windows
    event_list = [
        ("COVID-19\nCrash",        "2020-02-19", "2020-03-23", STYLE["accent4"]),
        ("Russia-\nUkraine",       "2022-02-24", "2022-03-08", STYLE["accent2"]),
        ("Israel-\nHamas",         "2023-10-07", "2023-10-26", STYLE["accent5"]),
    ]
    y_min = min(idx_dd.min(), ben_dd.min()) * 100 * 1.1

    for label, s, e, col in event_list:
        s_ts = pd.Timestamp(s)
        e_ts = pd.Timestamp(e)
        if s_ts <= idx_dd.index[-1]:
            ax.axvspan(s_ts, min(e_ts, idx_dd.index[-1]),
                       alpha=0.15, color=col, zorder=0)
            mid = s_ts + (min(e_ts, idx_dd.index[-1]) - s_ts) / 2
            ax.text(mid, y_min * 0.92, label, ha="center", va="top",
                    color=col, fontsize=7.5, fontweight="bold")

    ax.axhline(0, color=STYLE["border"], lw=0.8)
    ax.set_title("Rolling Drawdown Analysis — Momentum Index vs Nifty 50\n"
                 "Shaded regions: Geopolitical event windows (2020–2025)",
                 fontsize=13, fontweight="bold", pad=15, color=STYLE["text"])
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(loc="lower right", facecolor=STYLE["surface"],
              edgecolor=STYLE["border"], labelcolor=STYLE["text"], fontsize=10)
    ax.set_xlim(idx_dd.index[0], idx_dd.index[-1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=STYLE["bg"], edgecolor="none")
    plt.close()
    logger.info(f"Saved: {out_path}")


def plot_stress_test_comparison(
    stress_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Chart 4 — Grouped bar chart: crisis & recovery returns per event.

    Args:
        stress_df: DataFrame output from run_all_stress_tests().
        out_path: File path for the saved PNG.
    """
    df = stress_df.dropna(subset=["index_crisis_return", "benchmark_crisis_return"]).copy()
    if df.empty:
        logger.warning("No valid stress test data for chart; skipping.")
        return

    n_events = len(df)
    x = np.arange(n_events)
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    _apply_dark_style(fig, axes)

    titles = ["Crisis Period Return", "8-Week Recovery Return"]
    idx_cols  = ["index_crisis_return",   "index_recovery_return"]
    ben_cols  = ["benchmark_crisis_return", "benchmark_recovery_return"]

    for i, (ax, title, ic, bc) in enumerate(zip(axes, titles, idx_cols, ben_cols)):
        idx_vals = df[ic].values * 100
        ben_vals = df[bc].fillna(0).values * 100

        b1 = ax.bar(x - width / 2, idx_vals, width, label="Momentum Index",
                    color=STYLE["accent1"], edgecolor=STYLE["border"], linewidth=0.5)
        b2 = ax.bar(x + width / 2, ben_vals, width, label="Nifty 50",
                    color=STYLE["accent2"], edgecolor=STYLE["border"], linewidth=0.5)

        for bar, val in zip(b1, idx_vals):
            offset = 0.3 if val >= 0 else -1.2
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                    color=STYLE["accent1"], fontsize=8, fontweight="bold")
        for bar, val in zip(b2, ben_vals):
            offset = 0.3 if val >= 0 else -1.2
            ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                    f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                    color=STYLE["accent2"], fontsize=8, fontweight="bold")

        ax.axhline(0, color=STYLE["border"], lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["event"].tolist(), rotation=15, ha="right",
                           color=STYLE["text"], fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     pad=12, color=STYLE["text"])
        ax.set_ylabel("Return (%)", fontsize=10)
        ax.legend(facecolor=STYLE["surface"], edgecolor=STYLE["border"],
                  labelcolor=STYLE["text"], fontsize=9)

    fig.suptitle("Geopolitical Stress Test — NSE Momentum Index vs Nifty 50\n"
                 "*Iran-US: Post-project live test event (project completed Dec 2025)",
                 fontsize=13, fontweight="bold", color=STYLE["text"], y=1.01)
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
    print("NSE Momentum Factor Index — Geopolitical Stress Testing")
    print("Author: Hardik Gupta | December 2025 – January 2026")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, _, benchmark = load_data()
    index_df = pd.read_csv(
        os.path.join(RESULTS_DIR, "index_performance.csv"),
        index_col=0, parse_dates=True,
    )

    # Run stress tests
    stress_df = run_all_stress_tests(index_df, benchmark)
    stress_df.to_csv(
        os.path.join(RESULTS_DIR, "geopolitical_stress_tests.csv"), index=False
    )
    logger.info("Saved: geopolitical_stress_tests.csv")

    # Generate charts
    plot_drawdown_analysis(
        index_df, benchmark,
        os.path.join(RESULTS_DIR, "drawdown_analysis.png"),
    )
    plot_stress_test_comparison(
        stress_df,
        os.path.join(RESULTS_DIR, "stress_test_comparison.png"),
    )

    # Print results table
    print("\n── Stress Test Results ──────────────────────────────────────")
    display_cols = [
        "event", "index_crisis_return", "benchmark_crisis_return",
        "crisis_relative_perf", "index_recovery_return", "benchmark_recovery_return",
    ]
    disp = stress_df[display_cols].copy()
    for col in display_cols[1:]:
        disp[col] = disp[col].apply(
            lambda v: f"{v:.2%}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"
        )
    print(disp.to_string(index=False))
    print("─" * 65)
    print("✓ All stress tests complete. Project pipeline finished!")
    print("  Results saved to: results/")
