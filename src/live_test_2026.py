"""
NSE Momentum Factor Index — April 2026 Live Test Addendum
Author: Hardik Gupta
Date: April 2026

Simulates an out-of-sample forward test of the final momentum portfolio 
constructed in December 2025 through the early-2026 Iran-US escalations.
"""

import os
import sys
import logging
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stress_testing import STYLE, _apply_dark_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

EVENT_START = "2026-01-01"
EVENT_END   = "2026-04-20"


def run_live_test():
    logger.info("Starting April 2026 Live Test...")

    # 1. Load the frozen December 2025 portfolio
    rebal_path = os.path.join(RESULTS_DIR, "rebalancing_calendar.csv")
    if not os.path.exists(rebal_path):
        logger.error("Could not find rebalancing calendar. Run the main pipeline first.")
        return

    rebal_df = pd.read_csv(rebal_path)
    latest_rebal = rebal_df.iloc[-1]
    constituents = latest_rebal["constituents"].split("|")
    logger.info(f"Loaded frozen portfolio from {latest_rebal['rebal_date']}: {len(constituents)} stocks")

    logger.info(f"Downloading 2026 data ({EVENT_START} to {EVENT_END})...")
    import time
    BATCH_SIZE = 10
    MAX_RETRIES = 4
    all_frames = []

    for batch_start in range(0, len(constituents), BATCH_SIZE):
        batch = constituents[batch_start: batch_start + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = yf.download(batch, start=EVENT_START, end=EVENT_END, auto_adjust=True, progress=False, threads=False)
                if not raw.empty:
                    # Handle single ticker edge case in batch
                    if isinstance(raw.columns, pd.MultiIndex):
                        all_frames.append(raw["Close"])
                    else:
                        all_frames.append(raw[["Close"]].rename(columns={"Close": batch[0]}))
                break
            except Exception as exc:
                time.sleep(2 ** attempt)
        time.sleep(1.5)
        
    if not all_frames:
        logger.error("Failed to download any data due to rate limits.")
        return

    raw_prices = pd.concat(all_frames, axis=1)

    for attempt in range(1, 6):
        try:
            raw_bench = yf.download("^NSEI", start=EVENT_START, end=EVENT_END, auto_adjust=True, progress=False, threads=False)
            if not raw_bench.empty:
                raw_bench = raw_bench["Close"]
                break
        except Exception:
            time.sleep(2 * attempt)
    else:
        logger.error("Failed to download benchmark.")
        return

    prices = raw_prices.ffill(limit=3).bfill()
    bench  = raw_bench.ffill(limit=3).squeeze()

    # Equal weight for simplicity of forward-testing, or proxy market cap
    daily_returns = prices.pct_change().fillna(0.0)
    # Equal weight proxy:
    idx_returns = daily_returns.mean(axis=1) 
    
    idx_cum = (1 + idx_returns).cumprod() * 100
    idx_cum.iloc[0] = 100.0

    ben_returns = bench.pct_change().fillna(0.0)
    ben_cum = (1 + ben_returns).cumprod() * 100
    ben_cum.iloc[0] = 100.0

    # Align dates
    aligned = pd.concat([idx_cum, ben_cum], axis=1).dropna()
    aligned.columns = ["Index", "Nifty50"]

    # Calculate returns
    total_idx_ret = (aligned["Index"].iloc[-1] / 100) - 1
    total_ben_ret = (aligned["Nifty50"].iloc[-1] / 100) - 1

    logger.info(f"--- 2026 Live Test Results ---")
    logger.info(f"Momentum Index Return: {total_idx_ret:.2%}")
    logger.info(f"Nifty 50 Return      : {total_ben_ret:.2%}")
    logger.info(f"Relative Performance : {total_idx_ret - total_ben_ret:.2%}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, ax)

    ax.plot(aligned.index, aligned["Index"], color=STYLE["accent1"], lw=2, label=f"Momentum Index ({total_idx_ret:.2%})")
    ax.plot(aligned.index, aligned["Nifty50"], color=STYLE["accent2"], lw=2, linestyle="--", label=f"Nifty 50 ({total_ben_ret:.2%})")
    
    ax.set_title("April 2026 Live Test — Out-of-Sample Forward Test\n(Frozen Dec 2025 Portfolio vs Nifty 50)",
                 fontsize=12, fontweight="bold", pad=15, color=STYLE["text"])
    ax.legend(facecolor=STYLE["surface"], edgecolor=STYLE["border"], labelcolor=STYLE["text"])
    
    out_path = os.path.join(RESULTS_DIR, "live_test_2026.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, facecolor=STYLE["bg"])
    plt.close()
    
    logger.info(f"Saved live test chart to: {out_path}")


if __name__ == "__main__":
    run_live_test()
