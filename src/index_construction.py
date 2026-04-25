"""
NSE Momentum Factor Index — Index Construction Module
Author: Hardik Gupta
Project Timeline: December 2025 - January 2026

Implements the full momentum factor index methodology:
  - 12-month (252-day) trailing return as the ranking signal
  - Top 50 stocks selected at each quarterly rebalance
  - Market-cap proportional weighting
  - Full backtest engine from Jan 1, 2020 to Dec 31, 2025

Run after data_collection.py has saved data to the data/ directory.
"""

import os
import sys
import warnings
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running from project root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BACKTEST_START:  str   = "2020-01-01"
BACKTEST_END:    str   = "2025-12-31"
LOOKBACK_PERIOD: int   = 252
TOP_N:           int   = 50
INDEX_BASE:      float = 100.0

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ══════════════════════════════════════════════════════════════════════════════
# Core Factor Functions
# ══════════════════════════════════════════════════════════════════════════════

def calculate_momentum(
    price_data: pd.DataFrame,
    date: pd.Timestamp,
    lookback_period: int = LOOKBACK_PERIOD,
) -> pd.Series:
    """Calculate 12-month trailing price momentum for all stocks.

    Momentum is defined as total return over the prior `lookback_period`
    trading days: (P_t / P_{t-252}) - 1. Stocks with insufficient price
    history return NaN and are excluded from selection.

    Args:
        price_data: DataFrame of adjusted close prices (index=dates, cols=tickers).
        date: Rebalancing date at which momentum is evaluated.
        lookback_period: Number of trading days for the return window (default 252).

    Returns:
        Series of momentum scores indexed by ticker, sorted descending.
        Empty Series returned if insufficient history exists.
    """
    available = price_data.index[price_data.index <= date]
    if len(available) < lookback_period + 1:
        warnings.warn(
            f"Insufficient history on {date.date()} for {lookback_period}-day lookback.",
            UserWarning, stacklevel=2,
        )
        return pd.Series(dtype=float)

    t_now   = available[-1]
    t_start = available[-lookback_period]

    end_px   = price_data.loc[t_now]
    start_px = price_data.loc[t_start]

    # Total return: skip division where start price is 0 or NaN
    momentum = (end_px / start_px.replace(0, np.nan)) - 1.0
    momentum = momentum.dropna().sort_values(ascending=False)
    return momentum


def select_constituents(
    momentum_scores: pd.Series,
    n: int = TOP_N,
) -> List[str]:
    """Select the top N stocks by momentum score.

    Args:
        momentum_scores: Series of momentum scores (descending) indexed by ticker.
        n: Number of constituents to select.

    Returns:
        List of selected ticker symbols (top N by momentum).
    """
    if momentum_scores.empty:
        return []
    n = min(n, len(momentum_scores))
    return momentum_scores.nlargest(n).index.tolist()


def calculate_weights(
    constituents: List[str],
    market_caps: pd.Series,
) -> Dict[str, float]:
    """Compute market-cap proportional weights for selected constituents.

    Args:
        constituents: List of selected ticker symbols.
        market_caps: Series of market cap values indexed by ticker.

    Returns:
        Dictionary mapping ticker -> weight (weights sum to 1.0).
        Tickers missing market cap data are assigned the median cap.
    """
    if not constituents:
        return {}

    median_cap = market_caps.median()
    caps = pd.Series(
        {t: market_caps.get(t, median_cap) for t in constituents},
        dtype=float,
    )
    caps = caps.clip(lower=1.0)          # Guard against zero/negative caps
    total = caps.sum()
    weights = (caps / total).to_dict()
    return weights


def calculate_turnover(
    prev_constituents: List[str],
    new_constituents: List[str],
) -> float:
    """Calculate portfolio turnover rate between two consecutive rebalances.

    Turnover is the fraction of the portfolio replaced, measured as the weight
    of stocks that exit the index (assuming equal weight for simplicity here).

    Args:
        prev_constituents: List of tickers held before rebalancing.
        new_constituents: List of tickers selected after rebalancing.

    Returns:
        Turnover rate as a float in [0, 1]. 0 = no change, 1 = full replacement.
    """
    if not prev_constituents:
        return 0.0
    prev_set = set(prev_constituents)
    new_set  = set(new_constituents)
    exits    = prev_set - new_set   # stocks sold
    # Turnover = fraction of old portfolio that was sold
    return len(exits) / len(prev_set)


def rebalance_portfolio(
    date: pd.Timestamp,
    price_data: pd.DataFrame,
    market_caps: pd.Series,
    lookback_period: int = LOOKBACK_PERIOD,
    n: int = TOP_N,
) -> Tuple[List[str], Dict[str, float]]:
    """Execute one full rebalancing cycle on a given date.

    Args:
        date: Rebalancing date.
        price_data: Full price DataFrame.
        market_caps: Market cap Series.
        lookback_period: Momentum lookback in trading days.
        n: Number of constituents to select.

    Returns:
        Tuple of (constituents_list, weights_dict).
    """
    scores      = calculate_momentum(price_data, date, lookback_period)
    constituents = select_constituents(scores, n)
    weights      = calculate_weights(constituents, market_caps)
    return constituents, weights


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Engine
# ══════════════════════════════════════════════════════════════════════════════

def _get_rebalance_dates(
    price_index: pd.DatetimeIndex,
    start_date: str,
    end_date: str,
) -> List[pd.Timestamp]:
    """Derive quarterly rebalancing dates (last trading day of Mar/Jun/Sep/Dec).

    Args:
        price_index: DatetimeIndex of available trading days.
        start_date: Backtest start date string.
        end_date: Backtest end date string.

    Returns:
        List of rebalancing Timestamps (last trading day of each quarter-end month).
    """
    trading_days = price_index[
        (price_index >= pd.Timestamp(start_date)) &
        (price_index <= pd.Timestamp(end_date))
    ]
    # Group by (year, quarter) and take the last trading day of each quarter
    df = pd.DataFrame({"date": trading_days})
    df["quarter"] = df["date"].dt.to_period("Q")
    rebal_dates = df.groupby("quarter")["date"].max().tolist()
    return sorted(rebal_dates)


def backtest_index(
    price_data: pd.DataFrame,
    market_caps: pd.Series,
    start_date: str = BACKTEST_START,
    end_date: str   = BACKTEST_END,
    rebalance_freq: str = "Q",
    lookback_period: int = LOOKBACK_PERIOD,
    n: int = TOP_N,
    index_base: float = INDEX_BASE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full backtest of the NSE Momentum Factor Index.

    The index is computed by:
      1. Rebalancing quarterly (last trading day of Mar/Jun/Sep/Dec).
      2. At each rebalance: rank by 252-day momentum, select top 50, weight by market cap.
      3. Track daily index level as weighted average of constituent daily returns.
      4. Record constituents, turnover, and weights at each rebalance.

    Args:
        price_data: DataFrame of adjusted close prices.
        market_caps: Market cap Series for weighting.
        start_date: Backtest start date (default 2020-01-01).
        end_date: Backtest end date (default 2025-12-31).
        rebalance_freq: Rebalancing frequency (only 'Q' implemented).
        lookback_period: Momentum lookback in trading days.
        n: Number of constituents.
        index_base: Base index value at start (default 100).

    Returns:
        Tuple of:
          - index_df: DataFrame with columns [date, index_value, daily_return, constituents].
          - rebal_df: DataFrame with rebalancing calendar details.
    """
    logger.info(f"Starting backtest: {start_date} → {end_date}")

    bt_prices = price_data[
        (price_data.index >= pd.Timestamp(start_date)) &
        (price_data.index <= pd.Timestamp(end_date))
    ].copy()

    rebal_dates = _get_rebalance_dates(price_data.index, start_date, end_date)
    logger.info(f"Rebalancing dates: {len(rebal_dates)} quarters identified")

    # Daily return matrix
    daily_returns = bt_prices.pct_change().fillna(0.0)

    # ── State ────────────────────────────────────────────────────────────────
    index_values:    List[float]         = []
    daily_ret_list:  List[float]         = []
    date_list:       List[pd.Timestamp]  = []
    constituents_by_date: Dict[pd.Timestamp, List[str]] = {}

    rebal_records: List[dict] = []

    current_weights:      Dict[str, float] = {}
    current_constituents: List[str]        = []
    prev_constituents:    List[str]        = []
    index_value: float = index_base
    rebal_idx:   int   = 0

    trading_days = bt_prices.index.tolist()

    for i, date in enumerate(trading_days):
        # Rebalance if this date is (or has passed) the next scheduled rebalance
        while rebal_idx < len(rebal_dates) and date >= rebal_dates[rebal_idx]:
            rebal_date = rebal_dates[rebal_idx]
            prev_constituents    = current_constituents.copy()
            current_constituents, current_weights = rebalance_portfolio(
                rebal_date, price_data, market_caps, lookback_period, n
            )
            turnover = calculate_turnover(prev_constituents, current_constituents)

            added   = list(set(current_constituents) - set(prev_constituents))
            dropped = list(set(prev_constituents)    - set(current_constituents))

            rebal_records.append({
                "rebal_date":   rebal_date.date(),
                "n_constituents": len(current_constituents),
                "turnover_rate":  round(turnover, 4),
                "stocks_added":   len(added),
                "stocks_dropped": len(dropped),
                "added":   "|".join(sorted(added)),
                "dropped": "|".join(sorted(dropped)),
                "constituents": "|".join(sorted(current_constituents)),
            })
            logger.info(
                f"  Rebalanced {rebal_date.date()}: {len(current_constituents)} stocks, "
                f"turnover={turnover:.1%}, +{len(added)} -{len(dropped)}"
            )
            rebal_idx += 1

        # Daily index return = weighted sum of constituent returns
        if current_weights and i > 0:
            day_ret = sum(
                w * daily_returns.loc[date, t]
                for t, w in current_weights.items()
                if t in daily_returns.columns and not np.isnan(daily_returns.loc[date, t])
            )
        else:
            day_ret = 0.0

        index_value *= (1.0 + day_ret)
        index_values.append(index_value)
        daily_ret_list.append(day_ret)
        date_list.append(date)
        constituents_by_date[date] = current_constituents.copy()

    # ── Build output DataFrames ──────────────────────────────────────────────
    index_df = pd.DataFrame({
        "date":        date_list,
        "index_value": index_values,
        "daily_return": daily_ret_list,
    }).set_index("date")
    index_df["constituents"] = [
        "|".join(sorted(constituents_by_date.get(d, []))) for d in date_list
    ]

    rebal_df = pd.DataFrame(rebal_records)

    logger.info(
        f"Backtest complete. Final index value: {index_values[-1]:.2f} "
        f"(base {index_base})"
    )
    return index_df, rebal_df


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_collection import load_data

    print("=" * 65)
    print("NSE Momentum Factor Index — Index Construction")
    print("Author: Hardik Gupta | December 2025 – January 2026")
    print("=" * 65)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    prices, market_caps, benchmark = load_data()

    # Run backtest
    index_df, rebal_df = backtest_index(prices, market_caps)

    # Save outputs
    perf_path  = os.path.join(RESULTS_DIR, "index_performance.csv")
    rebal_path = os.path.join(RESULTS_DIR, "rebalancing_calendar.csv")

    index_df.drop(columns=["constituents"]).to_csv(perf_path)
    rebal_df.to_csv(rebal_path, index=False)

    print(f"\n── Backtest Summary ──────────────────────────────────────────")
    print(f"  Date range       : {index_df.index.min().date()} → {index_df.index.max().date()}")
    print(f"  Trading days     : {len(index_df)}")
    print(f"  Rebalances       : {len(rebal_df)}")
    print(f"  Starting value   : {INDEX_BASE:.1f}")
    print(f"  Final value      : {index_df['index_value'].iloc[-1]:.2f}")
    total_return = (index_df["index_value"].iloc[-1] / INDEX_BASE - 1) * 100
    years = (index_df.index[-1] - index_df.index[0]).days / 365.25
    cagr  = ((index_df["index_value"].iloc[-1] / INDEX_BASE) ** (1 / years) - 1) * 100
    print(f"  Total return     : {total_return:.1f}%")
    print(f"  CAGR             : {cagr:.2f}%")
    avg_turnover = rebal_df["turnover_rate"].mean() if not rebal_df.empty else 0
    print(f"  Avg turnover/qtr : {avg_turnover:.1%}")
    print(f"─" * 65)
    print(f"  Saved: {perf_path}")
    print(f"  Saved: {rebal_path}")
    print("✓ Done. Run performance_analysis.py next.")
