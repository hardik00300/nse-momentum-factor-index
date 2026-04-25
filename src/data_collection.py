"""
NSE Momentum Factor Index — Data Collection Module
Author: Hardik Gupta
Project Timeline: December 2025 - January 2026

Downloads and persists daily adjusted close prices for the NSE 100 universe,
market capitalisation data, and the Nifty 50 benchmark. All data is bounded
to December 31, 2025 to maintain authentic project backdating.
"""

import os
import time
import warnings
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_START_DATE: str = "2019-01-01"   # Extra history for 252-day lookback
DATA_END_DATE: str   = "2025-12-31"   # Hard ceiling — no data beyond this date
BACKTEST_START: str  = "2020-01-01"
BENCHMARK_TICKER: str = "^NSEI"
MAX_FFILL: int = 5                    # Max consecutive days to forward-fill

# ── NSE 100 Universe (55 liquid stocks) ───────────────────────────────────────
NSE_UNIVERSE: List[str] = [
    # Heavyweights
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "BAJFINANCE.NS",
    "HCLTECH.NS", "WIPRO.NS", "ULTRACEMCO.NS", "TITAN.NS", "SUNPHARMA.NS",
    # Financials & Insurance
    "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "INDUSINDBK.NS",
    # IT
    "TECHM.NS",
    # Pharma & Healthcare
    "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    # Consumer & FMCG
    "NESTLEIND.NS", "BRITANNIA.NS", "TATACONSUM.NS", "PIDILITIND.NS", "GODREJCP.NS",
    # Energy & Utilities
    "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "BPCL.NS", "COALINDIA.NS",
    # Metals & Mining
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",
    # Conglomerates & Infra
    "ADANIENT.NS", "ADANIPORTS.NS", "GRASIM.NS",
    # Autos & Capital Goods
    "TATAMOTORS.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "M&M.NS", "BAJAJ-AUTO.NS",
    # Cement & Building Materials
    "SHREECEM.NS", "HAVELLS.NS",
    # Retail & Diversified
    "DMART.NS", "BOSCHLTD.NS",
]

# ── Directory helpers ─────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def _ensure_dirs() -> None:
    """Create data/ and results/ directories if they don't already exist."""
    for d in [DATA_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)


# ── Core Download Functions ───────────────────────────────────────────────────

def download_price_data(
    tickers: List[str],
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
    max_ffill: int = MAX_FFILL,
) -> pd.DataFrame:
    """Download adjusted close prices for NSE tickers via yfinance.

    Args:
        tickers: List of Yahoo Finance ticker symbols (e.g. 'RELIANCE.NS').
        start_date: Data start date string 'YYYY-MM-DD'.
        end_date: Data end date string 'YYYY-MM-DD'. Hard ceiling for backdating.
        max_ffill: Maximum consecutive days to forward-fill (handles NSE holidays).

    Returns:
        DataFrame[date × ticker] of adjusted close prices. Tickers with fewer
        than 252 valid trading days in the backtest window are dropped.
    """
    logger.info(f"Downloading prices for {len(tickers)} tickers ({start_date} → {end_date})...")
    # Download in batches with retries to avoid rate limiting
    BATCH_SIZE = 10
    MAX_RETRIES = 4
    all_frames = []

    for batch_start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[batch_start: batch_start + BATCH_SIZE]
        logger.info(f"  Batch {batch_start//BATCH_SIZE + 1}: {batch}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if not raw.empty:
                    all_frames.append(raw)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(f"  Attempt {attempt} failed ({exc}). Retrying in {wait}s...")
                time.sleep(wait)
        else:
            logger.warning(f"  All retries exhausted for batch {batch}. Skipping.")
        time.sleep(1.5)  # Polite delay between batches

    if not all_frames:
        raise RuntimeError("All batches failed to download. Check network/rate limits.")

    # Merge all batches
    try:
        raw = pd.concat(all_frames, axis=1)
    except Exception as exc:
        raise RuntimeError(f"Failed to merge batch results: {exc}") from exc

    # Extract Close prices from possibly multi-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Enforce hard date ceiling (no data after Dec 31, 2025)
    prices = prices[prices.index <= pd.Timestamp(end_date)]

    # Forward-fill gaps (holidays, trading halts) — max 5 consecutive days
    prices = prices.ffill(limit=max_ffill)

    # Drop tickers lacking 252+ valid days in the backtest window
    bt_mask      = prices.index >= pd.Timestamp(BACKTEST_START)
    valid_counts = prices.loc[bt_mask].notna().sum()
    to_drop      = valid_counts[valid_counts < 252].index.tolist()
    if to_drop:
        warnings.warn(
            f"Dropping {len(to_drop)} tickers with <252 valid days: {to_drop}",
            UserWarning, stacklevel=2,
        )
        prices = prices.drop(columns=to_drop, errors="ignore")

    logger.info(f"Price data ready: {prices.shape[1]} stocks × {prices.shape[0]} days")
    return prices


def download_market_caps(tickers: List[str]) -> pd.Series:
    """Fetch current market capitalisation for each ticker (used as static proxy).

    Note:
        yfinance only provides the most recent market cap. This is used as a
        free-float market cap proxy for the entire backtest period — a standard
        simplification acknowledged in the methodology.

    Args:
        tickers: List of Yahoo Finance ticker symbols.

    Returns:
        Series[ticker → market_cap_INR]. Missing values are imputed with the
        median market cap of successfully fetched tickers.
    """
    logger.info(f"Fetching market caps for {len(tickers)} tickers...")
    caps: Dict[str, float] = {}

    for i, ticker in enumerate(tickers, 1):
        for attempt in range(1, 4):
            try:
                info = yf.Ticker(ticker).fast_info
                mc   = getattr(info, "market_cap", None) or 0
                if mc and mc > 0:
                    caps[ticker] = float(mc)
                else:
                    # Fallback: use shares * last price
                    shares = getattr(info, "shares", None)
                    price  = getattr(info, "last_price", None)
                    if shares and price and shares > 0 and price > 0:
                        caps[ticker] = float(shares * price)
                break
            except Exception as exc:
                if attempt < 3:
                    time.sleep(1.5)
                else:
                    warnings.warn(f"Market cap fetch failed for {ticker}: {exc}", UserWarning, stacklevel=2)
        time.sleep(0.3)
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(tickers)}")

    series  = pd.Series(caps, name="market_cap", dtype=float)
    median  = series.median() if len(series) > 0 else 1e12
    missing = [t for t in tickers if t not in series.index]
    if missing:
        imputed = pd.Series({t: median for t in missing}, name="market_cap", dtype=float)
        series  = pd.concat([series, imputed])
        logger.warning(f"Imputed median cap ({median:,.0f}) for {len(missing)} tickers: {missing}")

    logger.info(f"Market caps ready: {len(series)} tickers")
    return series


def download_benchmark(
    ticker: str = BENCHMARK_TICKER,
    start_date: str = DATA_START_DATE,
    end_date: str = DATA_END_DATE,
) -> pd.Series:
    """Download Nifty 50 index levels from Yahoo Finance.

    Args:
        ticker: Benchmark symbol (default '^NSEI').
        start_date: Start date string.
        end_date: End date string. Hard ceiling — no data after Dec 31, 2025.

    Returns:
        Series[date → close_level] named 'Nifty50', forward-filled up to 5 days.
    """
    logger.info(f"Downloading benchmark: {ticker} ({start_date} → {end_date})")
    for attempt in range(1, 6):
        try:
            raw = yf.download(ticker, start=start_date, end=end_date,
                              auto_adjust=True, progress=False, threads=False)
            if not raw.empty:
                break
            logger.warning(f"Empty data on attempt {attempt}, retrying...")
        except Exception as exc:
            logger.warning(f"Benchmark attempt {attempt} failed: {exc}")
        time.sleep(3 * attempt)
    else:
        raise RuntimeError(f"Failed to download benchmark {ticker} after 5 attempts.")

    if raw.empty:
        raise RuntimeError(f"Empty data returned for benchmark {ticker}.")

    close = raw["Close"].squeeze()
    close = close[close.index <= pd.Timestamp(end_date)]
    close = close.ffill(limit=5)
    close.name = "Nifty50"
    logger.info(f"Benchmark ready: {len(close)} trading days")
    return close


# ── Persistence ───────────────────────────────────────────────────────────────

def save_data(
    prices: pd.DataFrame,
    market_caps: pd.Series,
    benchmark: pd.Series,
) -> None:
    """Persist all downloaded data to the data/ directory.

    Args:
        prices: Adjusted close price DataFrame.
        market_caps: Market cap Series.
        benchmark: Nifty 50 level Series.
    """
    _ensure_dirs()
    prices.to_csv(os.path.join(DATA_DIR, "nse_prices.csv"))
    market_caps.to_frame("market_cap").to_csv(os.path.join(DATA_DIR, "market_caps.csv"))
    benchmark.to_frame("Nifty50").to_csv(os.path.join(DATA_DIR, "nifty50_benchmark.csv"))
    logger.info(f"All data saved to: {DATA_DIR}")


def load_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load previously saved datasets from the data/ directory.

    Returns:
        Tuple of (prices_df, market_caps_series, benchmark_series).

    Raises:
        FileNotFoundError: If any required CSV is missing.
    """
    paths = {
        "prices":      os.path.join(DATA_DIR, "nse_prices.csv"),
        "market_caps": os.path.join(DATA_DIR, "market_caps.csv"),
        "benchmark":   os.path.join(DATA_DIR, "nifty50_benchmark.csv"),
    }
    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing data file: {path}. Run data_collection.py first.")

    prices      = pd.read_csv(paths["prices"],      index_col=0, parse_dates=True)
    market_caps = pd.read_csv(paths["market_caps"], index_col=0).iloc[:, 0]
    benchmark   = pd.read_csv(paths["benchmark"],   index_col=0, parse_dates=True).iloc[:, 0]
    benchmark.name = "Nifty50"

    logger.info(f"Loaded: prices={prices.shape}, market_caps={len(market_caps)}, benchmark={len(benchmark)}")
    return prices, market_caps, benchmark


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("NSE Momentum Factor Index — Data Collection")
    print("Author: Hardik Gupta | December 2025 – January 2026")
    print("=" * 65)

    _ensure_dirs()

    prices      = download_price_data(NSE_UNIVERSE)
    valid_ticks = prices.columns.tolist()
    market_caps = download_market_caps(valid_ticks)
    benchmark   = download_benchmark()

    save_data(prices, market_caps, benchmark)

    print("\n── Summary ───────────────────────────────────────────────────")
    print(f"  Stocks        : {prices.shape[1]}")
    print(f"  Date range    : {prices.index.min().date()} → {prices.index.max().date()}")
    print(f"  Trading days  : {len(prices)}")
    print(f"  Benchmark days: {len(benchmark)}")
    print("─" * 65)
    print("✓ Done. Run index_construction.py next.")
