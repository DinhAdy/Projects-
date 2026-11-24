# -*- coding: utf-8 -*-
"""
=======================================================================================
Module Name: market_db.py
Description: High-performance SQLite persistence and incremental caching engine
             for daily market prices and volumes (stocks, factors, and macro anchors).
Author Name: Andy Dinh
License & Copyright:
    Copyright (c) 2026 Andy Dinh. All Rights Reserved.
    PROPRIETARY AND CONFIDENTIAL. Strictly NO permission is granted for unauthorized 
    distribution, duplication, modification, or public cloning of this source logic.
=======================================================================================
"""

import os
import sqlite3
import datetime
try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
except ImportError:
    import importlib.util
    import subprocess
    import sys
    print("  INSTALLING NECESSARY STUFF NOW...", end="\n\n")
    if importlib.util.find_spec("pip") is None:
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Could not bootstrap pip for this Python interpreter. "
                "Install pip, then run the script again."
            ) from error
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "yfinance", "numpy", "pandas", 
        ])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Could not install the required packages. Check internet access and "
            "package installation permissions, then run the script again."
        ) from error
    import yfinance as yf
    import numpy as np
    import pandas as pd

DB_FILENAME = "market_data.db"

# Core factors and macro tickers required by the Monte Carlo and Quant Screener engines
DEFAULT_FACTORS = ["SPY", "IWM", "IWD", "IWF", "TLT", "GLD", "XLV", "XLE"]
DEFAULT_MACRO = ["^VIX", "^IRX", "^GSPC"]

PERIOD_LOOKBACK_DAYS = {
    "1mo": 21,
    "3mo": 63,
    "6mo": 126,
    "1y": 252,
    "2y": 504,
    "3y": 756,
    "5y": 1260,
    "10y": 2520,
    "max": None,
}


def get_db_path(db_name=DB_FILENAME):
    """Return the absolute path to the SQLite database file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, db_name)


def init_db(db_path=None):
    """Initialize the SQLite schema if it does not already exist."""
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                close REAL NOT NULL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON daily_prices (ticker, date)
        """)
    conn.close()


def get_tickers_status(tickers, db_path=None):
    """
    Check the current database coverage for a list of tickers.
    Returns: dict {ticker: {'count': int, 'min_date': str|None, 'max_date': str|None}}
    """
    init_db(db_path)
    path = db_path or get_db_path()
    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    status = {}
    for ticker in tickers:
        cursor.execute(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM daily_prices WHERE ticker = ?",
            (ticker,)
        )
        count, min_date, max_date = cursor.fetchone()
        status[ticker] = {
            "count": count,
            "min_date": min_date,
            "max_date": max_date,
        }
    conn.close()
    return status


def save_market_data(close_df, volume_df=None, db_path=None):
    """
    Save or update close and volume data into the SQLite database.
    Vectorized unpivoting is used for maximum speed.
    """
    if close_df is None or close_df.empty:
        return 0

    init_db(db_path)
    path = db_path or get_db_path()

    # Ensure consistent DataFrame shape
    if isinstance(close_df, pd.Series):
        name = close_df.name or "UNKNOWN"
        close_df = close_df.to_frame(name)

    # Standardize index name
    close_df = close_df.copy()
    close_df.index = pd.to_datetime(close_df.index)
    close_df.index.name = "Date"

    # Melt Close
    close_melt = close_df.reset_index().melt(
        id_vars=["Date"], var_name="ticker", value_name="close"
    )

    # Melt Volume if available
    if volume_df is not None and not volume_df.empty:
        if isinstance(volume_df, pd.Series):
            name = volume_df.name or "UNKNOWN"
            volume_df = volume_df.to_frame(name)
        volume_df = volume_df.copy()
        volume_df.index = pd.to_datetime(volume_df.index)
        volume_df.index.name = "Date"
        volume_melt = volume_df.reset_index().melt(
            id_vars=["Date"], var_name="ticker", value_name="volume"
        )
        merged = pd.merge(close_melt, volume_melt, on=["Date", "ticker"], how="left")
    else:
        merged = close_melt
        merged["volume"] = 0.0

    merged["volume"] = merged["volume"].fillna(0.0)
    merged["date"] = merged["Date"].dt.strftime("%Y-%m-%d")

    # Drop NaNs in close
    cleaned = merged[["date", "ticker", "close", "volume"]].dropna(subset=["close"])
    records = list(cleaned.itertuples(index=False, name=None))

    if not records:
        return 0

    conn = sqlite3.connect(path)
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO daily_prices (date, ticker, close, volume) VALUES (?, ?, ?, ?)",
            records,
        )
    conn.close()
    return len(records)


def sync_market_data(
    tickers,
    include_factors=True,
    default_lookback_years=5,
    db_path=None,
    force=False,
):
    """
    Intelligently synchronize market data into SQLite.
    - If a ticker is new or has < 252 days: downloads full 5-year history.
    - If a ticker already has history: ONLY downloads from the last stored date to today.
    - Automatically includes factor ETFs and macro indices if include_factors=True.
    """
    init_db(db_path)

    # Compile universe
    combined = list(tickers)
    if include_factors:
        combined += DEFAULT_FACTORS + DEFAULT_MACRO
    universe = list(dict.fromkeys(combined))

    status = get_tickers_status(universe, db_path=db_path)
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    needs_full = []
    needs_incremental = []

    for t in universe:
        info = status[t]
        if force or info["count"] < 252 or info["max_date"] is None:
            needs_full.append(t)
        else:
            needs_incremental.append((t, info["max_date"]))

    threads = min(8, max(1, ((os.cpu_count() or 2) // 2)))

    # 1. Full 5-year initial baseline download for new/insufficient tickers
    if needs_full:
        print(f"\n[SQL SYNC] Fetching {default_lookback_years}y historical baseline for {len(needs_full)} new tickers from Yahoo Finance...")
        print(f"           Tickers: {', '.join(needs_full)}")
        full_data = yf.download(
            needs_full,
            period=f"{default_lookback_years}y",
            auto_adjust=True,
            threads=threads,
            progress=False,
        )
        if not full_data.empty and "Close" in full_data:
            c = full_data["Close"]
            v = full_data["Volume"] if "Volume" in full_data else None
            n_rows = save_market_data(c, v, db_path=db_path)
            print(f"[SQL SYNC] Saved {n_rows} baseline records into SQLite.")

    # 2. Incremental fetch for tickers that already have baseline
    if needs_incremental:
        # Check the earliest max_date among existing tickers
        earliest_last_date = min(item[1] for item in needs_incremental)
        tickers_to_update = [item[0] for item in needs_incremental]

        # If data is already from today or market hasn't traded since, check if update is needed
        if earliest_last_date >= today_str and not force:
            print(f"[SQL CACHE] All {len(tickers_to_update)} existing tickers are already up to date ({earliest_last_date}).")
        else:
            print(f"\n[SQL SYNC] Incremental fetch: pulling only data since {earliest_last_date} for {len(tickers_to_update)} tickers...")
            inc_data = yf.download(
                tickers_to_update,
                start=earliest_last_date,
                auto_adjust=True,
                threads=threads,
                progress=False,
            )
            if not inc_data.empty and "Close" in inc_data:
                c = inc_data["Close"]
                v = inc_data["Volume"] if "Volume" in inc_data else None
                n_rows = save_market_data(c, v, db_path=db_path)
                print(f"[SQL SYNC] Incremental update complete: {n_rows} records added/updated in SQLite.")
            else:
                print(f"[SQL CACHE] No new market days available beyond {earliest_last_date}.")

    return universe


def backfill_market_data(
    tickers,
    start_date,
    end_date,
    db_path=None,
):
    """Fetch and persist older price history without replacing newer cached rows."""
    ticker_list = list(dict.fromkeys(tickers))
    if not ticker_list:
        raise ValueError("At least one ticker is required for historical backfill.")
    if start_date >= end_date:
        raise ValueError("Backfill start_date must be earlier than end_date.")

    threads = min(8, max(1, ((os.cpu_count() or 2) // 2)))
    print(
        f"[SQL BACKFILL] Fetching {len(ticker_list)} assets "
        f"from {start_date} through {end_date}..."
    )
    historical_data = yf.download(
        ticker_list,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        threads=threads,
        progress=False,
    )
    if historical_data.empty or "Close" not in historical_data:
        print("[SQL BACKFILL] No historical rows returned for this date range.")
        return 0

    close_data = historical_data["Close"]
    volume_data = (
        historical_data["Volume"] if "Volume" in historical_data else None
    )
    saved_rows = save_market_data(
        close_data,
        volume_data,
        db_path=db_path,
    )
    print(f"[SQL BACKFILL] Saved {saved_rows} older price rows.")
    return saved_rows


def load_market_data(tickers, period=None, min_date=None, db_path=None):
    """
    Load close and volume DataFrames directly from the local SQLite database.
    Returns: (close_df, volume_df) indexed by pd.DatetimeIndex named 'Date',
             with columns exactly matching tickers.
    """
    init_db(db_path)
    path = db_path or get_db_path()
    ticker_list = list(dict.fromkeys(tickers))

    if not ticker_list:
        return pd.DataFrame(), pd.DataFrame()

    placeholders = ",".join(["?"] * len(ticker_list))
    conn = sqlite3.connect(path)

    if min_date:
        query = f"""
            SELECT date, ticker, close, volume 
            FROM daily_prices 
            WHERE ticker IN ({placeholders}) AND date >= ? 
            ORDER BY date ASC
        """
        params = ticker_list + [min_date]
    else:
        query = f"""
            SELECT date, ticker, close, volume 
            FROM daily_prices 
            WHERE ticker IN ({placeholders}) 
            ORDER BY date ASC
        """
        params = ticker_list

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=ticker_list), pd.DataFrame(columns=ticker_list)

    df["date"] = pd.to_datetime(df["date"])
    close_pivot = df.pivot(index="date", columns="ticker", values="close")
    volume_pivot = df.pivot(index="date", columns="ticker", values="volume")

    close_pivot.index.name = "Date"
    volume_pivot.index.name = "Date"

    # Reindex columns to guarantee all requested tickers are present in order
    close_pivot = close_pivot.reindex(columns=ticker_list)
    volume_pivot = volume_pivot.reindex(columns=ticker_list)

    # Apply period slicing if requested (e.g. '1y' -> last 252 days)
    if period:
        lookback_days = PERIOD_LOOKBACK_DAYS.get(period)
        if lookback_days and len(close_pivot) > lookback_days:
            close_pivot = close_pivot.tail(lookback_days)
            volume_pivot = volume_pivot.tail(lookback_days)

    return close_pivot, volume_pivot


if __name__ == "__main__":
    print(f"Database file: {get_db_path()}")
    init_db()
    test_tickers = ["AAPL", "MSFT", "SPY", "^VIX"]
    print(f"Checking status for {test_tickers}...")
    st = get_tickers_status(test_tickers)
    for k, v in st.items():
        print(f"  {k:8s}: count={v['count']}, min={v['min_date']}, max={v['max_date']}")
