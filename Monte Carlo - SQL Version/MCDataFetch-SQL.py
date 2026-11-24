# -*- coding: utf-8 -*-
"""
=======================================================================================
Project Name: Portfolio Risk Simulator - Data Ingestion Engine
File Name:    MCDataFetch-SQL.py
Author:       Andy Dinh
License & Copyright:
    Copyright (c) 2026 Andy Dinh. All Rights Reserved.
    PROPRIETARY AND CONFIDENTIAL. Strictly NO permission is granted for unauthorized 
    distribution, duplication, modification, or public cloning of this source logic.

Description:
    Pure data ingestion pipeline. Fetches historical price and volume data from
    Yahoo Finance and synchronizes it into a local SQLite database (market_data.db).
    
    MAKE SURE you have market_db.py in the same directory. This script is a prerequisite for running
    MC(1.1.0).py, which performs Monte Carlo simulations on the cached market data.
    
    Feeds directly into MC(1.1.0).py for instantaneous Monte Carlo simulations.

    Note: Calibrated for highly liquid US equities and sector ETFs! 
    Running it on illiquid or exotic assets may produce unreliable results.
=======================================================================================
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import pandas as pd
except ImportError:
    import importlib.util
    import subprocess
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
            "pandas", 
        ])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Could not install the required packages. Check internet access and "
            "package installation permissions, then run the script again."
        ) from error
    import pandas as pd
from market_db import (
    sync_market_data,
    get_tickers_status,
    get_db_path,
    DEFAULT_FACTORS,
    DEFAULT_MACRO,
)



def fetch_and_store_data(
    tickers,
    include_factors=True,
    lookback_years=5,
    force=False,
):
    """
    Synchronizes ticker data into the SQLite database with zero calculations.
    - If a ticker is already cached, only fetches missing days since last update.
    - If a ticker is new, fetches the full baseline (default: 5 years).
    """
    db_file = get_db_path()
    print("=" * 80)
    print("           MONTE CARLO SQL DATA INGESTION ENGINE")
    print("=" * 80)
    print(f"Database Target: {db_file}")
    print(f"Portfolio Tickers: {len(tickers)} assets")
    if include_factors:
        print(f"Factor ETFs:       {', '.join(DEFAULT_FACTORS)}")
        print(f"Macro Anchors:     {', '.join(DEFAULT_MACRO)}")
    print("-" * 80)

    # Run the incremental sync
    synced_universe = sync_market_data(
        tickers=tickers,
        include_factors=include_factors,
        default_lookback_years=lookback_years,
        force=force,
    )

    # Print Database Summary Status
    status = get_tickers_status(synced_universe)
    summary_rows = []
    for t in synced_universe:
        info = status.get(t, {})
        summary_rows.append({
            "Ticker": t,
            "Days Stored": info.get("count", 0),
            "Earliest Date": info.get("min_date", "N/A"),
            "Latest Date": info.get("max_date", "N/A"),
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "-" * 80)
    print("DATABASE STATUS SUMMARY (SAMPLE / ALL):")
    print("-" * 80)
    if len(summary_df) <= 25:
        print(summary_df.to_string(index=False))
    else:
        # If full S&P 500, display first 10, last 5, and factor summary
        print(summary_df.head(10).to_string(index=False))
        print("...")
        print(summary_df.tail(5).to_string(index=False))

    print("-" * 80)
    print(f"[SUCCESS] {len(synced_universe)} assets are fully cached in '{os.path.basename(db_file)}'.")
    print(
        "[READY] Run 'MC(1.1.0 - SQL).py'; it uses this cache and backfills "
        "older history if its panic-observation target is not met."
    )
    print("=" * 80 + "\n")
    return summary_df


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration: Choose which tickers to fetch and cache into SQLite
    # -------------------------------------------------------------------------

    # 1. Your portfolio stocks (matches MC(1.1.0 - SQL).py):
    stock_tickers = [
            "BRK-B", "CI", "EBAY", "VRSN", "XOM", 
    "PPL", "WMT", "MA", "COP", "PM", "UPS",  
]

    # Run ingestion (pure data fetch & store - no math/calculations)
    # Seeds the standard 5-year baseline; MC(1.1.0 - SQL).py backfills older
    # history when its usable panic-observation target is not met.
    fetch_and_store_data(
        tickers=stock_tickers,
        include_factors=True,
        lookback_years=5,
    )
