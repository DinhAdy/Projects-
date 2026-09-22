# -*- coding: utf-8 -*-
"""
=======================================================================================
Project Name: Fully Vectorized Real-Beta Quant Engine
File Name:    SP500Entirething.py
Author:       Andy Dinh
Date Created: September 22, 2026
Version:      1.0.0
Environment:  Python 3.13 or below (Stable Production Baseline)
Note:         This script is designed to be run in a local Python environment. It may 
              not function properly in online interpreters due to package dependencies 
              and data retrieval from external sources. This script also needs internet access 
              to fetch data from Yahoo Finance and Wikipedia.

Description:
    A loop-free asset screening framework. Looks up the list of S&P 500 tickers from Wikipedia, 
    downloads their historical price and volume data from Yahoo Finance, and computes a custom 
    score for each asset based on annualized returns, volatility, skewness, beta relative to 
    the S&P 500 index, and liquidity. The results are ranked and exported to a CSV file.
=======================================================================================
"""


#Auto Installer
try:
    import os
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import lxml
except ImportError:
    import subprocess
    import sys
    print("\n  INSTALLING NECESSARY STUFF NOW...\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "pandas", "numpy", "lxml"])
    import yfinance as yf
    import pandas as pd
    import numpy as np

def get_sp500_tickers():

    # Importing necessary libraries for web search
    from io import BytesIO
    from urllib.request import Request, urlopen

    # Making a request to Wikipedia(Don't use it as a source!)
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response:
        table = pd.read_html(BytesIO(response.read()))[0]

    #Returning list of tickers
    return (
        table["Symbol"]
        .astype(str)
        .str.replace(".", "-", regex=False)
        .drop_duplicates()
        .tolist()
    )


def vectorized_quant_screener(ticker_list, period):

    # Beta Calculation requires the S&P 500 index as a market anchor
    market_anchor = "^GSPC"
    download_list = list(set(ticker_list + [market_anchor]))

    # Downloading from Yahoo Finance
    print(f"\n[INFO] Pulling full matrix for {len(download_list)} assets via Yahoo Finance...")
    raw_data = yf.download(
        download_list,
        period = period,
        progress = True,
        group_by = 'column',
        threads = min(8, max(1, ((os.cpu_count() or 2) // 2))),
    )

    # All the Fancy Math!
    price_matrix = raw_data['Close']
    volume_matrix = raw_data['Volume']
    if price_matrix.empty:
        print("[ERROR] Matrix retrieval failed.")
        return None
    matrix_returns = np.log(price_matrix / price_matrix.shift(1)).dropna()
    cov_matrix = matrix_returns.cov()  # Daily covariance matrix
    market_variance = cov_matrix.loc[market_anchor, market_anchor]
    vector_beta = cov_matrix[market_anchor] / market_variance

    # Compute Metrics
    ann_returns = matrix_returns.mean() * 252 * 100
    ann_volatility = matrix_returns.std() * np.sqrt(252)
    skewness = matrix_returns.skew().fillna(0)
    avg_price = price_matrix.mean()
    avg_volume = volume_matrix.mean()
    dollar_liquidity = avg_price * avg_volume

    #Liquidity and Volume work
    volume_penalty = np.where(
        dollar_liquidity >= 20_000_000,
        0.0,
        np.where(
            dollar_liquidity < 5_000_000,
            (1.2 ** ((5_000_000 - dollar_liquidity) / 250_000)) - 1,
            0.5,
        ),
    )
    vol_penalty = ann_volatility * 20

    # Skew, Beta and Final Score Work
    skew_penalty = np.sign(skewness) * (2.5 ** np.abs(skewness))
    beta_penalty = np.where(vector_beta > 1.2, -((vector_beta - 1.2) / 0.2) * 2.0, 
                   np.where(vector_beta < 0.8, (0.8 - vector_beta) / 0.1, 0.0))
    scores = ann_returns - vol_penalty + skew_penalty + beta_penalty - volume_penalty
    
    # Build DataFrame for Reporting
    report_df = pd.DataFrame({
        "Ticker": price_matrix.columns,
        "Avg_Price($)": avg_price,
        "Daily_LQDTY($)": dollar_liquidity,
        "Ann_Return_%": ann_returns,
        "Volatility_%": ann_volatility * 100,
        "Skewness": skewness,
        "Beta": vector_beta,
        "Custom_Score": scores
    }).reset_index(drop=True)
    
    # Making Report
    report_df = report_df[report_df["Ticker"] != market_anchor].dropna(subset=["Custom_Score"])
    ranked_report = report_df.sort_values(by="Custom_Score", ascending=False)
    numeric_columns = ranked_report.select_dtypes(include="number").columns
    ranked_report[numeric_columns] = ranked_report[numeric_columns].round(2)

    # Displaying the top 100 ranked assets
    print("\n" + "\"="*95)
    print(ranked_report.to_string(
    index=False,
    float_format=lambda value: f"{value:.2f}",
    formatters={"Daily_LQDTY($)": lambda value: f"{value:.2e}"}, ))
    print("="*95)

    # Exporting to CSV
    csv_report = ranked_report.copy()
    csv_report["Daily_LQDTY($)"] = csv_report["Daily_LQDTY($)"].map(lambda value: f"{value:.2e}")
    csv_report.to_csv("SP500_Screened_Rankings_RealBeta.csv", index=False, float_format="%.2f")
    print("\n[SUCCESS] Custom report exported to 'SP500_Screened_Rankings_RealBeta.csv'")
    return ranked_report

    # Main execution block
if __name__ == "__main__":
    sp500_catalog = get_sp500_tickers()
    while(True):
        period = input("Enter period (1/5d, 1/3/6mo, 1/2/5/10y, ytd, max): ").lower().strip()
        if(period in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]):
            break
        else:
            print("[ERROR] Invalid period. Please enter a valid period (1/5d, 1/3/6mo, 1/2/5/10y, ytd, max).")
    vectorized_quant_screener(sp500_catalog, period=period)
