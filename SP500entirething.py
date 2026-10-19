# -*- coding: utf-8 -*-
"""
=======================================================================================
Project Name: Fully Vectorized Real-Beta Quant Engine
File Name:    SP500entirething.py
Author:       Andy Dinh
Date Created: September 22, 2026
Version:      4.0.0
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


# Auto Installer
import importlib.util
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import lxml
except ImportError:
    print("  INSTALLING NECESSARY PACKAGES NOW...\n")
    if importlib.util.find_spec("pip") is None:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "ensurepip", "--upgrade"]
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Could not bootstrap pip for this Python interpreter. "
                "Install pip, then run the script again."
            ) from error
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "yfinance", "pandas", "numpy", "lxml",
        ])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Could not install the required packages. Check internet access and "
            "package installation permissions, then run the script again."
        ) from error
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import lxml


MONTE_CARLO_SAMPLES = 10_000
MONTE_CARLO_BATCH_SIZE = 1_024
TRADING_DAYS_PER_YEAR = 252
VOLATILITY_HALF_LIFE_DAYS = 63


# --- 2. BATCHED LINEAR ALGEBRA MONTE CARLO (CORNISH-FISHER EXPANSION) ---
# --- 2. BATCHED LINEAR ALGEBRA MONTE CARLO (HORNER-OPTIMIZED CORNISH-FISHER) ---
def _exponentially_weighted_moments(return_matrix, half_life_days):
    """Calculate recency-weighted moments, with observation weights halving by age."""
    values = np.asarray(return_matrix, dtype=float)
    ages = np.arange(values.shape[0] - 1, -1, -1, dtype=float)
    weights = np.exp2(-ages / half_life_days)
    weights /= weights.sum()

    means = np.sum(values * weights[:, np.newaxis], axis=0)
    centered = values - means
    weighted_centered = centered * weights[:, np.newaxis]
    covariance = weighted_centered.T @ centered
    variances = np.maximum(np.diag(covariance), 0)
    volatilities = np.sqrt(variances)

    third_moments = np.sum(weights[:, np.newaxis] * centered**3, axis=0)
    fourth_moments = np.sum(weights[:, np.newaxis] * centered**4, axis=0)
    skewness = np.divide(
        third_moments,
        volatilities**3,
        out=np.zeros_like(volatilities),
        where=volatilities > 0,
    )
    kurtosis = np.divide(
        fourth_moments,
        volatilities**4,
        out=np.zeros_like(volatilities),
        where=volatilities > 0,
    ) - 3
    return means, covariance, volatilities, skewness, kurtosis


def _cornish_fisher_warp(normal_draws, skewness, excess_kurtosis):
    # --- 3. HORNER'S METHOD WARP: Pure sequential additions and multiplications ---
    skew_squared = skewness * skewness
    cubic_coefficient = excess_kurtosis / 24 - skew_squared / 18
    quadratic_coefficient = skewness / 6
    linear_coefficient = 1 - excess_kurtosis / 8 + 5 * skew_squared / 36
    constant_coefficient = -skewness / 6
    return (
        ((cubic_coefficient * normal_draws + quadratic_coefficient) * normal_draws
         + linear_coefficient) * normal_draws + constant_coefficient
    )


def _simulate_cf_downside_risk(
    mean_returns,
    covariance,
    daily_volatility,
    skewness,
    excess_kurtosis,
    sample_count=MONTE_CARLO_SAMPLES,
    batch_size=MONTE_CARLO_BATCH_SIZE,
):
    """Estimate annualized CF downside risk using recency-weighted moments."""
    covariance_array = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_array)
    # Pairwise/sample covariances may be singular or have tiny negative round-off eigenvalues.
    eigenvalues = np.maximum(eigenvalues, 0)
    covariance_factor = eigenvectors * np.sqrt(eigenvalues)[np.newaxis, :]

    means = np.asarray(mean_returns, dtype=float)
    volatilities = np.asarray(daily_volatility, dtype=float)
    skews = np.asarray(skewness, dtype=float)
    kurtoses = np.asarray(excess_kurtosis, dtype=float)
    simulated_returns = np.empty((sample_count, means.size), dtype=float)
    rng = np.random.default_rng(0)

    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        standard_draws = rng.standard_normal((stop - start, means.size))
        correlated_draws = standard_draws @ covariance_factor.T
        standardized_draws = np.divide(
            correlated_draws,
            volatilities,
            out=np.zeros_like(correlated_draws),
            where=volatilities > 0,
        )
        warped_draws = _cornish_fisher_warp(
            standardized_draws,
            skews[np.newaxis, :],
            kurtoses[np.newaxis, :],
        )
        simulated_returns[start:stop] = means + warped_draws * volatilities

    daily_fifth_percentile = np.quantile(simulated_returns, 0.05, axis=0)
    return np.maximum(-daily_fifth_percentile, 0) * np.sqrt(TRADING_DAYS_PER_YEAR)


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
    ticker_universe = list(dict.fromkeys(ticker_list))
    download_list = list(dict.fromkeys(ticker_universe + [market_anchor]))

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
    if price_matrix.empty:
        print("[ERROR] Matrix retrieval failed.")
        return None
    matrix_returns = np.log(price_matrix / price_matrix.shift(1)).dropna()
    if len(matrix_returns) < 2:
        print("[ERROR] Not enough complete return observations for covariance-based simulation.")
        return None
    cov_matrix = matrix_returns.cov()  # Daily covariance matrix
    market_variance = cov_matrix.loc[market_anchor, market_anchor]
    vector_beta = cov_matrix[market_anchor] / market_variance

    # Compute Metrics
    ann_returns = matrix_returns.mean() * TRADING_DAYS_PER_YEAR * 100
    (
        weighted_mean_returns,
        weighted_covariance,
        weighted_daily_volatility,
        weighted_skewness,
        weighted_excess_kurtosis,
    ) = _exponentially_weighted_moments(
        matrix_returns,
        half_life_days=VOLATILITY_HALF_LIFE_DAYS,
    )
    weighted_mean_returns = pd.Series(weighted_mean_returns, index=matrix_returns.columns)
    weighted_daily_volatility = pd.Series(
        weighted_daily_volatility,
        index=matrix_returns.columns,
    )
    weighted_skewness = pd.Series(weighted_skewness, index=matrix_returns.columns)
    weighted_excess_kurtosis = pd.Series(
        weighted_excess_kurtosis,
        index=matrix_returns.columns,
    )
    daily_volatility = weighted_daily_volatility
    ann_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    skewness = weighted_skewness
    excess_kurtosis = weighted_excess_kurtosis
    cf_downside_risk = _simulate_cf_downside_risk(
        mean_returns=weighted_mean_returns,
        covariance=weighted_covariance,
        daily_volatility=daily_volatility,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
    vol_penalty = cf_downside_risk * 20

    # Skew, Beta and Final Score Work
    skew_penalty = np.sign(skewness) * (2 ** np.abs(skewness))
    beta_penalty = np.where(vector_beta > 1.2, -((vector_beta - 1.2) / 0.2) * 2.0, 
                   np.where(vector_beta < 0.8, (0.8 - vector_beta) / 0.1, 0.0))
    scores = ann_returns - vol_penalty + skew_penalty + beta_penalty
    
    # Build DataFrame for Reporting
    report_df = pd.DataFrame({
        "Ann_Return_%": ann_returns,
        "Volatility_%": ann_volatility * 100,
        "CF_Downside_Risk_%": cf_downside_risk * 100,
        "Skewness": skewness,
        "Excess_Kurtosis": excess_kurtosis,
        "Beta": vector_beta,
        "Custom_Score": scores
    })
    report_df.index.name = "Ticker"
    report_df = report_df.reindex(ticker_universe).reset_index()

    # Making Report
    ranked_report = report_df.sort_values(by="Custom_Score", ascending=False)
    numeric_columns = ranked_report.select_dtypes(include="number").columns
    ranked_report[numeric_columns] = ranked_report[numeric_columns].round(2)
    output_report = ranked_report.fillna("N/A")

    # Displaying all scanned assets
    print("\n" + "\"="*95)
    print(f"S&P 500 universe size: {len(ticker_universe)}")
    print(f"Stocks displayed: {len(ranked_report)}")
    print(f"Displaying all {len(ranked_report)} scanned stocks (missing data shown as N/A):")
    display_batch_size = 50
    for start in range(0, len(output_report), display_batch_size):
        stop = min(start + display_batch_size, len(output_report))
        print(f"\nStocks {start + 1}-{stop} of {len(output_report)}:")
        print(output_report.iloc[start:stop].to_string(
            index=False,
            max_rows=None,
            float_format=lambda value: f"{value:.2f}",
        ))
    print("="*95)

    # Exporting to CSV
    output_report.to_csv(
        "SP500_Screened_Rankings_RealBeta.csv",
        index=False,
        float_format="%.2f",
    )
    print("\n[SUCCESS] Report exported to 'SP500_Screened_Rankings_RealBeta.csv'")
    return ranked_report

    # Main execution block
if __name__ == "__main__":
    sp500_catalog = get_sp500_tickers()
    vectorized_quant_screener(sp500_catalog, period = "1y")