"""
=======================================================================================
Project Name: Portfolio Risk Simulator 
File Name:    MC.py
Author:       Andy Dinh
Date Created: October 7, 2026
Version:      1.1.0
Environment:  Python 3.13 or below (Stable Production Baseline)

Note: Calibrated for highly liquid US equities and sector ETFs! 
Running it on illiquid or exotic assets may produce unreliable results.

MAKE SURE you have MCDataFetch-SQL.py in the same directory and run it first to fetch the required market data.

License & Copyright:
    Copyright (c) 2026 Andy Dinh. All Rights Reserved.
    PROPRIETARY AND CONFIDENTIAL. Strictly NO permission is granted for unauthorized 
    distribution, duplication, modification, or public cloning of this source logic.
=======================================================================================
"""



import datetime
import time
import os
import sys
from io import StringIO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import numpy as np
    from arch import arch_model
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
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
            "yfinance", "numpy", "arch", "scipy", "scikit-learn",
        ])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Could not install the required packages. Check internet access and "
            "package installation permissions, then run the script again."
        ) from error
    import numpy as np
    from arch import arch_model
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
from market_db import backfill_market_data, load_market_data, sync_market_data

def build_midas_scale(daily_returns, vix):
    monthly_vix = vix.resample("ME").mean().dropna()
    log_vix_variance = 2 * np.log(monthly_vix.to_numpy(dtype=np.float64))
    midas_lags = min(24, len(log_vix_variance) - 1)
    lag_positions = np.arange(1, midas_lags + 1, dtype=np.float64) / (
        midas_lags + 1
    )
    midas_weights = (1 - lag_positions) ** 4
    midas_weights /= midas_weights.sum()
    long_run_macro = log_vix_variance.mean()
    midas_signal = np.empty(len(monthly_vix), dtype=np.float64)

    for month_index in range(len(monthly_vix)):
        available_lags = min(month_index, midas_lags)
        if available_lags:
            lagged_values = log_vix_variance[
                month_index - available_lags:month_index
            ][::-1]
            midas_signal[month_index] = np.average(
                lagged_values,
                weights=midas_weights[:available_lags],
            )
        else:
            midas_signal[month_index] = long_run_macro

    macro_by_month = dict(zip(monthly_vix.index.to_period("M"), midas_signal))
    monthly_variance = daily_returns.groupby(
        daily_returns.index.to_period("M")
    ).agg(lambda values: np.mean(np.square(values.to_numpy())))
    monthly_macro = np.array(
        [
            macro_by_month.get(month, long_run_macro)
            for month in monthly_variance.index
        ],
        dtype=np.float64,
    )
    monthly_variance_values = monthly_variance.to_numpy(dtype=np.float64)
    valid_months = (
        (monthly_variance_values > 0).all(axis=1)
        & np.isfinite(monthly_macro)
    )
    regression_x = monthly_macro[valid_months] - long_run_macro
    regression_design = np.column_stack(
        (np.ones(regression_x.size), regression_x)
    )
    midas_parameters = np.linalg.lstsq(
        regression_design,
        np.log(monthly_variance_values[valid_months]),
        rcond=None,
    )[0]
    midas_sensitivity = midas_parameters[1]
    daily_macro = np.array(
        [
            macro_by_month.get(month, long_run_macro)
            for month in daily_returns.index.to_period("M")
        ],
        dtype=np.float64,
    )
    daily_midas_scale = np.exp(
        np.clip(
            (daily_macro - long_run_macro)[:, np.newaxis] * midas_sensitivity,
            -3,
            3,
        )
    )
    daily_midas_scale /= daily_midas_scale.mean(axis=0)
    return daily_midas_scale, midas_sensitivity

def fit_asset_garch_models(returns, daily_midas_scale):
    n_days, n_assets = returns.shape
    standardized_residuals = np.empty((n_days, n_assets), dtype=np.float64)
    forecast_variances = np.empty(n_assets, dtype=np.float64)
    student_degrees = np.empty(n_assets, dtype=np.float64)

    for asset_index in range(n_assets):
        scaled_returns = (
            returns[:, asset_index] * 100
            / np.sqrt(daily_midas_scale[:, asset_index])
        )
        fitted_model = arch_model(
            scaled_returns,
            mean="Constant",
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            dist="StudentsT",
            rescale=False,
        ).fit(disp="off", show_warning=False)
        model_forecast = fitted_model.forecast(horizon=1, reindex=False)
        current_scale = daily_midas_scale[-1, asset_index]
        standardized_residuals[:, asset_index] = np.nan_to_num(
            np.asarray(fitted_model.std_resid),
            nan=0.0,
        )
        forecast_variances[asset_index] = (
            model_forecast.variance.values[-1, 0] * current_scale / 10_000
        )
        student_degrees[asset_index] = fitted_model.params["nu"]

    return standardized_residuals, forecast_variances, student_degrees

def minimum_variance_weights(covariance, initial_weights=None):
    scale = 1_000_000.0
    result = minimize(
        lambda candidate: scale * (candidate @ covariance @ candidate),
        np.full(n_assets, 1 / n_assets) if initial_weights is None else initial_weights,
        jac=lambda candidate: 2 * scale * (covariance @ candidate),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda candidate: candidate.sum() - 1},
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    return result.x

def bootstrap_minimum_variance_weights(
    standardized_residuals,
    forecast_volatility,
    random_generator,
    n_bootstraps=20,
):
    weight_samples = []
    for _ in range(n_bootstraps):
        sample_indices = random_generator.integers(
            0,
            len(standardized_residuals),
            size=len(standardized_residuals),
        )
        sampled_residuals = standardized_residuals[sample_indices]
        sample_shrunk_covariance = LedoitWolf().fit(
            sampled_residuals
        ).covariance_
        sample_diag_inv = 1.0 / np.sqrt(
            np.maximum(np.diag(sample_shrunk_covariance), 1e-12)
        )
        sample_correlation = (
            sample_shrunk_covariance
            * np.outer(sample_diag_inv, sample_diag_inv)
        )
        sample_covariance = sample_correlation * np.outer(
            forecast_volatility,
            forecast_volatility,
        )
        weight_samples.append(minimum_variance_weights(sample_covariance))

    optimized_weights = np.mean(weight_samples, axis=0)
    return optimized_weights / optimized_weights.sum()

def format_table(rows, formats, delimiter=" "):
    buffer = StringIO()
    np.savetxt(buffer, rows, fmt=formats, delimiter=delimiter)
    return buffer.getvalue().rstrip()

def annualized_risk_ratios(returns, target_return, periods_per_year):
    mean_excess_return = np.mean(returns - target_return)
    volatility = np.std(returns)
    downside_deviation = np.sqrt(
        np.mean(np.minimum(returns - target_return, 0) ** 2)
    )
    sharpe = (
        mean_excess_return / volatility * np.sqrt(periods_per_year)
        if volatility > 0
        else 0.0
    )
    sortino = (
        mean_excess_return / downside_deviation * np.sqrt(periods_per_year)
        if downside_deviation > 0
        else 0.0
    )
    return sharpe, sortino

def build_factor_return_matrix(factor_asset_returns):
    return np.column_stack(
        (
            factor_asset_returns["SPY"].to_numpy(dtype=np.float64),
            (
                factor_asset_returns["IWM"] - factor_asset_returns["SPY"]
            ).to_numpy(dtype=np.float64),
            (
                factor_asset_returns["IWD"] - factor_asset_returns["IWF"]
            ).to_numpy(dtype=np.float64),
            factor_asset_returns["TLT"].to_numpy(dtype=np.float64),
            factor_asset_returns["GLD"].to_numpy(dtype=np.float64),
            (
                factor_asset_returns["XLV"] - factor_asset_returns["SPY"]
            ).to_numpy(dtype=np.float64),
            (
                factor_asset_returns["XLE"] - factor_asset_returns["SPY"]
            ).to_numpy(dtype=np.float64),
        )
    )

def classify_historical_regimes(daily_returns, vix):
    aligned_vix = vix.reindex(daily_returns.index).ffill()
    lagged_vix = aligned_vix.shift(1).to_numpy(dtype=np.float64)
    lagged_vix_velocity = (
        aligned_vix.pct_change(5).shift(1).to_numpy(dtype=np.float64)
    )
    valid_vix = np.isfinite(lagged_vix) & np.isfinite(lagged_vix_velocity)
    if (
        not valid_vix.any()
        or not np.isfinite(lagged_vix[-1] + lagged_vix_velocity[-1])
    ):
        raise ValueError(
            "Insufficient aligned VIX history to classify simulation regimes."
        )
    vix_level_threshold = np.quantile(lagged_vix[valid_vix], 0.75)
    vix_panic_level_threshold = np.quantile(lagged_vix[valid_vix], 0.90)
    positive_vix_velocity = lagged_vix_velocity[
        valid_vix & (lagged_vix_velocity > 0)
    ]
    if positive_vix_velocity.size == 0:
        raise ValueError(
            "Insufficient positive VIX momentum to calibrate the panic regime."
        )
    vix_panic_velocity_threshold = np.quantile(positive_vix_velocity, 0.75)
    historical_regimes = np.zeros(len(daily_returns), dtype=np.int8)
    historical_regimes[valid_vix & (lagged_vix >= vix_level_threshold)] = 1
    historical_regimes[
        valid_vix
        & (lagged_vix >= vix_panic_level_threshold)
        & (lagged_vix_velocity >= vix_panic_velocity_threshold)
    ] = 2
    return aligned_vix, valid_vix, historical_regimes

def count_usable_panic_observations(daily_returns, factor_returns, vix):
    _, valid_vix, historical_regimes = classify_historical_regimes(
        daily_returns,
        vix,
    )
    usable_factor_rows = valid_vix & np.isfinite(factor_returns).all(axis=1)
    return int(np.count_nonzero(usable_factor_rows & (historical_regimes == 2)))

def prepare_market_data(tickers, factor_tickers):
    if panic_history_expand:
        if panic_observation_target < 1:
            raise ValueError("panic_observation_target must be at least 1.")
        if panic_history_step_years < 1:
            raise ValueError("panic_history_step_years must be at least 1.")
        if panic_history_max_years < 1:
            raise ValueError("panic_history_max_years must be at least 1.")
    all_needed = list(dict.fromkeys(tickers + factor_tickers + ["^VIX", "^IRX"]))
    maximum_start = (
        datetime.datetime.combine(
            datetime.date.today()
            - datetime.timedelta(days=round(panic_history_max_years * 365.25)),
            datetime.time.min,
        )
        if panic_history_expand
        else None
    )
    minimum_history_date = (
        maximum_start.strftime("%Y-%m-%d")
        if maximum_start is not None
        else None
    )
    print(
        f"\n[SQL] Loading market data for {len(all_needed)} assets "
        "from local SQLite database...",
        end="\n",
    )

    close, volume = load_market_data(
        all_needed,
        min_date=minimum_history_date,
    )
    missing = [t for t in all_needed if t not in close.columns or close[t].dropna().count() < 252]
    if missing:
        print(f"[SQL CACHE] Missing or incomplete data in SQLite for {len(missing)} ticker(s): {missing}")
        print("Synchronizing required baseline history to SQLite from Yahoo Finance...")
        sync_market_data(all_needed, include_factors=True, default_lookback_years=5)
        close, volume = load_market_data(
            all_needed,
            min_date=minimum_history_date,
        )

    daily_returns = close[tickers].dropna().pct_change().dropna()
    factor_asset_returns = close[factor_tickers].pct_change().reindex(
        daily_returns.index
    )
    factor_returns = build_factor_return_matrix(factor_asset_returns)
    vix = close["^VIX"].dropna()

    if panic_history_expand:
        panic_observations = count_usable_panic_observations(
            daily_returns,
            factor_returns,
            vix,
        )
        while (
            panic_observations < panic_observation_target
            and daily_returns.index.min() > maximum_start
        ):
            previous_start = daily_returns.index.min()
            requested_start = max(
                previous_start
                - datetime.timedelta(days=round(panic_history_step_years * 365.25)),
                maximum_start,
            )
            if requested_start >= previous_start:
                break
            saved_rows = backfill_market_data(
                all_needed,
                requested_start.strftime("%Y-%m-%d"),
                previous_start.strftime("%Y-%m-%d"),
            )
            close, volume = load_market_data(
                all_needed,
                min_date=minimum_history_date,
            )
            daily_returns = close[tickers].dropna().pct_change().dropna()
            factor_asset_returns = close[factor_tickers].pct_change().reindex(
                daily_returns.index
            )
            factor_returns = build_factor_return_matrix(factor_asset_returns)
            vix = close["^VIX"].dropna()
            panic_observations = count_usable_panic_observations(
                daily_returns,
                factor_returns,
                vix,
            )
            if daily_returns.index.min() >= previous_start or saved_rows == 0:
                print(
                    "[SQL BACKFILL] Stopping: no earlier aligned portfolio "
                    "history was added."
                )
                break

        panic_target_reached = panic_observations >= panic_observation_target
        history_start = daily_returns.index.min()
        history_end = daily_returns.index.max()
        market_history_years = (
            (history_end - history_start).days / 365.25
            if len(daily_returns) > 1
            else 0.0
        )
        print(
            f"[PANIC HISTORY] {panic_observations} usable real panic observations; "
            f"target {panic_observation_target} "
            f"{'reached' if panic_target_reached else 'not reached'} "
            f"within {market_history_years:.1f} years."
        )
    else:
        panic_observations = count_usable_panic_observations(
            daily_returns,
            factor_returns,
            vix,
        )
        panic_target_reached = panic_observations >= panic_observation_target
        market_history_years = (
            (daily_returns.index[-1] - daily_returns.index[0]).days / 365.25
            if len(daily_returns) > 1
            else 0.0
        )

    latest_date = close.index.max().strftime("%Y-%m-%d") if len(close) else "Unknown"
    print(f"[SQL CACHE] Loaded from 'market_data.db' (latest trading day: {latest_date}).\n")

    irx = close["^IRX"].dropna() if "^IRX" in close else None
    live_risk_free_rate = (
        float(irx.iloc[-1] / 100.0)
        if irx is not None and len(irx) > 0 and np.isfinite(irx.iloc[-1])
        else 0.04
    )
    returns = daily_returns.to_numpy(dtype=np.float64)
    n_days, n_assets = returns.shape
    aligned_close = close[tickers].reindex(daily_returns.index)
    aligned_volume = volume[tickers].reindex(daily_returns.index)
    dollar_volume = aligned_close * aligned_volume
    lagged_average_dollar_volume = (
        dollar_volume.rolling(21, min_periods=21)
        .mean()
        .shift(1)
        .to_numpy(dtype=np.float64)
    )
    lagged_return_volatility = (
        daily_returns.rolling(21, min_periods=21)
        .std()
        .shift(1)
        .to_numpy(dtype=np.float64)
    )
    return {
        "close": close,
        "volume": volume,
        "daily_returns": daily_returns,
        "vix": vix,
        "live_risk_free_rate": live_risk_free_rate,
        "returns": returns,
        "n_days": n_days,
        "n_assets": n_assets,
        "factor_returns": factor_returns,
        "panic_observations": panic_observations,
        "panic_target_reached": panic_target_reached,
        "market_history_years": market_history_years,
        "dollar_volume": dollar_volume,
        "lagged_average_dollar_volume": lagged_average_dollar_volume,
        "lagged_return_volatility": lagged_return_volatility,
    }

def fit_portfolio_dynamics(
    standardized_residuals,
    forecast_volatility,
    returns,
    weights,
):
    lw_cov = LedoitWolf().fit(standardized_residuals).covariance_
    lw_diag_inv = 1.0 / np.sqrt(np.maximum(np.diag(lw_cov), 1e-12))
    cond_correlation = lw_cov * np.outer(lw_diag_inv, lw_diag_inv)
    conditional_covariance = cond_correlation * np.outer(
        forecast_volatility,
        forecast_volatility,
    )
    portfolio_cond_variance = float(weights @ conditional_covariance @ weights)
    portfolio_cond_volatility = np.sqrt(portfolio_cond_variance)
    historical_portfolio_returns = returns @ weights
    portfolio_garch_fit = arch_model(
        historical_portfolio_returns * 100,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="StudentsT",
        rescale=False,
    ).fit(disp="off", show_warning=False)
    portfolio_garch_parameters = portfolio_garch_fit.params
    portfolio_garch_forecast = portfolio_garch_fit.forecast(
        horizon=1,
        reindex=False,
    )
    portfolio_garch_base_variance = float(
        portfolio_garch_forecast.variance.values[-1, 0]
    )
    portfolio_garch_omega = float(portfolio_garch_parameters["omega"])
    portfolio_garch_alpha = float(portfolio_garch_parameters["alpha[1]"])
    portfolio_garch_gamma = float(portfolio_garch_parameters["gamma[1]"])
    portfolio_garch_beta = float(portfolio_garch_parameters["beta[1]"])
    if (
        not np.isfinite(
            [
                portfolio_garch_base_variance,
                portfolio_garch_omega,
                portfolio_garch_alpha,
                portfolio_garch_gamma,
                portfolio_garch_beta,
            ]
        ).all()
        or portfolio_garch_base_variance <= 0
        or portfolio_garch_omega < 0
        or portfolio_garch_alpha < 0
        or portfolio_garch_gamma < 0
        or portfolio_garch_beta < 0
    ):
        raise ValueError(
            "Portfolio GJR-GARCH fit returned invalid volatility parameters."
        )
    unconditional_portfolio_vol = float(np.std(historical_portfolio_returns))
    fhs_vol_scale = float(
        np.clip(
            portfolio_cond_volatility / max(unconditional_portfolio_vol, 1e-6),
            0.6,
            2.5,
        )
    )
    ewma_decay = 0.94
    ewma_weights = ewma_decay ** np.arange(
        len(historical_portfolio_returns) - 1,
        -1,
        -1,
        dtype=np.float64,
    )
    ewma_weights *= 1 - ewma_decay
    ewma_weights /= ewma_weights.sum()
    historical_mean = ewma_weights @ historical_portfolio_returns
    centered_returns = historical_portfolio_returns - historical_mean
    historical_variance = ewma_weights @ (centered_returns ** 2)
    dynamic_skewness = (
        ewma_weights @ (centered_returns ** 3)
    ) / historical_variance ** 1.5
    dynamic_kurtosis = (
        ewma_weights @ (centered_returns ** 4)
    ) / historical_variance ** 2 - 3
    return {
        "historical_portfolio_returns": historical_portfolio_returns,
        "portfolio_cond_variance": portfolio_cond_variance,
        "portfolio_cond_volatility": portfolio_cond_volatility,
        "portfolio_garch_base_variance": portfolio_garch_base_variance,
        "portfolio_garch_omega": portfolio_garch_omega,
        "portfolio_garch_alpha": portfolio_garch_alpha,
        "portfolio_garch_gamma": portfolio_garch_gamma,
        "portfolio_garch_beta": portfolio_garch_beta,
        "fhs_vol_scale": fhs_vol_scale,
        "dynamic_skewness": dynamic_skewness,
        "dynamic_kurtosis": dynamic_kurtosis,
    }

def fit_regime_factor_model(
    vix,
    daily_returns,
    factor_returns,
    historical_portfolio_returns,
):
    n_days = len(daily_returns)
    aligned_vix, valid_vix, historical_regimes = classify_historical_regimes(
        daily_returns,
        vix,
    )
    transition_counts = np.ones((3, 3), dtype=np.float64)
    valid_transition = valid_vix[:-1] & valid_vix[1:]
    np.add.at(
        transition_counts,
        (
            historical_regimes[:-1][valid_transition],
            historical_regimes[1:][valid_transition],
        ),
        1,
    )
    regime_transition = transition_counts / transition_counts.sum(
        axis=1,
        keepdims=True,
    )
    current_regime = historical_regimes[-1]
    current_regime_name = ("Calm", "High-volatility", "Panic")[current_regime]
    regime_transition_cumulative = np.cumsum(
        regime_transition,
        axis=1,
    ).astype(np.float32)
    valid_factor_rows = valid_vix & np.isfinite(factor_returns).all(axis=1)
    excluded_factor_rows = int((valid_vix & ~valid_factor_rows).sum())
    factor_design = np.column_stack(
        (np.ones(valid_factor_rows.sum()), factor_returns[valid_factor_rows])
    )
    factor_coefficients = np.linalg.lstsq(
        factor_design,
        historical_portfolio_returns[valid_factor_rows],
        rcond=None,
    )[0]
    regime_sample_masks = [
        valid_factor_rows & (historical_regimes == regime)
        for regime in range(3)
    ]
    regime_names = ("Calm", "High-volatility", "Panic")
    regime_shrinkage_prior = 100
    regime_factor_coefficients = np.empty(
        (3, factor_design.shape[1]),
        dtype=np.float64,
    )
    regime_factors = []
    regime_residuals = []
    regime_raw_counts = np.empty(3, dtype=np.int32)
    for regime, mask in enumerate(regime_sample_masks):
        regime_design = np.column_stack(
            (np.ones(mask.sum()), factor_returns[mask])
        )
        regime_returns_sample = historical_portfolio_returns[mask]
        regime_raw_counts[regime] = regime_returns_sample.size
        if regime_returns_sample.size > factor_design.shape[1]:
            regime_fit = np.linalg.lstsq(
                regime_design,
                regime_returns_sample,
                rcond=None,
            )[0]
            shrinkage = regime_returns_sample.size / (
                regime_returns_sample.size + regime_shrinkage_prior
            )
            regime_factor_coefficients[regime] = (
                factor_coefficients
                + shrinkage * (regime_fit - factor_coefficients)
            )
        else:
            regime_factor_coefficients[regime] = factor_coefficients
        regime_residuals.append(
            (
                regime_returns_sample
                - regime_design @ regime_factor_coefficients[regime]
            ).astype(np.float32)
        )
        regime_factors.append(factor_returns[mask].astype(np.float32))

    return {
        "aligned_vix": aligned_vix,
        "valid_vix": valid_vix,
        "historical_regimes": historical_regimes,
        "current_regime": current_regime,
        "current_regime_name": current_regime_name,
        "regime_transition": regime_transition,
        "regime_transition_cumulative": regime_transition_cumulative,
        "valid_factor_rows": valid_factor_rows,
        "excluded_factor_rows": excluded_factor_rows,
        "regime_sample_masks": regime_sample_masks,
        "regime_names": regime_names,
        "regime_factor_coefficients": regime_factor_coefficients,
        "regime_factors": regime_factors,
        "regime_residuals": regime_residuals,
        "regime_raw_counts": regime_raw_counts,
    }

def build_stress_and_panic_data(
    standardized_residuals,
    forecast_volatility,
    weights,
    current_regime,
    regime_sample_masks,
    regime_factors,
    regime_residuals,
    factor_returns,
    student_df,
    panic_observation_target,
    random_generator,
):
    active_regime_residuals = standardized_residuals[
        regime_sample_masks[current_regime]
    ]
    lw_active_cov = LedoitWolf().fit(active_regime_residuals).covariance_
    lw_active_diag_inv = 1.0 / np.sqrt(
        np.maximum(np.diag(lw_active_cov), 1e-12)
    )
    regime_correlation = lw_active_cov * np.outer(
        lw_active_diag_inv,
        lw_active_diag_inv,
    )
    conditional_covariance = regime_correlation * np.outer(
        forecast_volatility,
        forecast_volatility,
    )
    portfolio_cond_variance = float(weights @ conditional_covariance @ weights)
    panic_asset_residuals = standardized_residuals[regime_sample_masks[2]]
    if panic_asset_residuals.shape[0] < 2:
        raise ValueError(
            "At least two panic-regime observations are needed for stress covariance."
        )
    lw_panic_cov = LedoitWolf().fit(panic_asset_residuals).covariance_
    lw_panic_diag_inv = 1.0 / np.sqrt(
        np.maximum(np.diag(lw_panic_cov), 1e-12)
    )
    panic_correlation = lw_panic_cov * np.outer(
        lw_panic_diag_inv,
        lw_panic_diag_inv,
    )
    panic_conditional_covariance = panic_correlation * np.outer(
        forecast_volatility,
        forecast_volatility,
    )
    panic_portfolio_cov = panic_conditional_covariance @ weights
    panic_portfolio_variance = float(weights @ panic_portfolio_cov)
    panic_asset_betas = panic_portfolio_cov / max(
        panic_portfolio_variance,
        1e-12,
    )

    panic_factors = regime_factors[2]
    panic_resids = regime_residuals[2]
    n_panic_synth = max(0, panic_observation_target - len(panic_resids))
    if n_panic_synth:
        panic_sub_idx = random_generator.integers(
            0,
            len(panic_resids),
            size=n_panic_synth,
        )
        t_tail_shocks = (
            random_generator.standard_t(df=student_df, size=n_panic_synth)
            * np.sqrt((student_df - 2) / student_df)
        ).astype(np.float32)
        synth_panic_resids = panic_resids[panic_sub_idx] + (
            t_tail_shocks * np.std(panic_resids) * np.float32(0.5)
        )
        panic_factor_covariance = LedoitWolf().fit(panic_factors).covariance_
        panic_covariance_floor = max(
            float(np.trace(panic_factor_covariance) / factor_returns.shape[1]) * 1e-10,
            1e-12,
        )
        panic_factor_cholesky = np.linalg.cholesky(
            panic_factor_covariance
            + np.eye(factor_returns.shape[1]) * panic_covariance_floor
        ).astype(np.float32)
        factor_normal_shocks = random_generator.standard_normal(
            (n_panic_synth, factor_returns.shape[1]),
            dtype=np.float32,
        )
        factor_chi_square = random_generator.chisquare(
            student_df,
            size=n_panic_synth,
        ).astype(np.float32)
        t_factor_shocks = (factor_normal_shocks @ panic_factor_cholesky.T) * (
            np.sqrt((student_df - 2) / factor_chi_square)[:, np.newaxis]
        ).astype(np.float32)
        synth_panic_factors = panic_factors[panic_sub_idx] + (
            t_factor_shocks * np.float32(0.5)
        )
        regime_factors[2] = np.concatenate(
            [panic_factors, synth_panic_factors],
            axis=0,
        )
        regime_residuals[2] = np.concatenate(
            [panic_resids, synth_panic_resids]
        )
    return {
        "portfolio_cond_variance": portfolio_cond_variance,
        "panic_asset_betas": panic_asset_betas,
        "regime_factors": regime_factors,
        "regime_residuals": regime_residuals,
        "n_panic_synth": n_panic_synth,
    }

def build_regime_return_pools(
    regime_factors,
    regime_residuals,
    regime_factor_coefficients,
    fhs_vol_scale,
):
    regime_sample_counts = np.array(
        [len(factors) for factors in regime_factors],
        dtype=np.int32,
    )
    if np.any(regime_sample_counts == 0):
        raise ValueError(
            "Historical factor bootstrap requires observations in every VIX regime; "
            f"counts were {regime_sample_counts.tolist()}."
        )
    factor_sample_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int32),
            np.cumsum(regime_sample_counts[:-1], dtype=np.int32),
        )
    )
    regime_return_pools = []
    for regime in range(3):
        factor_prediction = (
            regime_factors[regime]
            @ regime_factor_coefficients[regime, 1:].astype(np.float32)
            + np.float32(regime_factor_coefficients[regime, 0])
        )
        regime_return_pools.append(
            factor_prediction
            + regime_residuals[regime] * np.float32(fhs_vol_scale)
        )
    return_sample_pool = np.concatenate(regime_return_pools).astype(np.float32)
    regime_return_means = np.array(
        [np.mean(pool, dtype=np.float64) for pool in regime_return_pools],
        dtype=np.float32,
    )
    regime_return_scales = np.array(
        [
            np.std(pool - regime_return_means[regime], dtype=np.float64)
            for regime, pool in enumerate(regime_return_pools)
        ],
        dtype=np.float32,
    )
    if not np.isfinite(regime_return_scales).all() or np.any(
        regime_return_scales <= 0
    ):
        raise ValueError("Regime return pools must have finite, positive volatility.")
    regime_cascade_high_gross = np.empty(3, dtype=np.float32)
    regime_cascade_panic_gross = np.empty(3, dtype=np.float32)
    for regime, pool in enumerate(regime_return_pools):
        regime_cascade_high_gross[regime] = 1 + np.quantile(pool, 0.10)
        regime_cascade_panic_gross[regime] = 1 + np.quantile(pool, 0.02)
    return {
        "regime_sample_counts": regime_sample_counts,
        "factor_sample_offsets": factor_sample_offsets,
        "return_sample_pool": return_sample_pool,
        "regime_return_means": regime_return_means,
        "regime_return_scales": regime_return_scales,
        "regime_cascade_high_gross": regime_cascade_high_gross,
        "regime_cascade_panic_gross": regime_cascade_panic_gross,
    }

def run_monte_carlo_simulation(
    n_scenarios,
    batch_size,
    simulation_horizon_days,
    current_regime,
    regime_sample_counts,
    factor_sample_offsets,
    return_sample_pool,
    regime_return_means,
    regime_return_scales,
    regime_transition_cumulative,
    regime_cascade_high_gross,
    regime_cascade_panic_gross,
    portfolio_garch_base_variance,
    portfolio_garch_omega,
    portfolio_garch_alpha,
    portfolio_garch_gamma,
    portfolio_garch_beta,
    random_generator,
    start_time,
):
    portfolio_returns = np.empty(n_scenarios, dtype=np.float32)
    portfolio_max_drawdowns = np.empty(n_scenarios, dtype=np.float32)
    max_batch_count = min(batch_size, n_scenarios)
    drawdown_min_buffer = np.empty(max_batch_count, dtype=np.float32)
    peak_equity_buffer = np.empty(max_batch_count, dtype=np.float32)
    corrected_normal_buffer = np.empty(max_batch_count, dtype=np.float32)
    transition_threshold_buffer = np.empty(max_batch_count, dtype=np.float32)
    transition_upper_threshold_buffer = np.empty(max_batch_count, dtype=np.float32)
    transition_uniform_buffer = np.empty(max_batch_count, dtype=np.float32)
    transition_upper_state_buffer = np.empty(max_batch_count, dtype=np.bool_)
    sample_count_buffer = np.empty(max_batch_count, dtype=np.int32)
    sample_offset_buffer = np.empty(max_batch_count, dtype=np.int32)
    sample_index_buffer = np.empty(max_batch_count, dtype=np.int32)
    path_variance_ratio_buffer = np.empty(max_batch_count, dtype=np.float32)
    path_volatility_buffer = np.empty(max_batch_count, dtype=np.float32)
    volatility_update_buffer = np.empty(max_batch_count, dtype=np.float32)
    garch_gamma_update_buffer = np.empty(max_batch_count, dtype=np.float32)
    garch_negative_mask_buffer = np.empty(max_batch_count, dtype=np.bool_)
    garch_omega_ratio = np.float32(
        portfolio_garch_omega / portfolio_garch_base_variance
    )
    garch_alpha = np.float32(portfolio_garch_alpha)
    garch_gamma = np.float32(portfolio_garch_gamma)
    garch_beta = np.float32(portfolio_garch_beta)
    path_variance_ratio_min = np.float32(0.25)
    path_variance_ratio_max = np.float32(9.0)

    for batch_start in range(0, n_scenarios, batch_size):
        batch_end = min(batch_start + batch_size, n_scenarios)
        batch_count = batch_end - batch_start
        batch_returns = portfolio_returns[batch_start:batch_end]
        batch_returns.fill(1)
        batch_min_equity = drawdown_min_buffer[:batch_count]
        batch_min_equity.fill(0)
        batch_peak_equity = peak_equity_buffer[:batch_count]
        batch_peak_equity.fill(1)
        regime_states = np.full(batch_count, current_regime, dtype=np.int8)
        corrected_normal = corrected_normal_buffer[:batch_count]
        transition_thresholds = transition_threshold_buffer[:batch_count]
        transition_upper_thresholds = transition_upper_threshold_buffer[:batch_count]
        transition_uniforms = transition_uniform_buffer[:batch_count]
        transition_upper_states = transition_upper_state_buffer[:batch_count]
        sample_counts = sample_count_buffer[:batch_count]
        sample_offsets = sample_offset_buffer[:batch_count]
        sample_indices = sample_index_buffer[:batch_count]
        path_variance_ratio = path_variance_ratio_buffer[:batch_count]
        path_variance_ratio.fill(1)
        path_volatility = path_volatility_buffer[:batch_count]
        volatility_update = volatility_update_buffer[:batch_count]
        garch_gamma_update = garch_gamma_update_buffer[:batch_count]
        garch_negative_mask = garch_negative_mask_buffer[:batch_count]

        for _ in range(simulation_horizon_days):
            random_generator.random(
                batch_count, dtype=np.float32, out=transition_uniforms
            )
            np.take(regime_sample_counts, regime_states, out=sample_counts)
            np.multiply(
                transition_uniforms,
                sample_counts,
                out=transition_uniforms,
                casting="unsafe",
            )
            np.floor(transition_uniforms, out=transition_uniforms)
            np.copyto(sample_indices, transition_uniforms, casting="unsafe")
            np.take(factor_sample_offsets, regime_states, out=sample_offsets)
            np.add(sample_indices, sample_offsets, out=sample_indices)
            np.take(return_sample_pool, sample_indices, out=corrected_normal)
            np.take(regime_return_means, regime_states, out=transition_thresholds)
            np.subtract(corrected_normal, transition_thresholds, out=corrected_normal)
            np.take(
                regime_return_scales,
                regime_states,
                out=transition_upper_thresholds,
            )
            np.sqrt(path_variance_ratio, out=path_volatility)
            np.multiply(corrected_normal, path_volatility, out=corrected_normal)

            np.divide(
                corrected_normal,
                transition_upper_thresholds,
                out=volatility_update,
            )
            np.square(volatility_update, out=volatility_update)
            np.multiply(
                volatility_update,
                garch_alpha,
                out=volatility_update,
            )
            np.multiply(
                path_variance_ratio,
                garch_beta,
                out=path_variance_ratio,
            )
            np.add(path_variance_ratio, garch_omega_ratio, out=path_variance_ratio)
            np.less(corrected_normal, np.float32(0), out=garch_negative_mask)
            np.divide(
                corrected_normal,
                transition_upper_thresholds,
                out=garch_gamma_update,
            )
            np.square(garch_gamma_update, out=garch_gamma_update)
            np.multiply(
                garch_gamma_update,
                garch_gamma,
                out=garch_gamma_update,
            )
            np.add(
                volatility_update,
                garch_gamma_update,
                out=volatility_update,
                where=garch_negative_mask,
            )
            np.add(
                path_variance_ratio,
                volatility_update,
                out=path_variance_ratio,
            )
            np.clip(
                path_variance_ratio,
                path_variance_ratio_min,
                path_variance_ratio_max,
                out=path_variance_ratio,
            )
            np.add(corrected_normal, transition_thresholds, out=corrected_normal)
            np.maximum(corrected_normal, np.float32(-0.999999), out=corrected_normal)
            np.add(corrected_normal, np.float32(1), out=corrected_normal)
            np.multiply(batch_returns, corrected_normal, out=batch_returns)
            np.maximum(
                batch_peak_equity,
                batch_returns,
                out=batch_peak_equity,
            )
            np.divide(batch_returns, batch_peak_equity, out=path_volatility)
            np.subtract(path_volatility, np.float32(1), out=path_volatility)
            np.minimum(
                batch_min_equity,
                path_volatility,
                out=batch_min_equity,
            )
            np.take(
                regime_transition_cumulative[:, 0],
                regime_states,
                out=transition_thresholds,
            )
            np.take(
                regime_transition_cumulative[:, 1],
                regime_states,
                out=transition_upper_thresholds,
            )
            random_generator.random(
                batch_count, dtype=np.float32, out=transition_uniforms
            )
            np.greater_equal(
                transition_uniforms, transition_thresholds, out=regime_states
            )
            np.greater_equal(
                transition_uniforms,
                transition_upper_thresholds,
                out=transition_upper_states,
            )
            np.add(regime_states, transition_upper_states, out=regime_states)

            # Escalate regimes only after returns cross historical regime-tail cutoffs.
            np.take(
                regime_cascade_high_gross,
                regime_states,
                out=transition_thresholds,
            )
            np.take(
                regime_cascade_panic_gross,
                regime_states,
                out=transition_upper_thresholds,
            )
            np.less(
                corrected_normal,
                transition_thresholds,
                out=transition_upper_states,
            )
            np.maximum(
                regime_states,
                np.int8(1),
                out=regime_states,
                where=transition_upper_states,
            )
            np.less(
                corrected_normal,
                transition_upper_thresholds,
                out=transition_upper_states,
            )
            np.maximum(
                regime_states,
                np.int8(2),
                out=regime_states,
                where=transition_upper_states,
            )

        np.subtract(batch_returns, np.float32(1), out=batch_returns)
        portfolio_max_drawdowns[batch_start:batch_end] = batch_min_equity

    computation_time = (time.perf_counter() - start_time) * 1000
    return portfolio_returns, portfolio_max_drawdowns, computation_time

def summarize_simulation_results(
    portfolio_returns,
    portfolio_max_drawdowns,
    simulation_horizon_days,
    initial_portfolio_value,
    live_risk_free_rate,
    historical_portfolio_returns,
    n_scenarios,
):
    mean_ret = np.mean(portfolio_returns, dtype=np.float64)
    volatility = np.std(portfolio_returns, dtype=np.float64)
    ann_vol = volatility * np.sqrt(252 / simulation_horizon_days)
    expected_pnl = mean_ret * initial_portfolio_value
    win_rate = np.mean(portfolio_returns > 0) * 100
    var_95_pct, var_99_pct = -np.percentile(portfolio_returns, [5, 1])
    var_95_dollar = var_95_pct * initial_portfolio_value
    mdd_95_pct, mdd_99_pct = -np.percentile(portfolio_max_drawdowns, [5, 1])
    mdd_95_dollar = mdd_95_pct * initial_portfolio_value
    mdd_99_dollar = mdd_99_pct * initial_portfolio_value
    mdd_ratio = mdd_95_dollar / var_95_dollar if var_95_dollar > 0 else 1.0
    cvar_95_pct = -np.mean(portfolio_returns[portfolio_returns <= -var_95_pct])
    cvar_95_dollar = cvar_95_pct * initial_portfolio_value
    period_target = (1 + live_risk_free_rate) ** (simulation_horizon_days / 252) - 1
    sharpe_ratio, sortino_ratio = annualized_risk_ratios(
        portfolio_returns,
        period_target,
        252 / simulation_horizon_days,
    )
    historical_daily_target = live_risk_free_rate / 252
    historical_sharpe_ratio, _ = annualized_risk_ratios(
        historical_portfolio_returns,
        historical_daily_target,
        252,
    )
    im_better_than_strange = np.ceil(win_rate / 100 * n_scenarios) - 1
    return {
        "mean_ret": mean_ret,
        "volatility": volatility,
        "ann_vol": ann_vol,
        "expected_pnl": expected_pnl,
        "win_rate": win_rate,
        "var_95_pct": var_95_pct,
        "var_99_pct": var_99_pct,
        "var_95_dollar": var_95_dollar,
        "mdd_95_pct": mdd_95_pct,
        "mdd_99_pct": mdd_99_pct,
        "mdd_95_dollar": mdd_95_dollar,
        "mdd_99_dollar": mdd_99_dollar,
        "mdd_ratio": mdd_ratio,
        "cvar_95_pct": cvar_95_pct,
        "cvar_95_dollar": cvar_95_dollar,
        "period_target": period_target,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "historical_sharpe_ratio": historical_sharpe_ratio,
        "im_better_than_strange": im_better_than_strange,
    }

def run_historical_backtest(
    returns,
    n_days,
    n_assets,
    backtest_lookback,
    backtest_days,
    initial_portfolio_value,
    trailing_vix,
    regime_vix_threshold,
    max_rebalance_interval,
    relative_drift_limit,
    minimum_absolute_drift,
    lagged_average_dollar_volume,
    lagged_return_volatility,
    sqrt_impact_coefficient,
    max_slippage_rate,
    broker_fee_rate,
    sec_sell_fee_rate,
    regime_tail_factor,
):
    backtest_returns = np.empty(backtest_days, dtype=np.float64)
    backtest_equity = np.empty(backtest_days + 1, dtype=np.float64)
    backtest_equity[0] = initial_portfolio_value
    held_weights = np.full(n_assets, 1 / n_assets)
    rebalance_target_weights = None
    days_since_rebalance = 21
    rebalance_count = 0
    stress_rebalance_count = 0
    total_transaction_costs = 0.0

    for backtest_index, return_index in enumerate(range(backtest_lookback, n_days)):
        stress_override = trailing_vix[return_index] > regime_vix_threshold
        rebalance_due = (
            rebalance_target_weights is None
            or days_since_rebalance >= max_rebalance_interval
            or stress_override
        )
        if not rebalance_due:
            drift_limits = np.maximum(
                rebalance_target_weights * relative_drift_limit,
                minimum_absolute_drift,
            )
            rebalance_due = np.any(
                np.abs(held_weights - rebalance_target_weights) > drift_limits
            )

        if rebalance_due:
            lookback_returns = returns[return_index - backtest_lookback:return_index]
            local_covariance = LedoitWolf().fit(lookback_returns).covariance_
            covariance_scale = np.trace(local_covariance) / n_assets
            local_covariance += np.eye(n_assets) * max(covariance_scale * 1e-8, 1e-12)
            if stress_override:
                diagonal = np.diag_indices_from(local_covariance)
                local_covariance[diagonal] *= regime_tail_factor
                stress_rebalance_count += 1
            rebalance_target_weights = minimum_variance_weights(
                local_covariance,
                held_weights,
            )
            daily_weights = rebalance_target_weights
            days_since_rebalance = 0
            rebalance_count += 1
        else:
            daily_weights = held_weights

        delta_weights = np.abs(daily_weights - held_weights)
        turnover = delta_weights.sum()
        sell_turnover = np.maximum(held_weights - daily_weights, 0).sum()
        current_equity = backtest_equity[backtest_index]
        trade_notionals = current_equity * delta_weights
        trade_fractions_of_adv = (
            trade_notionals / lagged_average_dollar_volume[return_index]
        )
        slippage_rates = np.minimum(
            sqrt_impact_coefficient
            * lagged_return_volatility[return_index]
            * np.sqrt(trade_fractions_of_adv),
            max_slippage_rate,
        )
        slippage_cost = trade_notionals @ slippage_rates
        transaction_cost = (
            current_equity * (turnover * broker_fee_rate + sell_turnover * sec_sell_fee_rate)
            + slippage_cost
        )
        total_transaction_costs += transaction_cost
        backtest_returns[backtest_index] = (
            daily_weights @ returns[return_index] - transaction_cost / current_equity
        )
        backtest_equity[backtest_index + 1] = current_equity * (
            1 + backtest_returns[backtest_index]
        )
        post_return_weights = daily_weights * (1 + returns[return_index])
        post_return_total = post_return_weights.sum()
        if not np.isfinite(post_return_total) or post_return_total <= 0:
            raise ValueError(
                f"Invalid post-return portfolio weights on backtest day {return_index}."
            )
        held_weights = post_return_weights / post_return_total
        days_since_rebalance += 1

    return (
        backtest_returns,
        backtest_equity,
        rebalance_count,
        stress_rebalance_count,
        total_transaction_costs,
    )

def prepare_model_calibration(calibration_random_generator):
    global aligned_vix, close, current_regime, current_regime_name, daily_midas_scale, daily_returns, dollar_volume, dynamic_kurtosis, dynamic_skewness, excluded_factor_rows, factor_returns, factor_sample_offsets, fhs_vol_scale, forecast_variances, forecast_volatility, historical_portfolio_returns, historical_regimes, lagged_average_dollar_volume, lagged_return_volatility, live_risk_free_rate, market, midas_sensitivity, n_assets, n_days, n_panic_synth, optimized_weights, panic_asset_betas, portfolio_cond_variance, portfolio_cond_volatility, portfolio_dynamics, portfolio_garch_alpha, portfolio_garch_base_variance, portfolio_garch_beta, portfolio_garch_gamma, portfolio_garch_omega, regime_cascade_high_gross, regime_cascade_panic_gross, regime_factor_coefficients, regime_factors, regime_model, regime_names, regime_raw_counts, regime_residuals, regime_return_means, regime_return_scales, regime_sample_counts, regime_sample_masks, regime_transition, regime_transition_cumulative, return_pools, return_sample_pool, returns, standardized_residuals, start_time, stress_data, student_degrees, student_df, valid_factor_rows, valid_vix, vix, volume
    global market_history_years, panic_observations, panic_target_reached
    market = prepare_market_data(tickers, factor_tickers)
    close = market["close"]
    volume = market["volume"]
    daily_returns = market["daily_returns"]
    vix = market["vix"]
    live_risk_free_rate = market["live_risk_free_rate"]
    returns = market["returns"]
    n_days = market["n_days"]
    n_assets = market["n_assets"]
    factor_returns = market["factor_returns"]
    panic_observations = market["panic_observations"]
    panic_target_reached = market["panic_target_reached"]
    market_history_years = market["market_history_years"]
    dollar_volume = market["dollar_volume"]
    lagged_average_dollar_volume = market["lagged_average_dollar_volume"]
    lagged_return_volatility = market["lagged_return_volatility"]
    start_time = time.perf_counter()

    daily_midas_scale, midas_sensitivity = build_midas_scale(daily_returns, vix)

    print("VIX MIDAS signals built; fitting GARCH models...", end="\n\n")
    standardized_residuals, forecast_variances, student_degrees = (
        fit_asset_garch_models(returns, daily_midas_scale)
    )

    student_df = max(4.1, float(np.median(student_degrees)))
    forecast_volatility = np.sqrt(forecast_variances)
    optimized_weights = bootstrap_minimum_variance_weights(
        standardized_residuals,
        forecast_volatility,
        calibration_random_generator,
    )
    portfolio_dynamics = fit_portfolio_dynamics(
        standardized_residuals,
        forecast_volatility,
        returns,
        weights,
    )
    historical_portfolio_returns = portfolio_dynamics["historical_portfolio_returns"]
    portfolio_cond_variance = portfolio_dynamics["portfolio_cond_variance"]
    portfolio_cond_volatility = portfolio_dynamics["portfolio_cond_volatility"]
    portfolio_garch_base_variance = portfolio_dynamics["portfolio_garch_base_variance"]
    portfolio_garch_omega = portfolio_dynamics["portfolio_garch_omega"]
    portfolio_garch_alpha = portfolio_dynamics["portfolio_garch_alpha"]
    portfolio_garch_gamma = portfolio_dynamics["portfolio_garch_gamma"]
    portfolio_garch_beta = portfolio_dynamics["portfolio_garch_beta"]
    fhs_vol_scale = portfolio_dynamics["fhs_vol_scale"]
    dynamic_skewness = portfolio_dynamics["dynamic_skewness"]
    dynamic_kurtosis = portfolio_dynamics["dynamic_kurtosis"]

    regime_model = fit_regime_factor_model(
        vix,
        daily_returns,
        factor_returns,
        historical_portfolio_returns,
    )
    aligned_vix = regime_model["aligned_vix"]
    valid_vix = regime_model["valid_vix"]
    historical_regimes = regime_model["historical_regimes"]
    current_regime = regime_model["current_regime"]
    current_regime_name = regime_model["current_regime_name"]
    regime_transition = regime_model["regime_transition"]
    regime_transition_cumulative = regime_model["regime_transition_cumulative"]
    valid_factor_rows = regime_model["valid_factor_rows"]
    excluded_factor_rows = regime_model["excluded_factor_rows"]
    regime_sample_masks = regime_model["regime_sample_masks"]
    regime_names = regime_model["regime_names"]
    regime_factor_coefficients = regime_model["regime_factor_coefficients"]
    regime_factors = regime_model["regime_factors"]
    regime_residuals = regime_model["regime_residuals"]
    regime_raw_counts = regime_model["regime_raw_counts"]

    stress_data = build_stress_and_panic_data(
        standardized_residuals,
        forecast_volatility,
        weights,
        current_regime,
        regime_sample_masks,
        regime_factors,
        regime_residuals,
        factor_returns,
        student_df,
        panic_observation_target,
        calibration_random_generator,
    )
    portfolio_cond_variance = stress_data["portfolio_cond_variance"]
    panic_asset_betas = stress_data["panic_asset_betas"]
    regime_factors = stress_data["regime_factors"]
    regime_residuals = stress_data["regime_residuals"]
    n_panic_synth = stress_data["n_panic_synth"]

    return_pools = build_regime_return_pools(
        regime_factors,
        regime_residuals,
        regime_factor_coefficients,
        fhs_vol_scale,
    )
    regime_sample_counts = return_pools["regime_sample_counts"]
    factor_sample_offsets = return_pools["factor_sample_offsets"]
    return_sample_pool = return_pools["return_sample_pool"]
    regime_return_means = return_pools["regime_return_means"]
    regime_return_scales = return_pools["regime_return_scales"]
    regime_cascade_high_gross = return_pools["regime_cascade_high_gross"]
    regime_cascade_panic_gross = return_pools["regime_cascade_panic_gross"]

    print("GARCH residuals and regime-conditioned factor model ready!", end="\n\n")

def run_simulation_analysis(path_random_generator):
    global computation_time, portfolio_max_drawdowns, portfolio_returns
    portfolio_returns, portfolio_max_drawdowns, computation_time = (
        run_monte_carlo_simulation(
            n_scenarios,
            batch_size,
            simulation_horizon_days,
            current_regime,
            regime_sample_counts,
            factor_sample_offsets,
            return_sample_pool,
            regime_return_means,
            regime_return_scales,
            regime_transition_cumulative,
            regime_cascade_high_gross,
            regime_cascade_panic_gross,
            portfolio_garch_base_variance,
            portfolio_garch_omega,
            portfolio_garch_alpha,
            portfolio_garch_gamma,
            portfolio_garch_beta,
            path_random_generator,
            start_time,
        )
    )

    return summarize_simulation_results(
        portfolio_returns,
        portfolio_max_drawdowns,
        simulation_horizon_days,
        initial_portfolio_value,
        live_risk_free_rate,
        historical_portfolio_returns,
        n_scenarios,
    )

def apply_simulation_summary(
    portfolio_returns,
    portfolio_max_drawdowns,
    total_scenarios,
):
    global ann_vol, component_cvar_dollar, component_risk_pct, component_var_dollar, cvar_95_dollar, cvar_95_pct, expected_pnl, factor_model_table, historical_sharpe_ratio, im_better_than_strange, mdd_95_dollar, mdd_95_pct, mdd_99_dollar, mdd_99_pct, mdd_ratio, mean_ret, optimized_weight_table, period_target, portfolio_weight_table, risk_ranking, sharpe_ratio, simulation_summary, sortino_ratio, var_95_dollar, var_95_pct, var_99_pct, volatility, win_rate
    simulation_summary = summarize_simulation_results(
        portfolio_returns,
        portfolio_max_drawdowns,
        simulation_horizon_days,
        initial_portfolio_value,
        live_risk_free_rate,
        historical_portfolio_returns,
        total_scenarios,
    )
    (
        mean_ret, volatility, ann_vol, expected_pnl, win_rate, var_95_pct, var_99_pct,
        var_95_dollar, mdd_95_pct, mdd_99_pct, mdd_95_dollar, mdd_99_dollar,
        mdd_ratio, cvar_95_pct, cvar_95_dollar, period_target, sharpe_ratio,
        sortino_ratio, historical_sharpe_ratio, im_better_than_strange,
    ) = (
        simulation_summary["mean_ret"],
        simulation_summary["volatility"],
        simulation_summary["ann_vol"],
        simulation_summary["expected_pnl"],
        simulation_summary["win_rate"],
        simulation_summary["var_95_pct"],
        simulation_summary["var_99_pct"],
        simulation_summary["var_95_dollar"],
        simulation_summary["mdd_95_pct"],
        simulation_summary["mdd_99_pct"],
        simulation_summary["mdd_95_dollar"],
        simulation_summary["mdd_99_dollar"],
        simulation_summary["mdd_ratio"],
        simulation_summary["cvar_95_pct"],
        simulation_summary["cvar_95_dollar"],
        simulation_summary["period_target"],
        simulation_summary["sharpe_ratio"],
        simulation_summary["sortino_ratio"],
        simulation_summary["historical_sharpe_ratio"],
        simulation_summary["im_better_than_strange"],
    )

    portfolio_weight_table = np.column_stack(
        (tickers, np.char.mod("%.2f%%", weights * 100))
    )
    optimized_weight_table = np.column_stack(
        (tickers, np.char.mod("%.2f%%", optimized_weights * 100))
    )
    factor_model_table = np.column_stack(
        (
            np.repeat(regime_names, len(factor_names) + 1),
            np.tile(["Intercept", *factor_names], 3),
            np.char.mod("%.6f", regime_factor_coefficients.ravel()),
        )
    )
    # --- PANIC-COVARIANCE STRESS SENSITIVITY ALLOCATION ---
    component_risk_pct = weights * panic_asset_betas * 100.0
    component_var_dollar = (weights * panic_asset_betas) * var_95_dollar
    component_cvar_dollar = (weights * panic_asset_betas) * cvar_95_dollar
    risk_ranking = np.argsort(-component_risk_pct)

def run_simulation_ensemble():
    global computation_time, ensemble_var_95_mean, ensemble_var_95_median, ensemble_var_95_min, ensemble_var_95_max, ensemble_var_95_std, ensemble_var_95_standard_error, ensemble_var_95_p05, ensemble_var_95_p95, n_total_scenarios, optimized_weights, portfolio_max_drawdowns, portfolio_returns, regime_cascade_high_gross, regime_cascade_panic_gross
    if n_calibration_runs < 1:
        raise ValueError("n_calibration_runs must be at least 1.")
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be at least 1.")

    calibration_seed_sequences = np.random.SeedSequence(
        calibration_seed
    ).spawn(n_calibration_runs)
    simulation_seed_sequences = np.random.SeedSequence(
        simulation_seed
    ).spawn(n_calibration_runs)
    calibration_generators = [
        np.random.default_rng(seed_sequence)
        for seed_sequence in calibration_seed_sequences
    ]
    simulation_generators = [
        np.random.default_rng(seed_sequence)
        for seed_sequence in simulation_seed_sequences
    ]
    scenario_returns = []
    scenario_drawdowns = []
    run_var_95 = []
    run_optimized_weights = []
    run_cascade_high = []
    run_cascade_panic = []
    total_computation_time = 0.0

    for run_index, (calibration_generator, simulation_generator) in enumerate(
        zip(calibration_generators, simulation_generators),
        start=1,
    ):
        print(
            f"[ENSEMBLE] Calibration {run_index}/{n_calibration_runs}...",
            end="\n\n",
        )
        prepare_model_calibration(calibration_generator)
        run_summary = run_simulation_analysis(simulation_generator)
        scenario_returns.append(portfolio_returns.copy())
        scenario_drawdowns.append(portfolio_max_drawdowns.copy())
        run_var_95.append(run_summary["var_95_pct"])
        run_optimized_weights.append(optimized_weights.copy())
        run_cascade_high.append(regime_cascade_high_gross.copy())
        run_cascade_panic.append(regime_cascade_panic_gross.copy())
        total_computation_time += computation_time

    portfolio_returns = np.concatenate(scenario_returns)
    portfolio_max_drawdowns = np.concatenate(scenario_drawdowns)
    n_total_scenarios = len(portfolio_returns)
    computation_time = total_computation_time
    ensemble_var_95_mean = float(np.mean(run_var_95))
    ensemble_var_95_median = float(np.median(run_var_95))
    ensemble_var_95_min = float(np.min(run_var_95))
    ensemble_var_95_max = float(np.max(run_var_95))
    ensemble_var_95_std = float(np.std(run_var_95, ddof=1)) if len(run_var_95) > 1 else 0.0
    ensemble_var_95_standard_error = (
        ensemble_var_95_std / np.sqrt(len(run_var_95))
        if len(run_var_95) > 1
        else 0.0
    )
    ensemble_var_95_p05, ensemble_var_95_p95 = np.percentile(
        run_var_95,
        [5, 95],
    )
    optimized_weights = np.mean(run_optimized_weights, axis=0)
    optimized_weights /= optimized_weights.sum()
    regime_cascade_high_gross = np.mean(run_cascade_high, axis=0)
    regime_cascade_panic_gross = np.mean(run_cascade_panic, axis=0)
    apply_simulation_summary(
        portfolio_returns,
        portfolio_max_drawdowns,
        n_total_scenarios,
    )

def run_liquidity_backtest_stage():
    global backtest_daily_target, backtest_days, backtest_drawdowns, backtest_equity, backtest_max_drawdown, backtest_net_return, backtest_returns, backtest_sharpe, backtest_sortino, broker_fee_rate, calibration_impact, full_lcvar_95_dollar, full_liquidation_haircut, full_liquidation_slippage, full_lvar_95_dollar, full_lvar_95_pct, historical_impact_rates, lcvar_95_dollar, liquidation_urgency_fraction, lvar_95_dollar, lvar_95_pct, max_slippage_rate, position_haircuts, rebalance_count, reference_trade_notional, sec_sell_fee_rate, slippage_rate, sqrt_impact_coefficient, stress_rebalance_count, stressed_impact, total_liquidity_haircut, total_transaction_costs, trailing_vix, urgency_slippage, valid_execution_data, var_threshold_position_dollars, var_threshold_terminal_equity
    backtest_days = n_days - backtest_lookback
    if backtest_days <= 0:
        raise ValueError("At least 253 daily return observations are required for backtesting.")

    broker_fee_rate = 0.00005
    slippage_rate = 0.0002
    sec_sell_fee_rate = 0.0000278
    max_slippage_rate = 5 * slippage_rate
    valid_execution_data = (
        np.isfinite(lagged_average_dollar_volume)
        & (lagged_average_dollar_volume > 0)
        & np.isfinite(lagged_return_volatility)
    )
    if not valid_execution_data[backtest_lookback:].all():
        raise ValueError(
            "Insufficient lagged dollar-volume or volatility history for backtest slippage."
        )
    reference_trade_notional = initial_portfolio_value * 0.01
    calibration_impact = (
        lagged_return_volatility[:backtest_lookback]
        * np.sqrt(
            reference_trade_notional
            / lagged_average_dollar_volume[:backtest_lookback]
        )
    )
    calibration_impact = calibration_impact[
        np.isfinite(calibration_impact) & (calibration_impact > 0)
    ]
    if calibration_impact.size == 0:
        raise ValueError("Could not calibrate square-root market-impact slippage.")
    sqrt_impact_coefficient = slippage_rate / np.median(calibration_impact)

    # --- Stressed Liquidity Haircut & Liquidity-Adjusted VaR (L-VaR) ---
    var_threshold_terminal_equity = initial_portfolio_value * max(
        0.0,
        1.0 - var_95_pct,
    )
    var_threshold_position_dollars = weights * var_threshold_terminal_equity
    historical_impact_rates = (
        lagged_return_volatility
        * np.sqrt(var_threshold_position_dollars / lagged_average_dollar_volume)
    )
    stressed_impact = np.nanpercentile(historical_impact_rates, 95, axis=0)
    liquidation_urgency_fraction = 0.20
    full_liquidation_slippage = np.minimum(
        sqrt_impact_coefficient * stressed_impact,
        max_slippage_rate,
    )
    urgency_slippage = np.minimum(
        sqrt_impact_coefficient
        * np.sqrt(liquidation_urgency_fraction)
        * stressed_impact,
        max_slippage_rate,
    )
    full_liquidation_haircut = float(
        np.sum(var_threshold_position_dollars * full_liquidation_slippage)
    )
    position_haircuts = (
        var_threshold_position_dollars
        * liquidation_urgency_fraction
        * urgency_slippage
    )
    total_liquidity_haircut = float(np.sum(position_haircuts))
    lvar_95_dollar = var_95_dollar + total_liquidity_haircut
    lcvar_95_dollar = cvar_95_dollar + total_liquidity_haircut
    lvar_95_pct = lvar_95_dollar / initial_portfolio_value
    full_lvar_95_dollar = var_95_dollar + full_liquidation_haircut
    full_lcvar_95_dollar = cvar_95_dollar + full_liquidation_haircut
    full_lvar_95_pct = full_lvar_95_dollar / initial_portfolio_value
    trailing_vix = (
        vix.reindex(daily_returns.index)
        .ffill()
        .rolling(regime_lookback, min_periods=regime_lookback)
        .mean()
        .shift(1)
        .to_numpy()
    )
    if not np.isfinite(trailing_vix[backtest_lookback:]).all():
        raise ValueError("Insufficient aligned VIX history for the backtest regime trigger.")

    (
        backtest_returns,
        backtest_equity,
        rebalance_count,
        stress_rebalance_count,
        total_transaction_costs,
    ) = run_historical_backtest(
        returns,
        n_days,
        n_assets,
        backtest_lookback,
        backtest_days,
        initial_portfolio_value,
        trailing_vix,
        regime_vix_threshold,
        max_rebalance_interval,
        relative_drift_limit,
        minimum_absolute_drift,
        lagged_average_dollar_volume,
        lagged_return_volatility,
        sqrt_impact_coefficient,
        max_slippage_rate,
        broker_fee_rate,
        sec_sell_fee_rate,
        regime_tail_factor,
    )

    backtest_net_return = (backtest_equity[-1] / initial_portfolio_value - 1) * 100
    backtest_drawdowns = backtest_equity / np.maximum.accumulate(backtest_equity) - 1
    backtest_max_drawdown = np.min(backtest_drawdowns) * 100
    backtest_daily_target = live_risk_free_rate / 252
    backtest_sharpe, backtest_sortino = annualized_risk_ratios(
        backtest_returns,
        backtest_daily_target,
        252,
    )

def render_report():
    separator = "=" * 78
    rule = "-" * 78
    historical_annualized_volatility = (
        np.std(historical_portfolio_returns, dtype=np.float64) * np.sqrt(252)
    )
    backtest_start_date = daily_returns.index[backtest_lookback].strftime("%Y-%m-%d")
    backtest_end_date = daily_returns.index[-1].strftime("%Y-%m-%d")
    portfolio_weights_text = format_table(
        portfolio_weight_table, ("   %-10s ", "%8s")
    )
    optimized_weights_text = format_table(
        optimized_weight_table, ("   %-10s ", "%8s")
    )
    factor_model_text = format_table(
        factor_model_table,
        ("   %-18s", "%-24s", "%10s"),
        delimiter="  ",
    )
    risk_rows = [
        f"  {tickers[idx]:<7} {weights[idx] * 100:>7.2f}% "
        f"{panic_asset_betas[idx]:>14.3f} {component_risk_pct[idx]:>12.2f}% "
        f"${component_var_dollar[idx]:>13,.2f} "
        f"${component_cvar_dollar[idx]:>13,.2f}   "
        f"{'Tail Aggressor' if panic_asset_betas[idx] > 1.15 else ('Diversifier' if panic_asset_betas[idx] < 0.85 else 'Neutral')}"
        for idx in risk_ranking
    ]
    risk_table = "\n".join(
        [
            f"  {'Ticker':<7} {'Weight':>8} {'Panic Beta':>14} "
            f"{'Stress Share':>13} {'VaR Sensitivity ($)':>19} "
            f"{'CVaR Sensitivity ($)':>20}   Role",
            f"  {'-' * 76}",
            *risk_rows,
            f"  {'-' * 76}",
            f"  {'TOTAL':<7} {100.0:>7.2f}% {1.0:>14.3f} "
            f"{100.0:>12.2f}% ${var_95_dollar:>13,.2f} "
            f"${cvar_95_dollar:>13,.2f}",
        ]
    )
    sections = [
        f"{separator}\n{'HISTORICAL BACKTEST -- OBSERVED MARKET DATA'.center(78)}\n{rule}",
        f" \n {'Strategy':<45}  Long-only minimum variance\n"
        f" \n {'Test period':<45}  {backtest_start_date} to {backtest_end_date}\n"
        f" \n {'Lookback / test observations':<45}  {backtest_lookback} / {backtest_days}\n"
        f" \n {'Rebalance events / VIX overrides':<45}  {rebalance_count} / {stress_rebalance_count}\n"
        f" \n {'Rebalance rule':<45}  20% relative / 0.5% floor / 21-day max / VIX stress\n"
        f" \n {'Cumulative transaction costs':<45}  ${total_transaction_costs:,.2f}\n"
        f" \n {'Backtest cumulative net return':<45}  {backtest_net_return:+.2f}%\n"
        f" \n {'Backtest maximum peak-to-trough drawdown':<45}  {backtest_max_drawdown:.2f}%\n"
        f" \n {'Backtest Sharpe (annualized)':<45}  {backtest_sharpe:.3f}\n"
        f" \n {'Backtest Sortino (annualized)':<45}  {backtest_sortino:.3f}\n"
        " \n Historical replay using observed daily close-to-close returns; includes modeled trading frictions.",
        f"\n{separator}\n{f'GARCH-MIDAS QUANT DASHBOARD -- {n_total_scenarios:,} SCENARIOS ({n_calibration_runs} CALIBRATIONS, {simulation_horizon_days}-DAY HORIZON)'.center(78)}\n{separator}",
        f" \n OBSERVED DATA BASIS\n"
        " \n Data source: Yahoo Finance auto-adjusted closing prices in local SQLite cache\n"
        f" \n {'Aligned observation period':<45}  {daily_returns.index[0].strftime('%Y-%m-%d')} to {daily_returns.index[-1].strftime('%Y-%m-%d')}\n"
        f" \n {'Usable portfolio return observations':<45}  {len(historical_portfolio_returns):,}\n"
        f" \n {'Fixed-weight historical annualized volatility':<45}  {historical_annualized_volatility * 100:.2f}%\n"
        f" \n {'Observed panic-classified days':<45}  {panic_observations:,} (VIX level + 5-day momentum rule)\n"
        f" \n {'Panic sample target':<45}  {panic_observation_target:,} "
        f"({'met' if panic_target_reached else 'not met'}; {market_history_years:.1f} years searched)\n"
        " \n Panic observations are historical dates classified by the model's VIX rule.\n"
        " \n This count describes the calibration sample, not future panic frequency.",
        f" \n Your portfolio weights\n\n{portfolio_weights_text}\n\n"
        f" \n Min-variance comparison\n\n{optimized_weights_text}",
        f" \n {'Median fitted GARCH-t degrees of freedom':<45}  {student_df:.2f}\n"
        f" \n {'Starting volatility regime':<45}  {current_regime_name}\n"
        f" \n {'Usable historical factor observations':<45}  Calm {regime_raw_counts[0]:,} / High {regime_raw_counts[1]:,} / Panic {regime_raw_counts[2]:,}\n"
        f" \n {'Bootstrap pool rows (real + synthetic)':<45}  Calm {regime_sample_counts[0]:,} / High {regime_sample_counts[1]:,} / Panic {regime_sample_counts[2]:,}\n\n"
        f" \n Regime-conditioned factor loadings\n"
        f" \n  {'Regime':<18}  {'Factor':<24}  {'Loading':>10}\n\n{factor_model_text}",
        f" \n {'Synthetic panic rows (not observed history)':<45}  {n_panic_synth:,}\n"
        f" \n {'Factor rows excluded (missing data)':<45}  {excluded_factor_rows:,}\n"
        f" \n {'Risk-free rate input (13-week T-Bill)':<45}  {live_risk_free_rate * 100:.2f}%\n"
        f" \n {'GARCH FHS volatility multiplier':<45}  {fhs_vol_scale:.3f}x\n"
        f" \n {'Calibration runs':<45}  {n_calibration_runs:,}\n"
        f" \n {'Calibration / path seeds':<45}  {calibration_seed!s} / {simulation_seed!s} (None = fresh randomness)\n"
        f" \n {'Mean run-level 95% VaR':<45}  {ensemble_var_95_mean * 100:.2f}%\n"
        f" \n {'Median run-level 95% VaR':<45}  {ensemble_var_95_median * 100:.2f}%\n"
        f" \n {'Run-level VaR standard deviation':<45}  {ensemble_var_95_std * 100:.2f} percentage points\n"
        f" \n {'Standard error of mean run-level VaR':<45}  {ensemble_var_95_standard_error * 100:.2f} percentage points\n"
        f" \n {'Run-level VaR 5th-95th percentile':<45}  {ensemble_var_95_p05 * 100:.2f}% - {ensemble_var_95_p95 * 100:.2f}% (empirical)\n"
        f" \n {'Run-level VaR observed min / max':<45}  {ensemble_var_95_min * 100:.2f}% / {ensemble_var_95_max * 100:.2f}%\n"
        " \n Ensemble spread includes calibration randomness and finite path sampling.\n"
        " \n Run-level percentiles are empirical summaries, not formal confidence bounds.",
        f" \n {'Path volatility memory':<45}  GJR-GARCH; variance ratio capped 0.25x-9.00x\n"
        f" \n {'Cascade cutoffs (current regime)':<45}  {(regime_cascade_high_gross[current_regime] - 1) * 100:.2f}% / {(regime_cascade_panic_gross[current_regime] - 1) * 100:.2f}%\n"
        f" \n {'Model + simulation time':<45}  {computation_time:.2f} ms\n"
        f" \n {'Initial portfolio value':<45}  ${initial_portfolio_value:,.2f}",
        f"\n{rule}\nMODEL-SIMULATED OUTCOMES -- {simulation_horizon_days}-DAY HORIZON\n{rule}\n"
        f" \n {'Mean simulated horizon return':<45}  {mean_ret * 100:+.3f}%\n"
        f" \n {'Mean simulated P&L':<45}  ${expected_pnl:+,.2f}\n"
        f" \n {'Median simulated horizon return':<45}  {np.median(portfolio_returns) * 100:+.3f}%\n"
        f" \n {'5th / 95th percentile horizon return':<45}  {np.percentile(portfolio_returns, 5) * 100:+.2f}% / {np.percentile(portfolio_returns, 95) * 100:+.2f}%\n"
        f" \n {'Horizon terminal-return standard deviation':<45}  {volatility * 100:.2f}%\n"
        f" \n {'Profitable simulated paths':<45}  {win_rate:.2f}%\n"
        f" \n {'Annualized-equivalent Sharpe (model)':<45}  {sharpe_ratio:.3f}\n"
        f" \n {'Annualized-equivalent Sortino (model)':<45}  {sortino_ratio:.3f}\n"
        f" \n {'Historical daily-return Sharpe (annualized)':<45}  {historical_sharpe_ratio:.3f}\n"
        " \n Simulated statistics are estimates from historical inputs and model assumptions.\n"
        " \n They are not actual returns, guarantees, or a prediction of future performance.",
        f"\n{rule}\n{'PANIC STRESS SENSITIVITY (COVARIANCE-BASED)'.center(78)}\n{rule}\n{risk_table}",
        f"\n{rule}\nMODEL-SIMULATED LOSS AND DRAWDOWN ESTIMATES\n{rule}\n"
        f" \n {'Pooled ensemble 95% horizon VaR':<45}  {var_95_pct * 100:.2f}% (${var_95_dollar:,.2f})\n"
        f" \n {'Pooled ensemble 99% horizon VaR':<45}  {var_99_pct * 100:.2f}% (${var_99_pct * initial_portfolio_value:,.2f})\n"
        f" \n {'95% expected shortfall (CVaR)':<45}  {cvar_95_pct * 100:.2f}% (${cvar_95_dollar:,.2f})\n"
        f" \n {'95% peak-to-trough drawdown estimate':<45}  {mdd_95_pct * 100:.2f}% (${mdd_95_dollar:,.2f})\n"
        f" \n {'99% peak-to-trough drawdown estimate':<45}  {mdd_99_pct * 100:.2f}% (${mdd_99_dollar:,.2f})\n"
        f" \n {'Drawdown VaR / horizon VaR ratio':<45}  {mdd_ratio:.2f}x\n"
        f" \n {'VaR-threshold terminal equity':<45}  ${var_threshold_terminal_equity:,.2f}\n"
        f" \n {'VaR-threshold liquidity haircut':<45}  ${total_liquidity_haircut:,.2f} ({liquidation_urgency_fraction:.0%} liquidation)\n"
        f" \n {'95% VaR + threshold liquidity adjustment':<45}  ${lvar_95_dollar:,.2f} ({lvar_95_pct * 100:.2f}%)\n"
        f" \n {'95% CVaR + same VaR-threshold add-on':<45}  ${lcvar_95_dollar:,.2f}\n"
        f" \n {'Full-liquidation VaR-threshold haircut':<45}  ${full_liquidation_haircut:,.2f}\n"
        f" \n {'95% VaR + full-liquidation threshold add-on':<45}  ${full_lvar_95_dollar:,.2f} ({full_lvar_95_pct * 100:.2f}%)\n"
        f" \n {'95% CVaR + full-liquidation threshold add-on':<45}  ${full_lcvar_95_dollar:,.2f}\n"
        " \n VaR is the simulated loss threshold exceeded by about 5% / 1% of paths.\n"
        " \n CVaR is average loss among the worst 5% of paths.\n"
        " \n Drawdown is each path's maximum within-horizon peak-to-trough decline.\n"
        " \n Liquidity add-ons are thresholds, not scenario-level simulated L-VaR.",
        f"\n{rule}\nDISTRIBUTION SHAPE -- DYNAMIC MOMENTUM\n{rule}\n"
        f" \n {'Real-time skewness':<45}  {dynamic_skewness:.3f}\n"
        f" \n {'Real-time excess kurtosis':<45}  {dynamic_kurtosis:.3f}",
    ]
    return "\n\n".join(sections) + f"\n\n{separator}\n"


# Your portfolio stocks (matches MCDataFetch-SQL.py):
# Order does NOT matter for syncing with MCDataFetch-SQL.py, but the tickers must match exactly.
# Order for "tickers" DOES matter for syncing with the weights array
# First ticker in the list "tickers" -> First weight in the "weights" array, etc. 
tickers = ["BRK-B", "CI", "EBAY", "VRSN", "XOM", 
    "PPL", "WMT", "MA", "COP", "PM", "LLY",  
]
factor_tickers = ["SPY", "IWM", "IWD", "IWF", "TLT", "GLD", "XLV", "XLE"]
factor_names = [
    "Market (SPY)",
    "Size (IWM-SPY)",
    "Value (IWD-IWF)",
    "Rates (TLT)",
    "Gold (GLD)",
    "Health Care (XLV-SPY)",
    "Energy (XLE-SPY)",
]

# MAKE SURE WEIGHTS SUM TO 1. 
# If not, they will be scaled automatically as percentages relative to the total weight sum.
weights = np.array([
    0.2,0.035,0.035,0.1,0.05,0.15,0.1,0.1,0.04,0.09,0.1
], dtype=np.float64)
weights /= weights.sum()

# SPECS. Feel free to change these for your own analysis, but more scenarios or longer horizons = more computation time.
initial_portfolio_value = 1075600.16
n_scenarios = 100_000
batch_size = 128_000
simulation_horizon_days = 252
n_calibration_runs = 1
panic_observation_target = 120
panic_history_expand = True
panic_history_step_years = 2
panic_history_max_years = 25
# Set either seed to an integer for repeatability of that random stream.
calibration_seed = None
simulation_seed = None
backtest_lookback = 252
max_rebalance_interval = 21
relative_drift_limit = 0.20
minimum_absolute_drift = 0.005
regime_vix_threshold = 25.0
regime_lookback = 21
regime_tail_factor = 1.25


run_simulation_ensemble()
run_liquidity_backtest_stage()
print(render_report())
