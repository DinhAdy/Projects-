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

License & Copyright:
    Copyright (c) 2026 Andy Dinh. All Rights Reserved.
    PROPRIETARY AND CONFIDENTIAL. Strictly NO permission is granted for unauthorized 
    distribution, duplication, modification, or public cloning of this source logic.
=======================================================================================
"""



import time
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import yfinance as yf
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
    import yfinance as yf
    import numpy as np
    from arch import arch_model
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf

# --- 1. USER INPUTS & LIVE DATA FETCHING ---
# Paste ANY list of ticker symbols you want!
tickers = [
    "BRK-B", "CI", "EBAY", "VRSN", "XOM", 
    "PPL", "WMT", "MA", "COP", "PM"
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

weights = np.array([
    0.12,  
0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.09,0.09,  
], dtype=np.float64)
weights /= weights.sum()

initial_portfolio_value = 1045600.16
n_scenarios = 14_000_605
    
print("\nFetching 5 years of stock, factor, rate, and VIX data from Yahoo Finance...", end="\n\n")
market_data = yf.download(
    list(dict.fromkeys(tickers + factor_tickers + ["^VIX", "^IRX"])),
    period="5y",
    auto_adjust=True,
    threads=min(8, max(1, ((os.cpu_count() or 2) // 2))),
    progress=False,
)
close = market_data["Close"]
volume = market_data["Volume"]
daily_returns = close[tickers].dropna().pct_change().dropna()
vix = close["^VIX"].dropna()
irx = close["^IRX"].dropna() if "^IRX" in close else None
live_risk_free_rate = (
    float(irx.iloc[-1] / 100.0)
    if irx is not None and len(irx) > 0 and np.isfinite(irx.iloc[-1])
    else 0.04
)
returns = daily_returns.to_numpy(dtype=np.float64)
n_days, n_assets = returns.shape
factor_asset_returns = close[factor_tickers].pct_change().reindex(daily_returns.index)
factor_returns = np.column_stack(
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
aligned_close = close[tickers].reindex(daily_returns.index)
aligned_volume = volume[tickers].reindex(daily_returns.index)
dollar_volume = aligned_close * aligned_volume
lagged_average_dollar_volume = (
    dollar_volume.rolling(21, min_periods=21).mean().shift(1)
    .to_numpy(dtype=np.float64)
)
lagged_return_volatility = (
    daily_returns.rolling(21, min_periods=21).std().shift(1)
    .to_numpy(dtype=np.float64)
)
start_time = time.perf_counter()

monthly_vix = vix.resample("ME").mean().dropna()
log_vix_variance = 2 * np.log(monthly_vix.to_numpy(dtype=np.float64))
midas_lags = min(24, len(log_vix_variance) - 1)
lag_positions = np.arange(1, midas_lags + 1, dtype=np.float64) / (midas_lags + 1)
midas_weights = (1 - lag_positions) ** 4
midas_weights /= midas_weights.sum()
long_run_macro = log_vix_variance.mean()
midas_signal = np.empty(len(monthly_vix), dtype=np.float64)

for month_index in range(len(monthly_vix)):
    available_lags = min(month_index, midas_lags)
    if available_lags:
        lagged_values = log_vix_variance[month_index - available_lags:month_index][::-1]
        lag_weights = midas_weights[:available_lags]
        midas_signal[month_index] = np.average(lagged_values, weights=lag_weights)
    else:
        midas_signal[month_index] = long_run_macro

macro_by_month = dict(zip(monthly_vix.index.to_period("M"), midas_signal))
monthly_variance = daily_returns.groupby(daily_returns.index.to_period("M")).agg(
    lambda values: np.mean(np.square(values.to_numpy()))
)
monthly_macro = np.array(
    [macro_by_month.get(month, long_run_macro) for month in monthly_variance.index],
    dtype=np.float64,
)
monthly_variance_values = monthly_variance.to_numpy(dtype=np.float64)
valid_months = (monthly_variance_values > 0).all(axis=1) & np.isfinite(monthly_macro)
regression_x = monthly_macro[valid_months] - long_run_macro
regression_design = np.column_stack((np.ones(regression_x.size), regression_x))
midas_parameters = np.linalg.lstsq(
    regression_design, np.log(monthly_variance_values[valid_months]), rcond=None
)[0]
midas_sensitivity = midas_parameters[1]
daily_macro = np.array(
    [macro_by_month.get(month, long_run_macro) for month in daily_returns.index.to_period("M")],
    dtype=np.float64,
)
daily_midas_scale = np.exp(
    np.clip((daily_macro - long_run_macro)[:, np.newaxis] * midas_sensitivity, -3, 3)
)
daily_midas_scale /= daily_midas_scale.mean(axis=0)

print("VIX MIDAS signals built; fitting GARCH models...", end="\n\n")
returns_percent = returns * 100
standardized_residuals = np.empty((n_days, n_assets), dtype=np.float64)
forecast_variances = np.empty(n_assets, dtype=np.float64)
student_degrees = np.empty(n_assets, dtype=np.float64)

for asset_index in range(n_assets):
    scaled_returns = returns_percent[:, asset_index] / np.sqrt(daily_midas_scale[:, asset_index])
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
        np.asarray(fitted_model.std_resid), nan=0.0
    )
    forecast_variances[asset_index] = (
        model_forecast.variance.values[-1, 0] * current_scale / 10_000
    )
    student_degrees[asset_index] = fitted_model.params["nu"]

student_df = max(4.1, float(np.median(student_degrees)))
forecast_volatility = np.sqrt(forecast_variances)
random_generator = np.random.default_rng()

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

weight_samples = []
for _ in range(20):
    sample_indices = random_generator.integers(0, n_days, size=n_days)
    sampled_residuals = standardized_residuals[sample_indices]
    sample_correlation = np.corrcoef(sampled_residuals, rowvar=False)
    sample_covariance = sample_correlation * np.outer(forecast_volatility, forecast_volatility)
    weight_samples.append(minimum_variance_weights(sample_covariance))

optimized_weights = np.mean(weight_samples, axis=0)
optimized_weights /= optimized_weights.sum()
# --- PRE-COMPUTED GARCH-MIDAS CONDITIONAL PORTFOLIO VARIANCE ---
lw_cov = LedoitWolf().fit(standardized_residuals).covariance_
lw_diag_inv = 1.0 / np.sqrt(np.maximum(np.diag(lw_cov), 1e-12))
cond_correlation = lw_cov * np.outer(lw_diag_inv, lw_diag_inv)
conditional_covariance = cond_correlation * np.outer(forecast_volatility, forecast_volatility)
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
portfolio_garch_forecast = portfolio_garch_fit.forecast(horizon=1, reindex=False)
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
    raise ValueError("Portfolio GJR-GARCH fit returned invalid volatility parameters.")
unconditional_portfolio_vol = float(np.std(historical_portfolio_returns))
fhs_vol_scale = float(
    np.clip(
        portfolio_cond_volatility / max(unconditional_portfolio_vol, 1e-6),
        0.6,
        2.5,
    )
)
ewma_decay = 0.94
ewma_weights = ewma_decay ** np.arange(n_days - 1, -1, -1, dtype=np.float64)
ewma_weights *= 1 - ewma_decay
ewma_weights /= ewma_weights.sum()
historical_mean = ewma_weights @ historical_portfolio_returns
centered_returns = historical_portfolio_returns - historical_mean
historical_variance = ewma_weights @ (centered_returns ** 2)
dynamic_skewness = (ewma_weights @ (centered_returns ** 3)) / historical_variance ** 1.5
dynamic_kurtosis = (ewma_weights @ (centered_returns ** 4)) / historical_variance ** 2 - 3

regime_returns = historical_portfolio_returns
aligned_vix = vix.reindex(daily_returns.index).ffill()
lagged_vix = aligned_vix.shift(1).to_numpy(dtype=np.float64)
lagged_vix_velocity = (
    aligned_vix.pct_change(5).shift(1).to_numpy(dtype=np.float64)
)
valid_vix = np.isfinite(lagged_vix) & np.isfinite(lagged_vix_velocity)
if not valid_vix.any() or not np.isfinite(lagged_vix[-1] + lagged_vix_velocity[-1]):
    raise ValueError("Insufficient aligned VIX history to classify simulation regimes.")
vix_level_threshold = np.quantile(lagged_vix[valid_vix], 0.75)
vix_panic_level_threshold = np.quantile(lagged_vix[valid_vix], 0.90)
positive_vix_velocity = lagged_vix_velocity[valid_vix & (lagged_vix_velocity > 0)]
if positive_vix_velocity.size == 0:
    raise ValueError("Insufficient positive VIX momentum to calibrate the panic regime.")
vix_panic_velocity_threshold = np.quantile(positive_vix_velocity, 0.75)
historical_regimes = np.zeros(n_days, dtype=np.int8)
historical_regimes[valid_vix & (lagged_vix >= vix_level_threshold)] = 1
historical_regimes[
    valid_vix
    & (lagged_vix >= vix_panic_level_threshold)
    & (lagged_vix_velocity >= vix_panic_velocity_threshold)
] = 2
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
regime_transition = transition_counts / transition_counts.sum(axis=1, keepdims=True)
current_regime = historical_regimes[-1]
current_regime_name = ("Calm", "High-volatility", "Panic")[current_regime]
regime_transition_cumulative = np.cumsum(regime_transition, axis=1).astype(np.float32)
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
    valid_factor_rows & (historical_regimes == regime) for regime in range(3)
]
regime_names = ("Calm", "High-volatility", "Panic")
regime_shrinkage_prior = 100
regime_factor_coefficients = np.empty((3, factor_design.shape[1]), dtype=np.float64)
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
        regime_fit = np.linalg.lstsq(regime_design, regime_returns_sample, rcond=None)[0]
        shrinkage = regime_returns_sample.size / (
            regime_returns_sample.size + regime_shrinkage_prior
        )
        regime_factor_coefficients[regime] = factor_coefficients + shrinkage * (
            regime_fit - factor_coefficients
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

# --- REGIME-AWARE CONDITIONAL PORTFOLIO VARIANCE ---
# Extract actual residuals from the active market regime.
active_regime_residuals = standardized_residuals[regime_sample_masks[current_regime]]
lw_active_cov = LedoitWolf().fit(active_regime_residuals).covariance_
lw_active_diag_inv = 1.0 / np.sqrt(np.maximum(np.diag(lw_active_cov), 1e-12))
regime_correlation = lw_active_cov * np.outer(lw_active_diag_inv, lw_active_diag_inv)

# Re-compute forward portfolio variance using the regime's empirical correlation.
conditional_covariance = regime_correlation * np.outer(forecast_volatility, forecast_volatility)
portfolio_cond_variance = float(weights @ conditional_covariance @ weights)
# --- Panic-stress covariance for explicitly labeled risk attribution ---
panic_asset_residuals = standardized_residuals[regime_sample_masks[2]]
if panic_asset_residuals.shape[0] < 2:
    raise ValueError("At least two panic-regime observations are needed for stress covariance.")
lw_panic_cov = LedoitWolf().fit(panic_asset_residuals).covariance_
lw_panic_diag_inv = 1.0 / np.sqrt(np.maximum(np.diag(lw_panic_cov), 1e-12))
panic_correlation = lw_panic_cov * np.outer(lw_panic_diag_inv, lw_panic_diag_inv)
panic_conditional_covariance = (
    panic_correlation * np.outer(forecast_volatility, forecast_volatility)
)
panic_portfolio_cov = panic_conditional_covariance @ weights
panic_portfolio_variance = float(weights @ panic_portfolio_cov)
panic_asset_betas = panic_portfolio_cov / max(panic_portfolio_variance, 1e-12)

# --- Correlated semi-parametric fat-tail augmentation for panic regime ---
panic_factors = regime_factors[2]
panic_resids = regime_residuals[2]
n_panic_synth = max(60, int(len(panic_resids) * 0.5))
panic_sub_idx = random_generator.integers(0, len(panic_resids), size=n_panic_synth)
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
    (n_panic_synth, factor_returns.shape[1]), dtype=np.float32
)
factor_chi_square = random_generator.chisquare(
    student_df, size=n_panic_synth
).astype(np.float32)
t_factor_shocks = (factor_normal_shocks @ panic_factor_cholesky.T) * (
    np.sqrt((student_df - 2) / factor_chi_square)[:, np.newaxis]
).astype(np.float32)
synth_panic_factors = panic_factors[panic_sub_idx] + (
    t_factor_shocks * np.float32(0.5)
)

regime_factors[2] = np.concatenate([panic_factors, synth_panic_factors], axis=0)
regime_residuals[2] = np.concatenate([panic_resids, synth_panic_resids])

regime_sample_counts = np.array([len(f) for f in regime_factors], dtype=np.int32)
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

# --- Filtered Historical Simulation (FHS): Scale Residuals by Forward GARCH-MIDAS Volatility ---
regime_return_pools = []
for regime in range(3):
    factor_prediction = (
        regime_factors[regime] @ regime_factor_coefficients[regime, 1:].astype(np.float32)
        + np.float32(regime_factor_coefficients[regime, 0])
    )
    regime_return_pools.append(
        factor_prediction + regime_residuals[regime] * np.float32(fhs_vol_scale)
    )
return_sample_pool = np.concatenate(regime_return_pools).astype(np.float32)
regime_return_means = np.array(
    [np.mean(pool, dtype=np.float64) for pool in regime_return_pools],
    dtype=np.float32,
)
regime_cascade_high_gross = np.empty(3, dtype=np.float32)
regime_cascade_panic_gross = np.empty(3, dtype=np.float32)
for regime, pool in enumerate(regime_return_pools):
    regime_cascade_high_gross[regime] = (
        1 + np.quantile(pool, 0.10)
    )
    regime_cascade_panic_gross[regime] = (
        1 + np.quantile(pool, 0.02)
    )

print("GARCH residuals and regime-conditioned factor model ready!", end="\n\n")

# --- 2. BATCHED REGIME-CONDITIONED FACTOR MONTE CARLO ---
portfolio_returns = np.empty(n_scenarios, dtype=np.float32)
portfolio_max_drawdowns = np.empty(n_scenarios, dtype=np.float32)
batch_size = 500_000
simulation_horizon_days = 10
max_batch_count = min(batch_size, n_scenarios)
drawdown_min_buffer = np.empty(max_batch_count, dtype=np.float32)
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
garch_alpha_ratio = np.float32(
    portfolio_garch_alpha / portfolio_garch_base_variance
)
garch_gamma_ratio = np.float32(
    portfolio_garch_gamma / portfolio_garch_base_variance
)
garch_base_variance = np.float32(portfolio_garch_base_variance)
garch_beta = np.float32(portfolio_garch_beta)
path_variance_ratio_min = np.float32(0.25)
path_variance_ratio_max = np.float32(9.0)

for batch_start in range(0, n_scenarios, batch_size):
    batch_end = min(batch_start + batch_size, n_scenarios)
    batch_count = batch_end - batch_start
    batch_returns = portfolio_returns[batch_start:batch_end]
    batch_returns.fill(1)
    batch_min_equity = drawdown_min_buffer[:batch_count]
    batch_min_equity.fill(1)
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
        np.sqrt(path_variance_ratio, out=path_volatility)
        np.multiply(corrected_normal, path_volatility, out=corrected_normal)

        np.multiply(corrected_normal, np.float32(100), out=volatility_update)
        np.square(volatility_update, out=volatility_update)
        np.multiply(
            volatility_update,
            garch_alpha_ratio,
            out=volatility_update,
        )
        np.multiply(
            path_variance_ratio,
            garch_beta,
            out=path_variance_ratio,
        )
        np.add(path_variance_ratio, garch_omega_ratio, out=path_variance_ratio)
        np.less(corrected_normal, np.float32(0), out=garch_negative_mask)
        np.multiply(
            corrected_normal,
            np.float32(100),
            out=garch_gamma_update,
        )
        np.square(garch_gamma_update, out=garch_gamma_update)
        np.multiply(
            garch_gamma_update,
            garch_gamma_ratio,
            out=garch_gamma_update,
        )
        np.add(
            volatility_update,
            garch_gamma_update,
            out=volatility_update,
            where=garch_negative_mask,
        )
        np.divide(
            volatility_update,
            garch_base_variance,
            out=volatility_update,
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
        np.minimum(batch_min_equity, batch_returns, out=batch_min_equity)
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
    portfolio_max_drawdowns[batch_start:batch_end] = batch_min_equity - np.float32(1)

computation_time = (time.perf_counter() - start_time) * 1000

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
sharpe_ratio = (
    (mean_ret - period_target) / volatility
    * np.sqrt(252 / simulation_horizon_days)
    if volatility > 0
    else 0.0
)
im_better_than_strange = np.ceil(win_rate / 100 * n_scenarios) - 1
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
backtest_lookback = 252
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
regime_vix_threshold = 25.0
regime_lookback = 21
regime_tail_factor = 1.25
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

backtest_returns = np.empty(backtest_days, dtype=np.float64)
backtest_equity = np.empty(backtest_days + 1, dtype=np.float64)
backtest_equity[0] = initial_portfolio_value
held_weights = np.full(n_assets, 1 / n_assets)
rebalance_target_weights = None
days_since_rebalance = 21
max_rebalance_interval = 21
relative_drift_limit = 0.20
minimum_absolute_drift = 0.005
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

backtest_net_return = (backtest_equity[-1] / initial_portfolio_value - 1) * 100
backtest_drawdowns = backtest_equity / np.maximum.accumulate(backtest_equity) - 1
backtest_max_drawdown = np.min(backtest_drawdowns) * 100
backtest_daily_target = live_risk_free_rate / 252
backtest_downside_deviation = np.sqrt(
    np.mean(np.minimum(backtest_returns - backtest_daily_target, 0) ** 2)
)
backtest_sortino = (
    (np.mean(backtest_returns) - backtest_daily_target)
    / backtest_downside_deviation * np.sqrt(252)
    if backtest_downside_deviation > 0
    else 0.0
)

print("=" * 78, end="\n\n")
print("BACKTEST PERFORMANCE SCORECARD".center(78), end="\n\n")
print("-" * 78, end="\n\n")
print(f"  {'Strategy':<34}  Long-only minimum variance", end="\n\n")
print(f"  {'Lookback / test days':<34}  {backtest_lookback} / {backtest_days}", end="\n\n")
print(
    f"  {'Rebalance events / VIX overrides':<34}  "
    f"{rebalance_count} / {stress_rebalance_count}",
    end="\n\n",
)
print(
    f"  {'Rebalance rule':<34}  20% relative / 0.5% floor / 21-day max / VIX stress",
    end="\n\n",
)
print(
    f"  {'Friction costs paid':<34}  ${total_transaction_costs:,.2f}",
    end="\n\n",
)
print(f"  {'Net realized return':<34}  {backtest_net_return:+.2f}%", end="\n\n")
print(f"  {'Maximum peak-to-trough drawdown':<34}  {backtest_max_drawdown:.2f}%", end="\n\n")
print(f"  {'Historical Sortino (annualized)':<34}  {backtest_sortino:.3f}", end="\n\n")
print("  Daily close-to-close; includes broker fee, slippage, and sell-side SEC fee.", end="\n\n")

print("=" * 78, end="\n\n")
print(
    f"GARCH-MIDAS QUANT DASHBOARD — {n_scenarios:,} SCENARIOS "
    f"({simulation_horizon_days}-DAY HORIZON)".center(78),
    end="\n\n",
)
print("=" * 78, end="\n\n")
print("  Your portfolio weights", end="\n\n")
np.savetxt(sys.stdout, portfolio_weight_table, fmt=('   %-10s ', '%8s'), delimiter=' ')
print(end="\n\n")
print("  Min-variance comparison", end="\n\n")
np.savetxt(sys.stdout, optimized_weight_table, fmt=('   %-10s ', '%8s'), delimiter=' ')
print(end="\n\n")
print(f"  {'Median fitted GARCH-t degrees of freedom':<34}  {student_df:.2f}", end="\n\n")
print(f"  {'Starting volatility regime':<34}  {current_regime_name}", end="\n\n")
print(
    f"  {'Historical factor observations':<34}  "
    f"Calm {regime_raw_counts[0]:,} / High {regime_raw_counts[1]:,} / "
    f"Panic {regime_raw_counts[2]:,}",
    end="\n\n",
)
print(
    f"  {'Bootstrap pool observations':<34}  "
    f"Calm {regime_sample_counts[0]:,} / High {regime_sample_counts[1]:,} / "
    f"Panic {regime_sample_counts[2]:,}",
    end="\n\n",
)
print("  Regime-conditioned factor loadings", end="\n\n")
print(f"   {'Regime':<18}  {'Factor':<24}  {'Loading':>10}", end="\n\n")
np.savetxt(
    sys.stdout,
    factor_model_table,
    fmt=("   %-18s", "%-24s", "%10s"),
    delimiter="  ",
)
print(end="\n\n")
print(f"  {'Synthetic panic observations':<34}  {n_panic_synth:,}", end="\n\n")
print(f"  {'Factor rows excluded (missing data)':<34}  {excluded_factor_rows}", end="\n\n")
print(f"  {'Risk-free rate (13-week T-Bill)':<34}  {live_risk_free_rate * 100:.2f}%", end="\n\n")
print(f"  {'GARCH FHS volatility multiplier':<34}  {fhs_vol_scale:.3f}x", end="\n\n")
print(
    f"  {'Path volatility memory':<34}  "
    f"GJR-GARCH; variance ratio capped 0.25x-9.00x",
    end="\n\n",
)
print(
    f"  {'Cascade cutoffs (current regime)':<34}  "
    f"{(regime_cascade_high_gross[current_regime] - 1) * 100:.2f}% / "
    f"{(regime_cascade_panic_gross[current_regime] - 1) * 100:.2f}%",
    end="\n\n",
)
print(f"  {'Model + simulation time':<34}  {computation_time:.2f} ms", end="\n\n")
print(f"  {'Initial portfolio value':<34}  ${initial_portfolio_value:,.2f}", end="\n\n")

print("-" * 78, end="\n\n")
print(f"EXPECTED OUTCOMES — {simulation_horizon_days}-DAY HORIZON", end="\n\n")
print("-" * 78, end="\n\n")
print(f"  {'Expected portfolio return':<34}  {mean_ret * 100:+.3f}%", end="\n\n")
print(f"  {'Expected P&L':<34}  ${expected_pnl:+,.2f}", end="\n\n")
print(f"  {'Annualized volatility':<34}  {ann_vol * 100:.2f}%", end="\n\n")
print(f"  {'Win rate (profitable)':<34}  {win_rate:.2f}%", end="\n\n")
print("-" * 78, end="\n\n")
print("PANIC STRESS SENSITIVITY (COVARIANCE-BASED)".center(78), end="\n\n")
print("-" * 78, end="\n\n")
print(f"  {'Ticker':<7} {'Weight':>8} {'Panic Beta':>14} {'Stress Share':>13} {'VaR Sensitivity ($)':>19} {'CVaR Sensitivity ($)':>20}   {'Role'}")
print(f"  {'-' * 76}")
for idx in risk_ranking:
    t = tickers[idx]
    w_pct = weights[idx] * 100.0
    beta = panic_asset_betas[idx]
    rc_pct = component_risk_pct[idx]
    c_var = component_var_dollar[idx]
    c_cvar = component_cvar_dollar[idx]
    role = "Tail Aggressor" if beta > 1.15 else ("Diversifier" if beta < 0.85 else "Neutral")
    print(f"  {t:<7} {w_pct:>7.2f}% {beta:>14.3f} {rc_pct:>12.2f}% ${c_var:>13,.2f} ${c_cvar:>13,.2f}   {role}")
print(f"  {'-' * 76}")
print(f"  {'TOTAL':<7} {100.0:>7.2f}% {1.0:>14.3f} {100.0:>12.2f}% ${var_95_dollar:>13,.2f} ${cvar_95_dollar:>13,.2f}")
print("-" * 78, end="\n\n")
print("RISK METRICS & TAIL EXPOSURE", end="\n\n")
print("-" * 78, end="\n\n")
print(f"  {'95% horizon VaR (percentage)':<34}  {var_95_pct * 100:.2f}% max loss", end="\n\n")
print(f"  {'95% horizon VaR (dollar)':<34}  ${var_95_dollar:,.2f}", end="\n\n")
print(f"  {'95% Intra-Horizon Max Drawdown VaR':<34}  ${mdd_95_dollar:,.2f} ({mdd_95_pct * 100:.2f}% peak-to-trough)", end="\n\n")
print(f"  {'Intra-Horizon Crash Depth Ratio':<34}  {mdd_ratio:.2f}x (mid-week trough vs Day 10)", end="\n\n")
print(
    f"  {'99% horizon VaR (dollar)':<34}  "
    f"${var_99_pct * initial_portfolio_value:,.2f}",
    end="\n\n",
)
print(f"  {'99% Intra-Horizon Max Drawdown VaR':<34}  ${mdd_99_dollar:,.2f} ({mdd_99_pct * 100:.2f}%)", end="\n\n")
print(f"  {'95% expected shortfall':<34}  ${cvar_95_dollar:,.2f} average crash loss", end="\n\n")
print(
    f"  {'VaR-threshold terminal equity':<34}  "
    f"${var_threshold_terminal_equity:,.2f}",
    end="\n\n",
)
print(
    f"  {'VaR-threshold liquidity haircut':<34}  "
    f"${total_liquidity_haircut:,.2f} ({liquidation_urgency_fraction:.0%} liquidation)",
    end="\n\n",
)
print(
    f"  {'95% VaR + threshold liquidity adjustment':<34}  "
    f"${lvar_95_dollar:,.2f} ({lvar_95_pct * 100:.2f}%)",
    end="\n\n",
)
print(
    f"  {'95% CVaR + same VaR-threshold add-on':<34}  "
    f"${lcvar_95_dollar:,.2f}",
    end="\n\n",
)
print(
    f"  {'Full-liquidation VaR-threshold haircut':<34}  "
    f"${full_liquidation_haircut:,.2f}",
    end="\n\n",
)
print(
    f"  {'95% VaR + full-liquidation threshold add-on':<34}  "
    f"${full_lvar_95_dollar:,.2f} ({full_lvar_95_pct * 100:.2f}%)",
    end="\n\n",
)
print(
    f"  {'95% CVaR + full-liquidation threshold add-on':<34}  "
    f"${full_lcvar_95_dollar:,.2f}",
    end="\n\n",
)
print(
    "  Liquidity add-ons are threshold estimates, not scenario-level simulated L-VaR.",
    end="\n\n",
)
print(f"  {'Monte Carlo Sharpe ratio':<34}  {sharpe_ratio:.3f}", end="\n\n")

print("-" * 78, end="\n\n")
print("DISTRIBUTION SHAPE — DYNAMIC MOMENTUM", end="\n\n")
print("-" * 78, end="\n\n")
print(f"  {'Real-time skewness':<34}  {dynamic_skewness:.3f}", end="\n\n")
print(f"  {'Real-time excess kurtosis':<34}  {dynamic_kurtosis:.3f}", end="\n\n")
if im_better_than_strange > 1:
    print(f"  DR STRANGE SUCKS! I WIN {im_better_than_strange:,.0f} MORE TIMES!", end="\n\n")
    print("  I'M SORRY, EARTH IS CLOSED TODAY! YOU BETTER PACK IT UP AND GET OUTTA HERE!", end="\n\n")
else:
    print("  0 WINS, I'M CRYING. PACK IT UP BRO IT'S OVER.", end="\n\n")
    print("  REJOICE! THE CHILDREN OF THANOS ARE COMING.", end="\n\n")
print("=" * 78, end="\n\n")