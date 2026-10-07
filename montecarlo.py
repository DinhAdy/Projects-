import time
import os
try:
    import yfinance as yf
    import numpy as np
    from arch import arch_model
    from scipy.optimize import minimize
except ImportError:
    import importlib.util
    import subprocess
    import sys
    print("\n  INSTALLING NECESSARY STUFF NOW...\n")
    if importlib.util.find_spec("pip") is None:
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Could not bootstrap pip for this Python interpreter. "
                "Install pip, then run the script again."
            ) from error
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "numpy", "arch", "scipy"])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Could not install the required packages. Check internet access and "
            "package installation permissions, then run the script again."
        ) from error
    import yfinance as yf
    import numpy as np
    from arch import arch_model
    from scipy.optimize import minimize

# --- 1. USER INPUTS & LIVE DATA FETCHING ---
# Paste ANY list of ticker symbols you want!
tickers = [
    "VEEV", "IQV", "TMO", "CRM", "MRK", 
    "GRMN", "DGX", "PSX", "ABBV", "APA", 
    "ABT", "TRV", "MPC"
]

weights = np.array([
    0.1879, 0.1381, 0.0957, 0.0951, 0.0656,
    0.0638, 0.0520, 0.0516, 0.0514, 0.0513,
    0.0506, 0.0501, 0.0478,
], dtype=np.float64)
weights /= weights.sum()

initial_portfolio_value = 908_003.79 # $1,000,000 portfolio
n_scenarios = 14_000_605

print(f"📡 Fetching 5 years of stock and VIX data from Yahoo Finance...")
market_data = yf.download(
    tickers + ["^VIX"],
    period="5y",
    threads=min(8, max(1, ((os.cpu_count() or 2) // 2))),
    progress=False,
)
close = market_data["Close"]
daily_returns = close[tickers].dropna().pct_change().dropna()
vix = close["^VIX"].dropna()
returns = daily_returns.to_numpy(dtype=np.float64)
n_days, n_assets = returns.shape
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

print("✅ VIX MIDAS signals built; fitting Student-t GARCH models...\n")
returns_percent = returns * 100
standardized_residuals = np.empty((n_days, n_assets), dtype=np.float64)
forecast_variances = np.empty(n_assets, dtype=np.float64)
forecast_means = np.empty(n_assets, dtype=np.float64)
student_degrees = np.empty(n_assets, dtype=np.float64)

for asset_index in range(n_assets):
    scaled_returns = returns_percent[:, asset_index] / np.sqrt(daily_midas_scale[:, asset_index])
    fitted_model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
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
    forecast_means[asset_index] = (
        model_forecast.mean.values[-1, 0] * np.sqrt(current_scale) / 100
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
residual_correlation = np.corrcoef(standardized_residuals, rowvar=False)
covariance = residual_correlation * np.outer(forecast_volatility, forecast_volatility)
covariance += np.eye(n_assets) * 1e-12
portfolio_mean = float(weights @ forecast_means)
portfolio_sigma = float(np.sqrt(weights @ covariance @ weights))
historical_portfolio_returns = returns @ weights
ewma_decay = 0.94
ewma_weights = ewma_decay ** np.arange(n_days - 1, -1, -1, dtype=np.float64)
ewma_weights *= 1 - ewma_decay
ewma_weights /= ewma_weights.sum()
historical_mean = ewma_weights @ historical_portfolio_returns
centered_returns = historical_portfolio_returns - historical_mean
historical_variance = ewma_weights @ (centered_returns ** 2)
dynamic_skewness = (ewma_weights @ (centered_returns ** 3)) / historical_variance ** 1.5
dynamic_kurtosis = (ewma_weights @ (centered_returns ** 4)) / historical_variance ** 2 - 3

print("✅ GARCH-MIDAS forecasts and portfolio comparison weights ready!\n")

# --- 2. BATCHED STUDENT-t MONTE CARLO ---
portfolio_returns = np.empty(n_scenarios, dtype=np.float32)
batch_size = 250_000
cf_skewness = np.float32(dynamic_skewness)
cf_kurtosis = np.float32(dynamic_kurtosis)
c1 = cf_skewness / np.float32(6)
c2 = cf_kurtosis / np.float32(24)
c3 = cf_skewness ** np.float32(2) / np.float32(36)
cf_linear = np.float32(1) - np.float32(3) * c2 + np.float32(5) * c3
cf_cubic = c2 - np.float32(2) * c3
cf_normal_scale = np.sqrt(
    cf_linear ** 2
    + np.float32(2) * c1 ** 2
    + np.float32(6) * cf_linear * cf_cubic
    + np.float32(15) * cf_cubic ** 2
)

for batch_start in range(0, n_scenarios, batch_size):
    batch_end = min(batch_start + batch_size, n_scenarios)
    batch_count = batch_end - batch_start
    normal_portfolio = random_generator.standard_normal(batch_count, dtype=np.float32)
    chi_square = random_generator.chisquare(student_df, size=batch_count).astype(np.float32)
    corrected_normal = normal_portfolio.copy()
    np.multiply(corrected_normal, cf_cubic, out=corrected_normal)
    np.add(corrected_normal, c1, out=corrected_normal)
    np.multiply(corrected_normal, normal_portfolio, out=corrected_normal)
    np.add(corrected_normal, cf_linear, out=corrected_normal)
    np.multiply(corrected_normal, normal_portfolio, out=corrected_normal)
    np.subtract(corrected_normal, c1, out=corrected_normal)
    np.divide(corrected_normal, cf_normal_scale, out=corrected_normal)
    np.divide(np.float32(student_df - 2), chi_square, out=chi_square)
    np.sqrt(chi_square, out=chi_square)
    np.multiply(corrected_normal, chi_square, out=corrected_normal)
    np.multiply(corrected_normal, portfolio_sigma, out=corrected_normal)
    np.add(corrected_normal, portfolio_mean, out=corrected_normal)
    portfolio_returns[batch_start:batch_end] = corrected_normal

computation_time = (time.perf_counter() - start_time) * 1000

mean_ret = np.mean(portfolio_returns, dtype=np.float64)
volatility = np.std(portfolio_returns, dtype=np.float64)
ann_vol = volatility * np.sqrt(252)
expected_pnl = mean_ret * initial_portfolio_value
win_rate = np.mean(portfolio_returns > 0) * 100
var_95_pct, var_99_pct = -np.percentile(portfolio_returns, [5, 1])
var_95_dollar = var_95_pct * initial_portfolio_value
cvar_95_pct = -np.mean(portfolio_returns[portfolio_returns <= -var_95_pct])
cvar_95_dollar = cvar_95_pct * initial_portfolio_value
sharpe_ratio = (mean_ret - (0.04 / 252)) / volatility
im_better_than_strange = np.ceil(win_rate / 100 * n_scenarios) - 1
portfolio_weight_text = ", ".join(
    f"{ticker} {weight:.2%}" for ticker, weight in zip(tickers, weights)
)
optimized_weight_text = ", ".join(
    f"{ticker} {weight:.2%}" for ticker, weight in zip(tickers, optimized_weights)
)

backtest_lookback = 252
backtest_days = n_days - backtest_lookback
if backtest_days <= 0:
    raise ValueError("At least 253 daily return observations are required for backtesting.")

broker_fee_rate = 0.00005
slippage_rate = 0.0002
sec_sell_fee_rate = 0.0000278
total_transaction_friction_rate = broker_fee_rate + slippage_rate + sec_sell_fee_rate
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
previous_weights = np.full(n_assets, 1 / n_assets)
total_transaction_costs = 0.0

for backtest_index, return_index in enumerate(range(backtest_lookback, n_days)):
    lookback_returns = returns[return_index - backtest_lookback:return_index]
    local_covariance = np.cov(lookback_returns, rowvar=False)
    covariance_scale = np.trace(local_covariance) / n_assets
    local_covariance += np.eye(n_assets) * max(covariance_scale * 1e-8, 1e-12)
    if trailing_vix[return_index] > regime_vix_threshold:
        diagonal = np.diag_indices_from(local_covariance)
        local_covariance[diagonal] *= regime_tail_factor
    daily_weights = minimum_variance_weights(local_covariance, previous_weights)
    delta_weights = np.abs(daily_weights - previous_weights)
    turnover = delta_weights.sum()
    sell_turnover = np.maximum(previous_weights - daily_weights, 0).sum()
    current_equity = backtest_equity[backtest_index]
    transaction_cost = current_equity * (
        turnover * (total_transaction_friction_rate - sec_sell_fee_rate)
        + sell_turnover * sec_sell_fee_rate
    )
    total_transaction_costs += transaction_cost
    backtest_returns[backtest_index] = (
        daily_weights @ returns[return_index] - transaction_cost / current_equity
    )
    backtest_equity[backtest_index + 1] = current_equity * (
        1 + backtest_returns[backtest_index]
    )
    previous_weights = daily_weights

backtest_net_return = (backtest_equity[-1] / initial_portfolio_value - 1) * 100
backtest_drawdowns = backtest_equity / np.maximum.accumulate(backtest_equity) - 1
backtest_max_drawdown = np.min(backtest_drawdowns) * 100
backtest_volatility = np.std(backtest_returns, ddof=1)
backtest_sharpe = (
    (np.mean(backtest_returns) - 0.04 / 252) / backtest_volatility * np.sqrt(252)
    if backtest_volatility > 0
    else 0.0
)

print("=" * 65)
print("BACKTEST PERFORMANCE SCORECARD")
print("-" * 65)
print(f"Strategy                    : Long-only minimum variance")
print(f"Lookback / test days        : {backtest_lookback} / {backtest_days}")
print(
    f"Friction Costs / Net Return: ${total_transaction_costs:,.2f} / "
    f"{backtest_net_return:+.2f}%"
)
print(f"Maximum Peak-to-Trough DD   : {backtest_max_drawdown:.2f}%")
print(f"Historical Sharpe (annual)  : {backtest_sharpe:.3f}")
print("Daily close-to-close; includes broker fee, slippage, and sell-side SEC fee.")

print("=" * 65)
print(f"🚀 GARCH-MIDAS QUANT DASHBOARD ({n_scenarios:,} SCENARIOS)")
print("=" * 65)
print(f"🎯 Portfolio Assets        : {tickers}")
print(f"⚖️ Your Portfolio Weights  : {portfolio_weight_text}")
print(f"📊 Min-Variance Comparison : {optimized_weight_text}")
print(f"📐 Student-t Degrees       : {student_df:.2f}")
print(f"⚡ Model + Simulation Time : {computation_time:.2f}ms")
print(f"💰 Initial Value           : ${initial_portfolio_value:,.2f}")
print("-" * 65)
print("📈 EXPECTED OUTCOMES (1-DAY HORIZON)")
print(f"  • Expected Portfolio Return : {mean_ret * 100:+.3f}%")
print(f"  • Expected P&L ($)          : ${expected_pnl:+,.2f}")
print(f"  • Annualized Volatility     : {ann_vol * 100:.2f}%")
print(f"  • Win Rate (Profitable)     : {win_rate:.2f}%")
print("\n🛡️ RISK METRICS & TAIL EXPOSURE")
print(f"  • 95% 1-Day VaR (Percentage): {var_95_pct * 100:.2f}% max loss")
print(f"  • 95% 1-Day VaR (Dollar)    : ${var_95_dollar:,.2f}")
print(f"  • 99% 1-Day VaR (Dollar)    : ${var_99_pct * initial_portfolio_value:,.2f}")
print(f"  • 95% Expected Shortfall    : ${cvar_95_dollar:,.2f} average crash loss")
print(f"  • Sharpe Ratio              : {sharpe_ratio:.3f}")
print("\n📊 DISTRIBUTION SHAPE (DYNAMIC MOMENTUM)")
print(f"  • Real-time Skewness        : {dynamic_skewness:.3f}")
print(f"  • Real-time Excess Kurtosis : {dynamic_kurtosis:.3f}")
if im_better_than_strange > 1:
    print(f"  • DR STRANGE SUCKS! I WIN   : {im_better_than_strange:,.0f} MORE TIMES THAN DR STRANGE!")
    print("  ----> I'M SORRY, EARTH IS CLOSED TODAY! YOU BETTER PACK IT UP AND GET OUTTA HERE!")
else:
    print("  • 0 WINS SON I'M CRYING IF YOU GENUINELY WIN LESS TIMES THAN DR STRANGE PACK IT UP BRO ITS OVER")
    print("  ----> HEAR ME, AND REJOICE! YOU'RE ABOUT TO DIE AT THE HANDS OF THE CHILDRENS OF THANOS")
print("=" * 65)