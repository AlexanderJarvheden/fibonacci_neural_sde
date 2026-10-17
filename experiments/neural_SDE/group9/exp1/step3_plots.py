from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
FIGURE_FOLDER = f"{EXPERIMENT_FOLDER}/figures"
raw_windows = np.load(f"{EXPERIMENT_FOLDER}/windows_raw.npy")
window_length = raw_windows.shape[1]
fibonacci_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

def moving_average_series(prices, days):
    cumulative = np.cumsum(np.insert(prices, 0, 0))
    averages = np.full(len(prices), np.nan)
    averages[days - 1:] = (cumulative[days:] - cumulative[:-days]) / days
    return averages

def exponential_average_series(prices, days):
    smoothing = 2.0 / (days + 1.0)
    averages = np.empty(len(prices))
    averages[0] = prices[0]
    for day in range(1, len(prices)):
        averages[day] = smoothing * prices[day] + (1 - smoothing) * averages[day - 1]
    return averages

def volatility_series(prices, days):
    volatility = np.full(len(prices), np.nan)
    log_returns = np.diff(np.log(prices))
    for day in range(days, len(prices)):
        volatility[day] = log_returns[day - days:day].std()
    return volatility

def rsi_series(prices, days=14):
    rsi = np.full(len(prices), np.nan)
    daily_changes = np.diff(prices)
    for day in range(days, len(prices)):
        recent_changes = daily_changes[day - days:day]
        average_gain = np.where(recent_changes > 0, recent_changes, 0).mean()
        average_loss = np.where(recent_changes < 0, -recent_changes, 0).mean()
        rsi[day] = 100 - 100 / (1 + average_gain / max(average_loss, 1e-8))
    return rsi

random_generator = np.random.default_rng(3)
chosen_windows = random_generator.choice(raw_windows.shape[0], size=4, replace=False)
days = np.arange(window_length)

figure, axes = plt.subplots(4, 3, figsize=(15, 12))
for row, window_index in enumerate(chosen_windows):
    prices = raw_windows[window_index]
    moving_average = moving_average_series(prices, 20)
    exponential_average = exponential_average_series(prices, 20)

    price_axis = axes[row, 0]
    price_axis.plot(days, prices, color="black", lw=0.9, label="price")
    price_axis.plot(days, moving_average, color="tab:blue", lw=1.1, label="SMA20")
    price_axis.plot(days, exponential_average, color="tab:orange", lw=1.1, label="EMA20")
    for ratio in fibonacci_ratios:
        price_axis.axhline(prices.min() + ratio * (prices.max() - prices.min()), color="grey", lw=0.4, ls=":")
    price_axis.set_ylabel(f"window {window_index}")
    if row == 0:
        price_axis.set_title("price with SMA20 / EMA20\n(dotted = fib levels)")
        price_axis.legend(fontsize=7)

    distance_axis = axes[row, 1]
    distance_axis.plot(days, (prices - moving_average) / moving_average, color="tab:blue", lw=1.0, label="d_MA20")
    distance_axis.plot(days, (prices - exponential_average) / exponential_average, color="tab:orange", lw=1.0, label="d_EMA20")
    distance_axis.axhline(0, color="black", lw=0.5)
    if row == 0:
        distance_axis.set_title("distance to moving average\n(share of price)")
        distance_axis.legend(fontsize=7)

    volatility_axis = axes[row, 2]
    volatility_axis.plot(days, volatility_series(prices, 21), color="tab:green", lw=1.0, label="roll_vol21")
    volatility_axis.set_ylabel("vol", color="tab:green")
    rsi_axis = volatility_axis.twinx()
    rsi_axis.plot(days, rsi_series(prices), color="tab:red", lw=1.0, label="RSI14")
    rsi_axis.axhline(70, color="tab:red", lw=0.4, ls=":")
    rsi_axis.axhline(30, color="tab:red", lw=0.4, ls=":")
    rsi_axis.set_ylabel("RSI", color="tab:red")
    if row == 0:
        volatility_axis.set_title("rolling volatility and RSI14")

for bottom_axis in axes[-1]:
    bottom_axis.set_xlabel("day in 252-day window")

figure.suptitle("Candidate features on four representative windows", fontsize=13)
figure.tight_layout()
figure.savefig(f"{FIGURE_FOLDER}/fig2_features_vs_price.png", dpi=150)
