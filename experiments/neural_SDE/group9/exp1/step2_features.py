from pathlib import Path
import numpy as np
from amgm.utils import common

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
raw_windows = np.load(f"{EXPERIMENT_FOLDER}/windows_raw.npy")
scaled_windows = np.load(f"{EXPERIMENT_FOLDER}/windows.npy")
print("windows:", raw_windows.shape)

last_price = raw_windows[:, -1]

def simple_moving_average(prices, days):
    return prices[:, -days:].mean(axis=1)

def exponential_moving_average(prices, days):
    smoothing = 2.0 / (days + 1.0)
    average = prices[:, 0].copy()
    for day in range(1, prices.shape[1]):
        average = smoothing * prices[:, day] + (1 - smoothing) * average
    return average

# distances and slope as a share of the price
moving_average_now = simple_moving_average(raw_windows, 20)
distance_to_ma20 = (last_price - moving_average_now) / moving_average_now
exponential_average_now = exponential_moving_average(raw_windows, 20)
distance_to_ema20 = (last_price - exponential_average_now) / exponential_average_now
moving_average_5_days_ago = raw_windows[:, -25:-5].mean(axis=1)
moving_average_slope = (moving_average_now - moving_average_5_days_ago) / moving_average_5_days_ago

# typical daily move, same function the model uses
rolling_volatility = common.calculate_logReturnVolatility(raw_windows, window=21).reshape(-1)

# price change in percent over 5, 21 and 63 days
momentum_5_days = (last_price - raw_windows[:, -6]) / raw_windows[:, -6]
momentum_21_days = (last_price - raw_windows[:, -22]) / raw_windows[:, -22]
momentum_63_days = (last_price - raw_windows[:, -64]) / raw_windows[:, -64]

# RSI
daily_changes = np.diff(raw_windows[:, -15:], axis=1)
average_gain = np.where(daily_changes > 0, daily_changes, 0).mean(axis=1)
average_loss = np.where(daily_changes < 0, -daily_changes, 0).mean(axis=1)
relative_strength = average_gain / np.maximum(average_loss, 1e-8)
rsi_14 = (100 - 100 / (1 + relative_strength)) / 100 - 0.5

# position in the window's range is defined on the scaled window
position_in_range = scaled_windows[:, -1]

feature_names = ["pos_in_range", "d_MA20", "d_EMA20", "MA_slope", "roll_vol",
                 "mom5", "mom21", "mom63", "RSI14"]
features = np.stack([position_in_range, distance_to_ma20, distance_to_ema20, moving_average_slope,
                     rolling_volatility, momentum_5_days, momentum_21_days, momentum_63_days, rsi_14], axis=1)

print("\nfeature summary:")
print(f"{'name':<14}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}")
for feature_index, feature_name in enumerate(feature_names):
    values = features[:, feature_index]
    print(f"{feature_name:<14}{values.mean():>10.4f}{values.std():>10.4f}{values.min():>10.4f}{values.max():>10.4f}")
print("any NaN:", np.isnan(features).any())

np.save(f"{EXPERIMENT_FOLDER}/features.npy", features)
np.save(f"{EXPERIMENT_FOLDER}/feature_names.npy", np.array(feature_names))
