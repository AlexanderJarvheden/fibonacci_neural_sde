from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, kurtosis

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
SAMPLES_FOLDER = EXPERIMENT_FOLDER.parent / "exp3" / "samples"
FIGURE_PATH = EXPERIMENT_FOLDER / "figures" / "fig7_trajectory_statistics.png"
MINIMUM_DAYS = 60
CONFIGS = ["A", "B", "C"]
SEEDS = ["s1", "s2", "s3"]
STATISTICS = ["daily_returns", "volatility", "max_drawdown"]
PLOT_COLOURS = {"Real": "black", "A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
PLOT_TITLES = {"daily_returns": "daily log returns (log scale)",
               "volatility": "volatility per path",
               "max_drawdown": "max drawdown per path"}


def statistics_of_paths(paths):
    daily_returns, volatility, max_drawdown = [], [], []
    for prices in paths:
        log_returns = np.diff(np.log(prices))
        daily_returns.append(log_returns)
        volatility.append(log_returns.std())
        max_drawdown.append(np.max(1 - prices / np.maximum.accumulate(prices)))
    return {"daily_returns": np.concatenate(daily_returns),
            "volatility": np.array(volatility),
            "max_drawdown": np.array(max_drawdown)}


reference_samples = np.load(SAMPLES_FOLDER / "Exp2_A_s1.npz")
real_lengths = reference_samples["original_price_len"]
real_prices = reference_samples["original_price"]
scenarios_to_use = np.where(real_lengths >= MINIMUM_DAYS)[0]
real_statistics = statistics_of_paths([real_prices[scenario, :real_lengths[scenario]] for scenario in scenarios_to_use])

statistics_per_run, distances_per_run = {}, {}
for config in CONFIGS:
    for seed in SEEDS:
        samples = np.load(SAMPLES_FOLDER / f"Exp2_{config}_{seed}.npz")
        synthetic_paths = samples["synthetic_paths"]
        scenario_of_path = samples["base_sample_id"]
        path_is_valid = samples["is_valid"]
        valid_paths = []
        for path_index, scenario in enumerate(scenario_of_path):
            prices = synthetic_paths[path_index, :real_lengths[scenario]]
            if path_is_valid[path_index] and real_lengths[scenario] >= MINIMUM_DAYS and np.all(prices > 0):
                valid_paths.append(prices)
        run = (config, seed)
        statistics_per_run[run] = statistics_of_paths(valid_paths)
        distances_per_run[run] = {name: wasserstein_distance(statistics_per_run[run][name], real_statistics[name])
                                  for name in STATISTICS}
        print(f"{config}_{seed} ({len(valid_paths)} paths): "
              + "  ".join(f"{name} {value:.4f}" for name, value in distances_per_run[run].items()))

print("\nDistance to real, mean +- std over seeds (0 = identical to real)")
for config in CONFIGS:
    summary_parts = []
    for name in STATISTICS:
        values = [distances_per_run[(config, seed)][name] for seed in SEEDS]
        summary_parts.append(f"{name} {np.mean(values):.4f}+-{np.std(values, ddof=1):.4f}")
    print(config, "  ".join(summary_parts))

print("\nReturn std per seed (real {:.4f})".format(real_statistics["daily_returns"].std()))
for config in CONFIGS:
    print(config, "  ".join(f"{seed} {statistics_per_run[(config, seed)]['daily_returns'].std():.4f}" for seed in SEEDS))

pooled_statistics = {"Real": real_statistics}
for config in CONFIGS:
    pooled_statistics[config] = {name: np.concatenate([statistics_per_run[(config, seed)][name] for seed in SEEDS])
                                 for name in STATISTICS}

print("\nTypical values")
for label, statistics in pooled_statistics.items():
    print(f"{label:5} return std {statistics['daily_returns'].std():.4f}  "
          f"return excess kurtosis {kurtosis(statistics['daily_returns']):.2f}  "
          f"mean volatility {statistics['volatility'].mean():.4f}  "
          f"median max drawdown {np.median(statistics['max_drawdown']):.3f}")

figure, axes = plt.subplots(1, 3, figsize=(16, 5))
for axis, name in zip(axes, STATISTICS):
    all_values = np.concatenate([statistics[name] for statistics in pooled_statistics.values()])
    histogram_bins = np.linspace(*np.percentile(all_values, [0.5, 99.5]), 60)
    for label, statistics in pooled_statistics.items():
        axis.hist(statistics[name], bins=histogram_bins, density=True, histtype="step",
                  linewidth=1.5, color=PLOT_COLOURS[label], label=label)
    axis.set_title(PLOT_TITLES[name])
axes[0].set_yscale("log")
axes[0].legend()
figure.suptitle("Experiment 4: real vs generated paths (3 seeds)")
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=150)
