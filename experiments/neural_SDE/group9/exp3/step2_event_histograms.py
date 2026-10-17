import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fibonacci_state_detection import detect_fibonacci_interactions

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
SAMPLES_FOLDER = EXPERIMENT_FOLDER / "samples"
FIGURE_PATH = EXPERIMENT_FOLDER / "figures" / "fig6_event_histograms.png"
FIBONACCI_LEVELS = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
EVENT_TYPES = ["pullback", "hover", "breakout", "timeout"]
MINIMUM_DAYS = 60
CONFIGS = ["A", "B", "C"]
SEEDS = ["s1", "s2", "s3"]
CONFIG_COLOURS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}


def count_events(prices, price_min, price_range, event_counts):
    scaled_path = (prices - price_min) / price_range
    detected_events = detect_fibonacci_interactions(scaled_path, FIBONACCI_LEVELS, tolerance=0.02, min_dwell_steps=3,
                                                    confirmation_steps=10, max_dwell_steps=20)
    for event in detected_events:
        event_counts[event["event_type"].value] += 1

def summarise(event_counts, number_of_paths):
    shares = np.array([event_counts[event_type] for event_type in EVENT_TYPES]) / sum(event_counts.values())
    events_per_path = sum(event_counts.values()) / number_of_paths
    return shares, events_per_path


reference_samples = np.load(SAMPLES_FOLDER / "Exp2_A_s1.npz")
real_lengths = reference_samples["original_price_len"]
scenarios_to_use = np.where(real_lengths >= MINIMUM_DAYS)[0]

real_counts = dict.fromkeys(EVENT_TYPES, 0)
for scenario in scenarios_to_use:
    real_prices = reference_samples["original_price"][scenario, :real_lengths[scenario]]
    count_events(real_prices, reference_samples["price_min"][scenario], reference_samples["price_range"][scenario], real_counts)
real_shares, real_events_per_path = summarise(real_counts, len(scenarios_to_use))
print(f"Real ({len(scenarios_to_use)} paths): {real_counts}, events per path {real_events_per_path:.2f}")

shares, events_per_path, distance_to_real = {}, {}, {}
for config in CONFIGS:
    for seed in SEEDS:
        samples = np.load(SAMPLES_FOLDER / f"Exp2_{config}_{seed}.npz")
        synthetic_paths = samples["synthetic_paths"]
        scenario_of_path = samples["base_sample_id"]
        path_is_valid = samples["is_valid"]
        price_minimums = samples["price_min"]
        price_ranges = samples["price_range"]
        model_counts = dict.fromkeys(EVENT_TYPES, 0)
        number_of_paths = 0
        for path_index, scenario in enumerate(scenario_of_path):
            if not path_is_valid[path_index] or real_lengths[scenario] < MINIMUM_DAYS:
                continue
            synthetic_prices = synthetic_paths[path_index, :real_lengths[scenario]]
            count_events(synthetic_prices, price_minimums[scenario], price_ranges[scenario], model_counts)
            number_of_paths += 1
        run = (config, seed)
        shares[run], events_per_path[run] = summarise(model_counts, number_of_paths)
        distance_to_real[run] = 0.5 * np.abs(shares[run] - real_shares).sum()
        print(f"{config}_{seed} ({number_of_paths} paths): {model_counts}, "
              f"events per path {events_per_path[run]:.2f}, distance to real {distance_to_real[run]:.4f}")

print("\nSummary, mean +- std over seeds")
print("Real  " + "  ".join(f"{event_type} {share:.3f}" for event_type, share in zip(EVENT_TYPES, real_shares)))
mean_shares, std_shares = {}, {}
for config in CONFIGS:
    config_shares = np.array([shares[(config, seed)] for seed in SEEDS])
    config_rates = np.array([events_per_path[(config, seed)] for seed in SEEDS])
    config_distances = np.array([distance_to_real[(config, seed)] for seed in SEEDS])
    mean_shares[config], std_shares[config] = config_shares.mean(0), config_shares.std(0, ddof=1)
    print(f"{config}     " + "  ".join(f"{event_type} {mean:.3f}+-{std:.3f}" for event_type, mean, std
                                in zip(EVENT_TYPES, mean_shares[config], std_shares[config]))
          + f"  events/path {config_rates.mean():.2f}+-{config_rates.std(ddof=1):.2f}"
          + f"  distance {config_distances.mean():.4f}+-{config_distances.std(ddof=1):.4f}")

figure, axis = plt.subplots(figsize=(10, 5))
bar_positions = np.arange(len(EVENT_TYPES))
axis.bar(bar_positions - 0.3, real_shares, 0.2, color="grey", label="Real data")
for config_index, config in enumerate(CONFIGS):
    axis.bar(bar_positions - 0.1 + 0.2 * config_index, mean_shares[config], 0.2, yerr=std_shares[config],
             capsize=3, color=CONFIG_COLOURS[config], label=f"Config {config}")
axis.set_xticks(bar_positions, ["pullback (bounce)", "hover", "breakout", "timeout"])
axis.set_ylabel("share of all events")
axis.set_title("Experiment 3: Fibonacci event mix, real vs generated (error bars = std over 3 seeds)")
axis.legend()
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=150)
