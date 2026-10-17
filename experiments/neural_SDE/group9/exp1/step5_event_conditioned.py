from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fibonacci_state_detection import detect_fibonacci_interactions

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
FIGURE_FOLDER = f"{EXPERIMENT_FOLDER}/figures"
scaled_windows = np.load(f"{EXPERIMENT_FOLDER}/windows.npy")
features = np.load(f"{EXPERIMENT_FOLDER}/features.npy")
feature_names = list(np.load(f"{EXPERIMENT_FOLDER}/feature_names.npy"))
fibonacci_ratios = np.array([0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0])
window_length = scaled_windows.shape[1]

last_event_types, days_since_last_event = [], []
for window in scaled_windows:
    detected_events = detect_fibonacci_interactions(
        window, fibonacci_ratios, tolerance=0.02, min_dwell_steps=3,
        confirmation_steps=10, max_dwell_steps=20)
    if not detected_events:
        last_event_types.append("none")
        days_since_last_event.append(10**6)
        continue
    last_event = detected_events[-1]
    event_type = last_event["event_type"]
    last_event_types.append(event_type.value if hasattr(event_type, "value") else str(event_type))
    event_day = last_event.get("output_index", last_event.get("entry_index", 0))
    days_since_last_event.append(window_length - 1 - int(event_day))

last_event_types = np.array(last_event_types)
days_since_last_event = np.array(days_since_last_event)
print("recency of last event (days before window end):")
print("  median %d, 25th %d, 75th %d" % (np.median(days_since_last_event),
      np.percentile(days_since_last_event, 25), np.percentile(days_since_last_event, 75)))

event_order = ["hover", "breakout", "pullback", "timeout"]
features_to_show = ["d_MA20", "MA_slope", "roll_vol", "mom21", "RSI14", "pos_in_range"]

for file_tag, window_selection, description in [
    ("all", np.ones(len(days_since_last_event), bool), "all windows"),
    ("recent", days_since_last_event <= 21, "last event within 21 days"),
]:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7))
    for axis, feature_name in zip(axes.ravel(), features_to_show):
        feature_index = feature_names.index(feature_name)
        box_values, box_labels = [], []
        for event in event_order:
            selected = window_selection & (last_event_types == event)
            if selected.sum() >= 10:
                box_values.append(features[selected, feature_index])
                box_labels.append(f"{event}\nn={selected.sum()}")
        axis.boxplot(box_values, tick_labels=box_labels, showfliers=False)
        axis.axhline(0, color="grey", lw=0.5)
        axis.set_title(feature_name, fontsize=10)
        axis.tick_params(labelsize=8)
    figure.suptitle(f"Feature distributions by last detected event ({description})", fontsize=13)
    figure.tight_layout()
    figure.savefig(f"{FIGURE_FOLDER}/fig3_event_conditioned_{file_tag}.png", dpi=150)
    print(f"saved fig3_event_conditioned_{file_tag}.png")

print("\ngroup medians (all windows):")
print(f"{'feature':<14}" + "".join(f"{event:>12}" for event in event_order))
for feature_name in features_to_show:
    feature_index = feature_names.index(feature_name)
    table_row = f"{feature_name:<14}"
    for event in event_order:
        selected = last_event_types == event
        table_row += f"{np.median(features[selected, feature_index]):>12.4f}" if selected.sum() >= 10 else f"{'-':>12}"
    print(table_row)
