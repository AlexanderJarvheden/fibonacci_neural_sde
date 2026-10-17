import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
LOGS_FOLDER = EXPERIMENT_FOLDER / "logs"
FIGURE_FOLDER = EXPERIMENT_FOLDER / "figures"
RUN_NAMES = ["real", "A_s1", "A_s2", "A_s3", "C_s1", "C_s2", "C_s3"]
TRAINING_DATA_GROUPS = {"Real": ["real"], "Synthetic A": ["A_s1", "A_s2", "A_s3"], "Synthetic C": ["C_s1", "C_s2", "C_s3"]}
GROUP_COLOURS = {"Real": "dimgrey", "Synthetic A": "tab:blue", "Synthetic C": "tab:green"}
DISPLAY_NAMES = {"Real": "Real",
                 "Synthetic A": "Synthetic A (Fibonacci only)",
                 "Synthetic C": "Synthetic C (Fibonacci + MA20 + roll_vol)"}
CLASS_NAMES = ["breaks down", "no breakout", "breaks up"]


def confusion_matrices_from_log(run_name):
    log_text = (LOGS_FOLDER / f"clf_{run_name}.log").read_text()
    matrix_block = log_text.split("CM list:")[1].split("Train Accuracy List:")[0]
    numbers = [float(number) for number in re.findall(r"\d+\.\d*", matrix_block)]
    assert len(numbers) == 90, f"{run_name}: expected 90 numbers, found {len(numbers)}"
    return np.array(numbers).reshape(10, 3, 3)


def accuracy_and_f1(confusion_matrix):
    # rows are in percent of each true class, and the test classes are balanced
    recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    macro_f1 = np.mean(2 * precision * recall / (precision + recall))
    return np.mean(recall) * 100, macro_f1 * 100


confusion_matrices = {run_name: confusion_matrices_from_log(run_name) for run_name in RUN_NAMES}
scores_per_classifier = {run_name: np.array([accuracy_and_f1(matrix) for matrix in matrices])
                         for run_name, matrices in confusion_matrices.items()}
mean_scores_per_run = {run_name: scores.mean(axis=0) for run_name, scores in scores_per_classifier.items()}
real_accuracies = scores_per_classifier["real"][:, 0]

print("per run (mean +- std over the 10 classifier repeats)")
for run_name, scores in scores_per_classifier.items():
    print(f"{run_name:5s} accuracy {scores[:, 0].mean():.2f} +- {scores[:, 0].std():.2f}   "
          f"macro F1 {scores[:, 1].mean():.2f} +- {scores[:, 1].std():.2f}")

print("\nsummary (synthetic: mean +- std over the 3 generator seeds)")
for group, run_names in TRAINING_DATA_GROUPS.items():
    group_scores = np.array([mean_scores_per_run[run_name] for run_name in run_names])
    if len(run_names) == 1:
        print(f"{group:12s} accuracy {group_scores[0, 0]:.2f}   macro F1 {group_scores[0, 1]:.2f}")
        continue
    spread = group_scores.std(axis=0, ddof=1)
    print(f"{group:12s} accuracy {group_scores[:, 0].mean():.2f} +- {spread[0]:.2f}   "
          f"macro F1 {group_scores[:, 1].mean():.2f} +- {spread[1]:.2f}")

figure, (scores_axis, seed_axis) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})

bar_positions = np.arange(2)
for group_index, (group, run_names) in enumerate(TRAINING_DATA_GROUPS.items()):
    group_means = np.array([mean_scores_per_run[run_name] for run_name in run_names]).mean(axis=0)
    bars = scores_axis.bar(bar_positions + (group_index - 1) * 0.25, group_means, width=0.25,
                           color=GROUP_COLOURS[group], label=DISPLAY_NAMES[group])
    for bar, value in zip(bars, group_means):
        scores_axis.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}",
                         ha="center", va="bottom", fontsize=8)
scores_axis.set_xticks(bar_positions)
scores_axis.set_xticklabels(["Accuracy", "Macro F1"])
scores_axis.set_ylim(0, 85)
scores_axis.set_ylabel("Score on real data (%)")
scores_axis.set_title("Accuracy and macro F1")
scores_axis.legend(fontsize=8, loc="upper left", ncol=1, frameon=False)
scores_axis.spines[["top", "right"]].set_visible(False)

seed_axis.axhspan(real_accuracies.mean() - real_accuracies.std(), real_accuracies.mean() + real_accuracies.std(),
                  color="grey", alpha=0.15)
seed_axis.axhline(real_accuracies.mean(), color="dimgrey", lw=1.2, label="Real (shaded area = ± std)")
for group_index, group in enumerate(["Synthetic A", "Synthetic C"]):
    for seed_index, run_name in enumerate(TRAINING_DATA_GROUPS[group]):
        dot_position = group_index + (seed_index - 1) * 0.15
        seed_accuracy = mean_scores_per_run[run_name][0]
        seed_axis.scatter(dot_position, seed_accuracy, color=GROUP_COLOURS[group], s=50, zorder=3)
        seed_axis.annotate(f"seed {seed_index + 1}", (dot_position, seed_accuracy), xytext=(0, -14),
                           textcoords="offset points", ha="center", fontsize=7)
seed_axis.set_xticks([0, 1])
seed_axis.set_xticklabels(["Synthetic A\n(Fibonacci only)", "Synthetic C\n(Fibonacci + MA20 + roll_vol)"])
seed_axis.set_xlim(-0.6, 1.6)
seed_axis.set_ylim(61.0, 62.5)
seed_axis.set_ylabel("Accuracy (%)")
seed_axis.set_title("Accuracy for each generator seed")
seed_axis.legend(fontsize=8, loc="upper right", frameon=False)
seed_axis.spines[["top", "right"]].set_visible(False)

figure.tight_layout()
figure.savefig(FIGURE_FOLDER / "fig8_downstream_scores.png", dpi=150)

figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
for axis, (group, run_names) in zip(axes, TRAINING_DATA_GROUPS.items()):
    mean_matrix = np.concatenate([confusion_matrices[run_name] for run_name in run_names]).mean(axis=0)
    axis.imshow(mean_matrix, cmap="Blues", vmin=0, vmax=100)
    for row in range(3):
        for column in range(3):
            axis.text(column, row, f"{mean_matrix[row, column]:.1f}%", ha="center", va="center",
                      color="white" if mean_matrix[row, column] > 50 else "black")
    axis.set_xticks(range(3))
    axis.set_xticklabels(CLASS_NAMES)
    axis.set_yticks(range(3))
    axis.set_yticklabels(CLASS_NAMES)
    axis.set_xlabel("predicted")
    axis.set_title(DISPLAY_NAMES[group], fontsize=11)
axes[0].set_ylabel("true")
figure.tight_layout()
figure.savefig(FIGURE_FOLDER / "fig9_confusion_matrices.png", dpi=150)
