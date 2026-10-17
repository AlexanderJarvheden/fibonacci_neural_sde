from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
FIGURE_FOLDER = f"{EXPERIMENT_FOLDER}/figures"
features = np.load(f"{EXPERIMENT_FOLDER}/features.npy")
feature_names = [str(name) for name in np.load(f"{EXPERIMENT_FOLDER}/feature_names.npy")]

correlation_matrix = np.corrcoef(features.T)
number_of_features = len(feature_names)

print(f"{'':<14}" + "".join(f"{name[:8]:>10}" for name in feature_names))
for row, name in enumerate(feature_names):
    print(f"{name:<14}" + "".join(f"{correlation_matrix[row, column]:>10.2f}" for column in range(number_of_features)))

print("\npairs with |corr| > 0.8:")
for row in range(number_of_features):
    for column in range(row + 1, number_of_features):
        if abs(correlation_matrix[row, column]) > 0.8:
            print(f"  {feature_names[row]} vs {feature_names[column]}: {correlation_matrix[row, column]:.3f}")

figure, axis = plt.subplots(figsize=(8, 7))
heatmap = axis.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
axis.set_xticks(range(number_of_features))
axis.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
axis.set_yticks(range(number_of_features))
axis.set_yticklabels(feature_names, fontsize=8)
for row in range(number_of_features):
    for column in range(number_of_features):
        axis.text(column, row, f"{correlation_matrix[row, column]:.2f}", ha="center", va="center", fontsize=7)
axis.set_title("Candidate feature correlations")
figure.colorbar(heatmap)
figure.tight_layout()
figure.savefig(f"{FIGURE_FOLDER}/fig4_feature_correlation.png", dpi=150)
