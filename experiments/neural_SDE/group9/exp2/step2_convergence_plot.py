import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_FOLDER = pathlib.Path(__file__).resolve().parents[4] / "workspace/neural_SDE/logs/train_neural_SDE"
FIGURE_PATH = pathlib.Path(__file__).resolve().parent / "figures/fig5_convergence.png"
CONFIG_COLOURS = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}

figure, axis = plt.subplots(figsize=(8, 5))

for config in "ABC":
    for seed in (1, 2, 3):
        run_name = f"Exp2_{config}_s{seed}"
        log_file = next((LOGS_FOLDER / run_name).rglob("events.out.tfevents.*"))
        training_log = EventAccumulator(str(log_file.parent))
        training_log.Reload()
        validation_nll = training_log.Scalars("val/loss_sde")
        axis.plot([entry.step for entry in validation_nll], [entry.value for entry in validation_nll],
                  color=CONFIG_COLOURS[config], alpha=0.75, linewidth=1.2,
                  label=config if seed == 1 else None)

axis.set_title("Experiment 2: validation NLL during training, 3 seeds per configuration")
axis.set_xlabel("training step")
axis.set_ylabel("validation NLL")
axis.legend(title="config")
axis.grid(alpha=0.3)
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=150)
