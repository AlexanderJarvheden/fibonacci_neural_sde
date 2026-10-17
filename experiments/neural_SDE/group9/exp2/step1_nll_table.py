import pathlib
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_FOLDER = pathlib.Path(__file__).resolve().parents[4] / "workspace/neural_SDE/logs/train_neural_SDE"

for config in "ABC":
    final_nll_per_seed = []
    for seed in (1, 2, 3):
        run_name = f"Exp2_{config}_s{seed}"
        log_file = next((LOGS_FOLDER / run_name).rglob("events.out.tfevents.*"))
        training_log = EventAccumulator(str(log_file.parent))
        training_log.Reload()
        final_nll_per_seed.append(training_log.Scalars("val/loss_sde")[-1].value)
    print(f"{config}  seeds {['%.4f' % nll for nll in final_nll_per_seed]}  "
          f"mean {np.mean(final_nll_per_seed):.4f}  std {np.std(final_nll_per_seed, ddof=1):.4f}")
