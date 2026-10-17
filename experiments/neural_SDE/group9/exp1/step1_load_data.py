from pathlib import Path
import os
import numpy as np

from amgm import config as amgm_config
from amgm.data.neural_SDE import NeuralSDEDataset

EXPERIMENT_FOLDER = Path(__file__).resolve().parent
os.makedirs(f"{EXPERIMENT_FOLDER}/figures", exist_ok=True)

dataset = NeuralSDEDataset(
    issue_ids=[],
    num_iids=50,
    start_date="2014-12-31",
    end_date="2017-12-31",
    lookback_window=252,
    max_windows_per_issue=1000,
    dt=1.0,
    price_value_col="ClAdjLoc",
    training_data_type="US_Stocks",
    data_source="local",
    dset_path=amgm_config.dataset_root / "data-20250505",
    rng_seed=1580886929,
)

scaled_windows = dataset.price_window.numpy()
np.save(f"{EXPERIMENT_FOLDER}/windows.npy", scaled_windows)
print("saved windows.npy", scaled_windows.shape)

window_minimums = dataset.sample_min.numpy().reshape(-1, 1)
window_ranges = dataset.sample_range.numpy().reshape(-1, 1)
raw_windows = scaled_windows * window_ranges + window_minimums
np.save(f"{EXPERIMENT_FOLDER}/windows_raw.npy", raw_windows)
