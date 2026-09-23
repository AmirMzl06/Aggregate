from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
ACORN_DIR = ROOT / "acorn-main"
CYCLEGAN_DIR = Path("/mnt/upmwmathis/scratch/hossein/aj_project/adversarial_BCI")
DATA_ROOT = "/data/hossein/data"
sys.path.insert(0, str(ACORN_DIR))
sys.path.insert(0, str(CYCLEGAN_DIR))

from cebra import CEBRA
from decoder_standard.wiener_filter import (
    format_data_from_trials,
    train_wiener_filter,
    test_wiener_filter,
)

def run_acorn():
    print("\n========== ACORN SINGLE SESSION ==========\n")

    # TODO:
    # replace with exact loader import from your dataset loader
    from dataset_loader import DatasetLoader

    loader = DatasetLoader(data_root_dir=DATA_ROOT)

    # CO-M day0
    spike, behavior = loader.load_dataset_day(dataset_name="CO-M", day_id=0)

    X = np.asarray(spike)
    Y = np.asarray(behavior)

    n = len(X)
    split = int(0.8 * n)

    X_train = X[:split]
    Y_train = Y[:split]
    X_test = X[split:]
    Y_test = Y[split:]

    print("train:", X_train.shape, "test:", X_test.shape)

    model = CEBRA(
        model_architecture="offset36-model",
        batch_size=102,
        max_iterations=30000,
        output_dimension=32,
        training_mode="adversarial",
        adv_epsilon=0.5,
        adv_alpha=0.1,
        adv_steps=10,
        attack_norm="linf",
        device="cuda"
    )

    model.fit(X_train, Y_train)

    Z_train = model.transform(X_train)
    Z_test = model.transform(X_test)

    # Wiener decoder
    n_lags = 4

    Z_train_w, Y_train_w = format_data_from_trials(Z_train, Y_train, n_lags)
    Z_test_w, Y_test_w = format_data_from_trials(Z_test, Y_test, n_lags)

    decoder = train_wiener_filter(Z_train_w, Y_train_w)
    pred = test_wiener_filter(Z_test_w, decoder)

    r2 = r2_score(Y_test_w, pred, multioutput="raw_values")

    return {
        "model": "ACORN",
        "dataset": "CO-M-day0",
        "r2_mean": float(np.mean(r2)),
        "r2_each": r2.tolist()
    }

if __name__ == "__main__":
    results = []
    acorn_result = run_acorn()
    results.append(acorn_result)
    df = pd.DataFrame(results)
    out = ROOT / "acorn_cyclegan_results.csv"
    df.to_csv(out, index=False)
    print("\nDONE")
    print(df)
    print("\nSaved:", out)
