# from pathlib import Path
# import sys
# import json
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import r2_score

# ROOT = Path(__file__).resolve().parent
# ACORN_DIR = ROOT / "acorn-main" #"CEBRA-idea2"  # "acorn-main"
# CYCLEGAN_DIR = Path("/mnt/upmwmathis/scratch/hossein/aj_project/adversarial_BCI")
# DATA_ROOT = "/data/hossein/data"
# sys.path.insert(0, str(ACORN_DIR))
# sys.path.insert(0, str(CYCLEGAN_DIR))

# from cebra import CEBRA
# from decoder_standard.wiener_filter import (
#     format_data_from_trials,
#     train_wiener_filter,
#     test_wiener_filter,
# )

# SESSION_NAME = "Mihili_CO_2014_raw"
# N_DAYS = 11
# TRAIN_FRACTION = 0.8
# N_LAGS = 4


# def run_acorn():
#     print("\n========== ACORN SINGLE SESSION ==========\n")

#     # TODO:
#     # replace with exact loader import from your dataset loader
#     from dataset_loader import DatasetLoader

#     loader = DatasetLoader(data_root_dir=DATA_ROOT)

#     # Load all 11 days, keep only the first behavior column
#     all_data = {}
#     for day in range(N_DAYS):
#         spike, behavior = loader.load_dataset_day(day, SESSION_NAME)
#         X = np.asarray(spike)
#         Y = np.asarray(behavior)[:, :1]
#         all_data[day] = (X, Y)
#         print(f"day {day:2d}: spike={X.shape}, behavior={Y.shape}")

#     # Train on the first 80% of day 0
#     X0, Y0 = all_data[0]
#     split0 = int(TRAIN_FRACTION * len(X0))
#     X_train = X0[:split0]
#     Y_train = Y0[:split0]

#     print(f"\nTrain on day 0 (first {int(TRAIN_FRACTION*100)}%): {X_train.shape}")

#     n_neurons = X_train.shape[1]
#     neuron_count = max(1, int(np.ceil(n_neurons * 0.25)))
#     print(f"Input neurons: {n_neurons} | adv_neuron_count = {neuron_count} (25%)")

#     # Baseline (all neurons)
#     model = CEBRA(model_architecture="offset36-model",
#                   batch_size=102,
#                   max_iterations=3000,
#                   output_dimension=32,
#                   training_mode="adversarial",
#                   adv_epsilon=0.5,
#                   adv_alpha=0.1,
#                   adv_steps=10,
#                   attack_norm="linf",
#                   device="cuda")

#     # IDEA2 (attack only on 25% of neurons)
#     # model = CEBRA(
#     #     model_architecture="offset36-model",
#     #     batch_size=102,
#     #     max_iterations=3000,
#     #     output_dimension=32,
#     #     training_mode="adversarial",
#     #     adv_epsilon=0.5,
#     #     adv_alpha=0.1,
#     #     adv_steps=10,
#     #     attack_norm="linf",
#     #     device="cuda",

#     #     adv_neuron_count=neuron_count,
#     #     adv_neuron_selection="gradient",

#     #     attack_target="reference",
#     #     adv_restarts=1,
#     #     adv_best_iterate=False,
#     #     adv_random_start=True,
#     #     adv_eval_mode=False,
#     #     adv_budget_mode="per_view",
#     #     adv_clip_min=None,
#     #     adv_clip_max=None,
#     # )

#     model.fit(X_train, Y_train)

#     # Freeze: transform + train Wiener decoder on day 0 train split
#     Z_train = model.transform(X_train)
#     Z_train_w, Y_train_w = format_data_from_trials(Z_train, Y_train, N_LAGS)
#     decoder = train_wiener_filter(Z_train_w, Y_train_w)

#     # Test on the last 20% of every day (0 to 10)
#     per_day_results = []
#     for day in range(N_DAYS):
#         Xd, Yd = all_data[day]
#         split_d = int(TRAIN_FRACTION * len(Xd))
#         X_test = Xd[split_d:]
#         Y_test = Yd[split_d:]

#         Z_test = model.transform(X_test)
#         Z_test_w, Y_test_w = format_data_from_trials(Z_test, Y_test, N_LAGS)

#         pred = test_wiener_filter(Z_test_w, decoder)
#         r2 = r2_score(Y_test_w, pred, multioutput="raw_values")

#         per_day_results.append({
#             "model": "ACORN",
#             "dataset": SESSION_NAME,
#             "day": day,
#             "n_test_samples": int(len(X_test)),
#             "r2_mean": float(np.mean(r2)),
#             "r2_each": r2.tolist(),
#         })
#         print(f"day {day:2d} | test={X_test.shape} | R2 mean = {np.mean(r2):.4f}")

#     overall_r2_mean = float(np.mean([r["r2_mean"] for r in per_day_results]))
#     print(f"\nOverall R2 (mean across {N_DAYS} days): {overall_r2_mean:.4f}")

#     return per_day_results, overall_r2_mean


# if __name__ == "__main__":
#     per_day_results, overall_r2_mean = run_acorn()

#     df = pd.DataFrame(per_day_results)
#     out = ROOT / "acorn_cyclegan_results.csv"
#     df.to_csv(out, index=False)

#     summary = {
#         "session": SESSION_NAME,
#         "n_days": N_DAYS,
#         "train_day": 0,
#         "train_fraction": TRAIN_FRACTION,
#         "n_labels": 1,
#         "overall_r2_mean_across_days": overall_r2_mean,
#         "per_day_r2_mean": {int(r["day"]): r["r2_mean"] for r in per_day_results},
#     }
#     out_summary = ROOT / "acorn_cyclegan_summary.json"
#     out_summary.write_text(json.dumps(summary, indent=2))

#     print("\nDONE")
#     print(df[["day", "n_test_samples", "r2_mean"]])
#     print("\nSaved:", out)
#     print("Saved:", out_summary)

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
ACORN_DIR = ROOT / "acorn-main" #"CEBRA-idea2"  # "acorn-main"
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

SESSION_NAME = "Mihili_CO_2014_raw"
N_DAYS = 11             
TRAIN_FRACTION = 0.8    
N_LAGS = 4             
USE_ALL_BEHAVIOR = True 


def run_acorn():
    print("\n========== ACORN SINGLE SESSION ==========\n")

    # TODO:
    # replace with exact loader import from your dataset loader
    from dataset_loader import DatasetLoader

    loader = DatasetLoader(data_root_dir=DATA_ROOT)

    # all_data = {}
    # for day in range(N_DAYS):
    #     spike, behavior = loader.load_dataset_day(day, SESSION_NAME)
    #     X = np.asarray(spike)
    #     Y = np.asarray(behavior)
    #     if not USE_ALL_BEHAVIOR:
    #         Y = Y[:, :1]
    #     all_data[day] = (X, Y)
    #     print(f"day {day:2d}: spike={X.shape}, behavior={Y.shape}")
    all_data = {}
    for day in range(N_DAYS):
        spike, behavior = loader.load_dataset_day(day, SESSION_NAME)
        X = np.asarray(spike)
        Y = np.asarray(behavior)
        if not USE_ALL_BEHAVIOR:
            Y = Y[:, :1]
        # # ==========================
        # # neuron-wise normalization
        # # independently for each day
        # # ==========================
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        all_data[day] = (X, Y)
        print(f"day {day:2d}: spike={X.shape}, behavior={Y.shape}")

    X0, Y0 = all_data[0]
    split0 = int(TRAIN_FRACTION * len(X0))
    X_train = X0[:split0]
    Y_train = Y0[:split0]

    print(f"\nTrain on day 0 (first {int(TRAIN_FRACTION*100)}%): {X_train.shape}")

    n_neurons = X_train.shape[1]
    neuron_count = max(1, int(np.ceil(n_neurons * 0.25)))
    print(f"Input neurons: {n_neurons} | adv_neuron_count = {neuron_count} (25%)")

    model = CEBRA(model_architecture="offset36-model",
                  batch_size=102,
                  max_iterations=3000,
                  output_dimension=32,
                  training_mode="adversarial",
                  adv_epsilon=0.5,
                  adv_alpha=0.1,
                  adv_steps=10,
                  attack_norm="linf",
                  device="cuda")

    # model = CEBRA(
    #     model_architecture="offset36-model",
    #     batch_size=102,
    #     max_iterations=3000,
    #     output_dimension=32,
    #     training_mode="adversarial",
    #     adv_epsilon=0.5,
    #     adv_alpha=0.1,
    #     adv_steps=10,
    #     attack_norm="linf",
    #     device="cuda",
    
    #     adv_neuron_count=neuron_count,
    #     adv_neuron_selection="gradient",
    
    #     attack_target="reference",
    #     adv_restarts=1,
    #     adv_best_iterate=False,
    #     adv_random_start=True,
    #     adv_eval_mode=False,
    #     adv_budget_mode="per_view",
    #     adv_clip_min=None,
    #     adv_clip_max=None,
    # )

    model.fit(X_train, Y_train)

    Z_train = model.transform(X_train)
    Z_train_w, Y_train_w = format_data_from_trials(Z_train, Y_train, N_LAGS)
    decoder = train_wiener_filter(Z_train_w, Y_train_w)

    per_day_results = []
    for day in range(N_DAYS):
        Xd, Yd = all_data[day]
        split_d = int(TRAIN_FRACTION * len(Xd))
        X_test = Xd[split_d:]
        Y_test = Yd[split_d:]

        Z_test = model.transform(X_test)
        Z_test_w, Y_test_w = format_data_from_trials(Z_test, Y_test, N_LAGS)

        pred = test_wiener_filter(Z_test_w, decoder)
        r2 = r2_score(Y_test_w, pred, multioutput="raw_values")

        per_day_results.append({
            "model": "ACORN",
            "dataset": SESSION_NAME,
            "day": day,
            "n_test_samples": int(len(X_test)),
            "r2_mean": float(np.mean(r2)),
            "r2_each": r2.tolist(),
        })
        print(f"day {day:2d} | test={X_test.shape} | R2 mean = {np.mean(r2):.4f}")

    overall_r2_mean = float(np.mean([r["r2_mean"] for r in per_day_results]))
    print(f"\nOverall R2 (mean across {N_DAYS} days): {overall_r2_mean:.4f}")

    return per_day_results, overall_r2_mean


if __name__ == "__main__":
    per_day_results, overall_r2_mean = run_acorn()

    df = pd.DataFrame(per_day_results)
    out = ROOT / "acorn_cyclegan_results.csv"
    df.to_csv(out, index=False)

    summary = {
        "session": SESSION_NAME,
        "n_days": N_DAYS,
        "train_day": 0,
        "train_fraction": TRAIN_FRACTION,
        "use_all_behavior": USE_ALL_BEHAVIOR,
        "overall_r2_mean_across_days": overall_r2_mean,
        "per_day_r2_mean": {int(r["day"]): r["r2_mean"] for r in per_day_results},
    }
    out_summary = ROOT / "acorn_cyclegan_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))

    print("\nDONE")
    print(df[["day", "n_test_samples", "r2_mean"]])
    print("\nSaved:", out)
    print("Saved:", out_summary)

# from pathlib import Path
# import sys
# import json
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import r2_score

# ROOT = Path(__file__).resolve().parent
# ACORN_DIR = ROOT / "CEBRA-idea2" #"acorn-main" 
# CYCLEGAN_DIR = Path("/mnt/upmwmathis/scratch/hossein/aj_project/adversarial_BCI")
# DATA_ROOT = "/data/hossein/data"
# sys.path.insert(0, str(ACORN_DIR))
# sys.path.insert(0, str(CYCLEGAN_DIR))

# from cebra import CEBRA
# from decoder_standard.wiener_filter import (
#     format_data_from_trials,
#     train_wiener_filter,
#     test_wiener_filter,
# )

# def run_acorn():
#     print("\n========== ACORN SINGLE SESSION ==========\n")

#     # TODO:
#     # replace with exact loader import from your dataset loader
#     from dataset_loader import DatasetLoader

#     loader = DatasetLoader(data_root_dir=DATA_ROOT)

#     # CO-M day0
#     spike, behavior = loader.load_dataset_day(0,"Mihili_CO_2014_raw")

#     X = np.asarray(spike)
#     Y = np.asarray(behavior)

#     n = len(X)
#     split = int(0.8 * n)

#     # X_train = X[:split]
#     # Y_train = Y[:split]
#     # X_test = X[split:]
#     # Y_test = Y[split:]
#     X_train = X[:split]
#     Y_train = Y[:split, :1]
#     X_test = X[split:]
#     Y_test = Y[split:, :1]

#     print("train:", X_train.shape, "test:", X_test.shape)

#     model = CEBRA(
#         model_architecture="offset36-model",
#         batch_size=102,
#         max_iterations=3000,
#         output_dimension=32,
#         training_mode="adversarial",
#         adv_epsilon=0.5,
#         adv_alpha=0.1,
#         adv_steps=10,
#         attack_norm="linf",
#         device="cuda"
#     )

#     model.fit(X_train, Y_train)

#     Z_train = model.transform(X_train)
#     Z_test = model.transform(X_test)

#     # Wiener decoder
#     n_lags = 4

#     Z_train_w, Y_train_w = format_data_from_trials(Z_train, Y_train, n_lags)
#     Z_test_w, Y_test_w = format_data_from_trials(Z_test, Y_test, n_lags)

#     decoder = train_wiener_filter(Z_train_w, Y_train_w)
#     pred = test_wiener_filter(Z_test_w, decoder)

#     r2 = r2_score(Y_test_w, pred, multioutput="raw_values")

#     return {
#         "model": "ACORN",
#         "dataset": "CO-M-day0",
#         "r2_mean": float(np.mean(r2)),
#         "r2_each": r2.tolist()
#     }

# if __name__ == "__main__":
#     results = []
#     acorn_result = run_acorn()
#     results.append(acorn_result)
#     df = pd.DataFrame(results)
#     out = ROOT / "acorn_cyclegan_results.csv"
#     df.to_csv(out, index=False)
#     print("\nDONE")
#     print(df)
#     print("\nSaved:", out)
