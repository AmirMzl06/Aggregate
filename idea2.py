"""C-CO12: adversarial baseline versus PGD on a subset of input neurons.

Run this file next to the CEBRA-idea2 folder:
    python idea2.py

Both arms: reference only, one restart, last iterate, epsilon=0.2.
Each arm trains a fresh CEBRA encoder and a separate, identically initialized
MLP decoder. Data stay in their original scale. Evaluation uses valid_data /
valid_label from the NPZ: the reported R2 is VALIDATION R2, not held-out test R2.
No Jacobian computation or CLEAN arm is included in this two-arm comparison.
The default runs one paired seed; add seeds below to assess run variability.
"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import gc
import inspect
import json
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "CEBRA-idea2"
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
SESSION = "C-CO12"
NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
OUT_ROOT = ROOT / "ACORN_IDEA2_CCO12_COMPARE"

# Keep your original encoder / attack hyperparameters.
EPSILON = 0.2
EPS_STEPS = 10
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMP = 0.4
ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192
SEEDS = (42,)  # For a more reliable comparison: (42, 43, 44).
SAVE_CHECKPOINTS = True

# None derives k from a fraction; or set an explicit integer such as 5.
IDEA2_NEURON_COUNT = None
IDEA2_NEURON_FRACTION = 0.25
IDEA2_SELECTION = "gradient"  # "random" or "gradient"; no input normalization.
# gradient adds one clean-gradient probe per attack. Baseline has no probe.
# adv_eval_mode remains False in both arms to preserve your original setting.


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Seeds improve comparability; bitwise GPU reproducibility is not guaranteed.


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_cebra_fork():
    if not (CEBRA_DIR / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(f"Expected fork at {CEBRA_DIR}. Put the script next to it.")
    # Select this exact fork, including when executing in a reused interpreter.
    for name in list(sys.modules):
        if name == "cebra" or name.startswith("cebra."):
            del sys.modules[name]
    sys.path.insert(0, str(CEBRA_DIR))
    import cebra
    if not Path(cebra.__file__).resolve().is_relative_to(CEBRA_DIR.resolve()):
        raise RuntimeError(f"Wrong CEBRA imported: {cebra.__file__}")
    parameters = inspect.signature(cebra.CEBRA.__init__).parameters
    required = ("adv_neuron_count", "adv_neuron_selection")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise RuntimeError(f"This fork lacks {missing}. Install the matching idea implementation.")
    print("Using:", cebra.__file__, flush=True)
    return cebra


def load_data():
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        arrays = [np.asarray(data[key], dtype=np.float32) for key in
                  ("train_data", "valid_data", "train_label", "valid_label")]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    for name, values in zip(("X_train", "X_valid", "Y_train", "Y_valid"),
                            (x_train, x_valid, y_train, y_valid)):
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array; got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("Feature and label lengths do not match.")
    if x_train.shape[1] != x_valid.shape[1] or y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Training and validation dimensions do not match.")
    if len(x_train) < 36 or len(x_valid) < 36:
        raise ValueError("The offset36 encoder requires at least 36 timepoints per split.")
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def common_encoder_config():
    return dict(
        batch_size=BATCH_SIZE, temperature=TEMP, model_architecture=ARCH,
        time_offsets=1, max_iterations=MAX_ITER, output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN, learning_rate=3e-4, temperature_mode="constant",
        training_mode="adversarial", adv_epsilon=EPSILON,
        adv_alpha=EPSILON / 5, adv_steps=EPS_STEPS, attack_norm="linf",
        attack_target="reference", adv_restarts=1, adv_best_iterate=False,
        adv_random_start=True, adv_eval_mode=False, adv_budget_mode="per_view",
        adv_clip_min=None, adv_clip_max=None, device=DEVICE, verbose=True,
    )


def experiment_configs(n_neurons):
    if n_neurons < 2:
        raise ValueError("At least two neurons are required for a strict subset comparison.")
    if IDEA2_SELECTION not in ("random", "gradient"):
        raise ValueError("IDEA2_SELECTION must be random or gradient.")
    if IDEA2_NEURON_COUNT is None:
        if not 0.0 < IDEA2_NEURON_FRACTION < 1.0:
            raise ValueError("IDEA2_NEURON_FRACTION must be strictly between 0 and 1.")
        count = max(1, int(np.ceil(n_neurons * IDEA2_NEURON_FRACTION)))
    else:
        count = IDEA2_NEURON_COUNT
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count < n_neurons:
        raise ValueError(f"Use 1 <= IDEA2_NEURON_COUNT < {n_neurons}; got {count!r}.")
    count = int(count)
    common = common_encoder_config()
    # The selection method is identical; it is inactive when count=None.
    common["adv_neuron_selection"] = IDEA2_SELECTION
    return [
        ("BASELINE_all_neurons", dict(common, adv_neuron_count=None)),
        (f"IDEA2_{IDEA2_SELECTION}_k{count}", dict(common, adv_neuron_count=count)),
    ]


class TwoLayerMLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, MLP_HIDDEN), nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(), nn.Dropout(MLP_DROP), nn.Linear(MLP_HIDDEN, out),
        )

    def forward(self, x):
        return self.net(x)


def train_decoder(z_train, y_train, seed, device):
    # Same decoder initialization / dropout RNG for baseline and idea.
    seed_all(seed)
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    criterion = nn.MSELoss()
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    losses = []
    decoder.train()
    for epoch in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(decoder(z), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if (epoch + 1) % 500 == 0 or epoch + 1 == MLP_EPOCHS:
            print(f"Decoder {epoch + 1}/{MLP_EPOCHS}: train MSE={losses[-1]:.6f}", flush=True)
    # No early stopping or parameter selection on validation data.
    return decoder, losses


def predict_decoder(decoder, embeddings):
    decoder.eval()  # Essential: no Dropout noise while measuring R2.
    device = next(decoder.parameters()).device
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
            z = torch.as_tensor(embeddings[start:start + EVAL_BATCH_SIZE],
                                dtype=torch.float32, device=device)
            predictions.append(decoder(z).cpu().numpy())
    return np.concatenate(predictions, axis=0)


def compute_r2(labels, predictions):
    if predictions.shape != labels.shape or not np.isfinite(predictions).all():
        raise ValueError("Invalid decoder prediction shape or nonfinite predictions.")
    per_output = np.atleast_1d(r2_score(labels, predictions, multioutput="raw_values"))
    return float(per_output.mean()), per_output.tolist()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def run_arm(cebra, label, config, seed, arrays, out):
    x_train, x_valid, y_train, y_valid = arrays
    arm_dir = out / f"seed_{seed}" / label
    arm_dir.mkdir(parents=True, exist_ok=False)
    encoder = decoder = z_train = z_valid = None
    started = time.perf_counter()
    try:
        print(f"\nTRAIN {label} | encoder seed={seed}", flush=True)
        seed_all(seed)
        encoder = cebra.CEBRA(**config)
        encoder.fit(x_train, y_train)
        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        for z, x in ((z_train, x_train), (z_valid, x_valid)):
            if len(z) != len(x) or not np.isfinite(z).all():
                raise ValueError("Invalid embeddings or label/embedding misalignment.")
        encoder_seconds = time.perf_counter() - started
        decoder_seed = seed + 100_000
        decoder, losses = train_decoder(z_train, y_train, decoder_seed,
                                         torch.device(encoder.device_))
        predictions = predict_decoder(decoder, z_valid)
        mean_r2, per_output = compute_r2(y_valid, predictions)
        row = dict(arm=label, seed=seed, decoder_seed=decoder_seed,
                   valid_mean_r2=mean_r2, valid_r2_per_output=per_output,
                   encoder_seconds=encoder_seconds,
                   total_seconds=time.perf_counter() - started,
                   encoder_config=config)
        save_json(arm_dir / "metrics.json", row)
        np.save(arm_dir / "decoder_train_mse.npy", np.asarray(losses))
        np.savez_compressed(arm_dir / "validation_predictions.npz",
                            y_true=y_valid, y_pred=predictions)
        if SAVE_CHECKPOINTS:
            encoder.save(str(arm_dir / "cebra.pt"), backend="sklearn")
            torch.save(dict(
                state_dict={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                input_dim=z_train.shape[1], output_dim=y_train.shape[1],
                hidden=MLP_HIDDEN, dropout=MLP_DROP, seed=decoder_seed,
            ), arm_dir / "decoder.pt")
        print(f"{label}: validation mean R2={mean_r2:.6f}; per output={per_output}", flush=True)
        return row
    finally:
        del encoder, decoder, z_train, z_valid
        cleanup()


def summarize(rows, arms, out):
    labels = [name for name, _ in arms]
    with (out / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "arm", "valid_mean_r2"] +
                        [f"valid_r2_output_{i}" for i in range(len(rows[0]["valid_r2_per_output"]))])
        for row in rows:
            writer.writerow([row["seed"], row["arm"], row["valid_mean_r2"],
                             *row["valid_r2_per_output"]])
    by_arm = {label: [r for r in rows if r["arm"] == label] for label in labels}
    scores = {label: np.array([r["valid_mean_r2"] for r in by_arm[label]]) for label in labels}
    deltas = scores[labels[1]] - scores[labels[0]]
    summary = dict(
        evaluation_split="validation", seeds=list(SEEDS),
        by_arm={label: dict(mean=float(values.mean()),
                           std=float(values.std(ddof=1)) if len(values) > 1 else None)
                for label, values in scores.items()},
        paired_delta_idea_minus_baseline=deltas.tolist(),
        mean_paired_delta=float(deltas.mean()),
    )
    save_json(out / "summary.json", summary)
    means = [scores[label].mean() for label in labels]
    errors = [scores[label].std(ddof=1) for label in labels] if len(SEEDS) > 1 else None
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([0, 1], means, yerr=errors, capsize=5, color=["#64748b", "#0d9488"])
    ax.set_xticks([0, 1], labels, rotation=8)
    ax.set_ylabel("Validation mean R2 (uniform average over outputs)")
    ax.set_title(f"{SESSION} | reference / 1 restart / last iterate")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    fig.tight_layout()
    fig.savefig(out / "R2_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_outputs = len(rows[0]["valid_r2_per_output"])
    positions = np.arange(n_outputs)
    for index, label in enumerate(labels):
        values = np.asarray([r["valid_r2_per_output"] for r in by_arm[label]]).mean(axis=0)
        ax.bar(positions + (index - 0.5) * 0.36, values, width=0.36, label=label)
    ax.set_xticks(positions, [str(i) for i in range(n_outputs)])
    ax.set_xlabel("Label output index")
    ax.set_ylabel("Validation R2")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "R2_per_output.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("\nFINAL VALIDATION R2", flush=True)
    for label in labels:
        print(label, summary["by_arm"][label], flush=True)
    print("Paired deltas (idea - baseline):", deltas.tolist(), flush=True)
    print("Mean delta:", summary["mean_paired_delta"], flush=True)
    print("Saved:", out, flush=True)


def main():
    if not SEEDS or len(set(SEEDS)) != len(SEEDS):
        raise ValueError("SEEDS must contain at least one seed, with no duplicates.")
    cebra = load_cebra_fork()
    arrays = load_data()
    arms = experiment_configs(arrays[0].shape[1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "run_config.json", dict(
        fork=str(CEBRA_DIR), imported_cebra=str(cebra.__file__), session=SESSION,
        data_path=str(NPZ_PATH), seeds=list(SEEDS),
        data_shapes=[list(a.shape) for a in arrays], arms=dict(arms),
        decoder=dict(epochs=MLP_EPOCHS, hidden=MLP_HIDDEN, dropout=MLP_DROP,
                     learning_rate=MLP_LR, batch_mode="full", eval_mode=True),
        versions=dict(cebra=cebra.__version__, torch=str(torch.__version__), numpy=np.__version__),
    ))
    print("Arms:", json.dumps(dict(arms), indent=2), flush=True)
    rows = []
    for seed in SEEDS:
        for label, config in arms:
            rows.append(run_arm(cebra, label, config, seed, arrays, out))
            save_json(out / "results_so_far.json", rows)
    summarize(rows, arms, out)


if __name__ == "__main__":
    main()
