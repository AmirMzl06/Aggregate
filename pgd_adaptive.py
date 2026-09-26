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
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "Acorn-adaptive"
PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
SESSION = "C-CO12"
NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"
OUT_ROOT = ROOT / "ACORN_ADAPTIVE_EPS_CCO12"

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
TIME_OFFSETS = 1
LEARNING_RATE = 3e-4
DEVICE = "cuda_if_available"
ADV_STEPS = 10
ATTACK_NORM = "linf"
CONSTANT_EPSILON = 0.5
ADV_ALPHA = 0.1
ADV_ALPHA_SCALE_WITH_EPSILON = False
ADAPTIVE_EPSILON_COEF = 0.2
SEEDS = (42, 43)
SAVE_CHECKPOINTS = True

MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192

EPSILON_MODES = (
    "constant",
    "per_neuron_std",
    "global_std",
    "per_neuron_mad",
    "per_neuron_std_dataset",
    "global_std_dataset",
    "per_neuron_mad_dataset",
)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_cebra_fork():
    if not (CEBRA_DIR / "cebra" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"Expected adaptive fork at: {CEBRA_DIR}\n"
            "Put this runner next to the Acorn-adaptive directory."
        )
    for name in list(sys.modules):
        if name == "cebra" or name.startswith("cebra."):
            del sys.modules[name]
    while str(CEBRA_DIR) in sys.path:
        sys.path.remove(str(CEBRA_DIR))
    sys.path.insert(0, str(CEBRA_DIR))
    import cebra
    imported = Path(cebra.__file__).resolve()
    expected = CEBRA_DIR.resolve()
    if expected not in imported.parents:
        raise RuntimeError(
            f"Wrong CEBRA imported: {imported}\nExpected fork: {expected}"
        )
    params = inspect.signature(cebra.CEBRA.__init__).parameters
    required = (
        "training_mode",
        "adv_epsilon",
        "adv_alpha",
        "adv_steps",
        "attack_norm",
        "adv_epsilon_mode",
        "adv_epsilon_coef",
        "adv_alpha_scale_with_epsilon",
    )
    missing = [name for name in required if name not in params]
    if missing:
        raise RuntimeError(
            "Acorn-adaptive does not expose the required API. "
            f"Missing constructor arguments: {missing}"
        )
    print("Using adaptive fork:", imported, flush=True)
    print("CEBRA version:", getattr(cebra, "__version__", "unknown"), flush=True)
    return cebra

def load_data():
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {NPZ_PATH}")
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        arrays = [
            np.asarray(data[key], dtype=np.float32)
            for key in ("train_data", "valid_data", "train_label", "valid_label")
        ]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    for name, values in zip(
        ("X_train", "X_valid", "Y_train", "Y_valid"),
        (x_train, x_valid, y_train, y_valid),
    ):
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array; got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    if len(x_train) != len(y_train):
        raise ValueError(
            f"Training feature/label lengths differ: {len(x_train)} vs {len(y_train)}."
        )
    if len(x_valid) != len(y_valid):
        raise ValueError(
            f"Validation feature/label lengths differ: {len(x_valid)} vs {len(y_valid)}."
        )
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError("Training and validation neuron counts do not match.")
    if y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Training and validation label dimensions do not match.")
    if len(x_train) < 36 or len(x_valid) < 36:
        raise ValueError("The offset36 encoder requires at least 36 timepoints per split.")
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))

class TwoLayerMLP(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, MLP_HIDDEN),
            nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROP),
            nn.Linear(MLP_HIDDEN, out),
        )
    def forward(self, x):
        return self.net(x)

def train_decoder(z_train, y_train, seed, device):
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
    return decoder, losses

def predict_decoder(decoder, embeddings):
    decoder.eval()
    device = next(decoder.parameters()).device
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
            z = torch.as_tensor(
                embeddings[start:start + EVAL_BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(decoder(z).cpu().numpy())
    return np.concatenate(predictions, axis=0)

def compute_r2(labels, predictions):
    if predictions.shape != labels.shape:
        raise ValueError(f"Prediction/label shape mismatch: {predictions.shape} vs {labels.shape}.")
    if not np.isfinite(predictions).all():
        raise ValueError("Decoder predictions contain NaN/Inf.")
    per_output = np.atleast_1d(r2_score(labels, predictions, multioutput="raw_values"))
    return float(per_output.mean()), per_output.tolist()

def common_encoder_config():
    return dict(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSETS,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        learning_rate=LEARNING_RATE,
        temperature_mode="constant",
        training_mode="adversarial",
        adv_epsilon=CONSTANT_EPSILON,
        adv_alpha=ADV_ALPHA,
        adv_steps=ADV_STEPS,
        attack_norm=ATTACK_NORM,
        adv_alpha_scale_with_epsilon=ADV_ALPHA_SCALE_WITH_EPSILON,
        device=DEVICE,
        verbose=True,
    )

def experiment_configs():
    configs = []
    for mode in EPSILON_MODES:
        cfg = common_encoder_config()
        cfg["adv_epsilon_mode"] = mode
        if mode == "constant":
            cfg["adv_epsilon"] = CONSTANT_EPSILON
            cfg["adv_epsilon_coef"] = ADAPTIVE_EPSILON_COEF
        else:
            cfg["adv_epsilon_coef"] = ADAPTIVE_EPSILON_COEF
            cfg["adv_epsilon"] = CONSTANT_EPSILON
        configs.append((mode, cfg))
    return configs

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

def run_arm(cebra, mode, config, seed, arrays, out):
    (x_train, x_valid, y_train, y_valid) = arrays
    arm_dir = out / f"seed_{seed}" / mode
    arm_dir.mkdir(parents=True, exist_ok=False)
    encoder = None
    decoder = None
    z_train = None
    z_valid = None
    started = time.perf_counter()
    try:
        print(
            "\n" + "=" * 100
            + f"\nSEED {seed} | EPSILON MODE: {mode}\n"
            + "=" * 100,
            flush=True,
        )
        print(f"constant epsilon scalar = {CONSTANT_EPSILON}", flush=True)
        print(f"adaptive epsilon coefficient = {ADAPTIVE_EPSILON_COEF}", flush=True)
        print(
            f"attack = {ATTACK_NORM}, alpha={ADV_ALPHA}, steps={ADV_STEPS}, "
            f"alpha_scale_with_epsilon={ADV_ALPHA_SCALE_WITH_EPSILON}",
            flush=True,
        )
        seed_all(seed)
        encoder = cebra.CEBRA(**config)
        encoder.fit(x_train, y_train)
        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        if len(z_train) != len(x_train):
            raise ValueError(f"Train embedding length mismatch: {len(z_train)} vs {len(x_train)}.")
        if len(z_valid) != len(x_valid):
            raise ValueError(f"Validation embedding length mismatch: {len(z_valid)} vs {len(x_valid)}.")
        if not np.isfinite(z_train).all() or not np.isfinite(z_valid).all():
            raise ValueError("Encoder produced NaN/Inf embeddings.")
        encoder_seconds = time.perf_counter() - started
        decoder_seed = seed + 100_000
        decoder, losses = train_decoder(
            z_train=z_train,
            y_train=y_train,
            seed=decoder_seed,
            device=torch.device(encoder.device_),
        )
        predictions = predict_decoder(decoder, z_valid)
        mean_r2, per_output = compute_r2(y_valid, predictions)
        row = dict(
            session=SESSION,
            seed=seed,
            mode=mode,
            valid_mean_r2=mean_r2,
            valid_r2_per_output=per_output,
            encoder_seconds=encoder_seconds,
            total_seconds=time.perf_counter() - started,
            epsilon_constant=CONSTANT_EPSILON,
            epsilon_coef=ADAPTIVE_EPSILON_COEF,
            adv_alpha=ADV_ALPHA,
            adv_steps=ADV_STEPS,
            attack_norm=ATTACK_NORM,
            adv_alpha_scale_with_epsilon=ADV_ALPHA_SCALE_WITH_EPSILON,
            encoder_config=config,
        )
        save_json(arm_dir / "metrics.json", row)
        np.save(arm_dir / "decoder_train_mse.npy", np.asarray(losses, dtype=np.float32))
        np.savez_compressed(
            arm_dir / "validation_predictions.npz",
            y_true=y_valid,
            y_pred=predictions,
        )
        if SAVE_CHECKPOINTS:
            encoder.save(str(arm_dir / "cebra.pt"), backend="sklearn")
            torch.save(
                dict(
                    state_dict={k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                    input_dim=int(z_train.shape[1]),
                    output_dim=int(y_train.shape[1]),
                    hidden=MLP_HIDDEN,
                    dropout=MLP_DROP,
                    seed=decoder_seed,
                ),
                arm_dir / "decoder.pt",
            )
        print(f"{mode}: validation mean R2={mean_r2:.6f}", flush=True)
        print(f"{mode}: per-output R2={per_output}", flush=True)
        return row
    finally:
        del encoder
        del decoder
        del z_train
        del z_valid
        cleanup()

def summarize(rows, configs, out):
    mode_names = [mode for mode, _ in configs]
    n_outputs = len(rows[0]["valid_r2_per_output"])
    csv_path = out / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "session",
                "seed",
                "adv_epsilon_mode",
                "valid_mean_r2",
                "epsilon_constant",
                "epsilon_coef",
                "adv_alpha",
                "adv_steps",
                "attack_norm",
                "adv_alpha_scale_with_epsilon",
            ] + [f"valid_r2_output_{i}" for i in range(n_outputs)]
        )
        for row in rows:
            writer.writerow(
                [
                    row["session"],
                    row["seed"],
                    row["mode"],
                    row["valid_mean_r2"],
                    row["epsilon_constant"],
                    row["epsilon_coef"],
                    row["adv_alpha"],
                    row["adv_steps"],
                    row["attack_norm"],
                    row["adv_alpha_scale_with_epsilon"],
                ] + row["valid_r2_per_output"]
            )
    by_mode = {}
    for mode in mode_names:
        mode_rows = [r for r in rows if r["mode"] == mode]
        vals = np.asarray([r["valid_mean_r2"] for r in mode_rows], dtype=float)
        per_output = np.asarray([r["valid_r2_per_output"] for r in mode_rows], dtype=float)
        by_mode[mode] = dict(
            mean=float(vals.mean()),
            std=float(vals.std(ddof=1)) if len(vals) > 1 else None,
            per_seed={str(r["seed"]): float(r["valid_mean_r2"]) for r in mode_rows},
            mean_r2_per_output=per_output.mean(axis=0).tolist(),
        )
    summary = dict(
        session=SESSION,
        seeds=list(SEEDS),
        evaluation_split="valid_data",
        modes=mode_names,
        constant_epsilon=CONSTANT_EPSILON,
        adaptive_epsilon_coef=ADAPTIVE_EPSILON_COEF,
        attack_norm=ATTACK_NORM,
        adv_alpha=ADV_ALPHA,
        adv_steps=ADV_STEPS,
        adv_alpha_scale_with_epsilon=ADV_ALPHA_SCALE_WITH_EPSILON,
        by_mode=by_mode,
    )
    save_json(out / "summary.json", summary)
    print("\n" + "=" * 100, flush=True)
    print("FINAL VALIDATION R2 SUMMARY", flush=True)
    print("=" * 100, flush=True)
    for mode in mode_names:
        item = by_mode[mode]
        print(
            f"{mode:<26} mean={item['mean']:.6f} std={item['std']} "
            f"per_seed={item['per_seed']}",
            flush=True,
        )
    print("\nSaved:", csv_path, flush=True)
    print("Saved:", out / "summary.json", flush=True)

def main():
    if len(SEEDS) != 2:
        raise ValueError(f"Expected exactly 2 seeds, got {SEEDS}.")
    if len(set(SEEDS)) != len(SEEDS):
        raise ValueError("SEEDS contains duplicates.")
    cebra = load_cebra_fork()
    arrays = load_data()
    (x_train, x_valid, y_train, y_valid) = arrays
    print("\nDATA", flush=True)
    print("session:", SESSION, flush=True)
    print("X_train:", x_train.shape, flush=True)
    print("X_valid:", x_valid.shape, flush=True)
    print("Y_train:", y_train.shape, flush=True)
    print("Y_valid:", y_valid.shape, flush=True)
    configs = experiment_configs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = OUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=False)
    save_json(
        out / "run_config.json",
        dict(
            fork=str(CEBRA_DIR),
            imported_cebra=str(cebra.__file__),
            session=SESSION,
            data_path=str(NPZ_PATH),
            seeds=list(SEEDS),
            data_shapes=[list(a.shape) for a in arrays],
            epsilon_modes=list(EPSILON_MODES),
            constant_epsilon=CONSTANT_EPSILON,
            adaptive_epsilon_coef=ADAPTIVE_EPSILON_COEF,
            attack_norm=ATTACK_NORM,
            adv_alpha=ADV_ALPHA,
            adv_steps=ADV_STEPS,
            adv_alpha_scale_with_epsilon=ADV_ALPHA_SCALE_WITH_EPSILON,
            encoder_common=dict(
                latent_dim=LATENT_DIM,
                hidden=HIDDEN,
                batch_size=BATCH_SIZE,
                max_iterations=MAX_ITER,
                temperature=TEMPERATURE,
                model_architecture=MODEL_ARCH,
                time_offsets=TIME_OFFSETS,
                learning_rate=LEARNING_RATE,
                training_mode="adversarial",
            ),
            decoder=dict(
                epochs=MLP_EPOCHS,
                hidden=MLP_HIDDEN,
                dropout=MLP_DROP,
                learning_rate=MLP_LR,
                batch_mode="full",
                eval_mode=True,
            ),
            configs={mode: cfg for mode, cfg in configs},
        ),
    )
    print("\nOUTPUT:", out, flush=True)
    print("\nMODES:", flush=True)
    for mode, cfg in configs:
        print(
            f"  {mode}: eps={cfg['adv_epsilon']}, coef={cfg['adv_epsilon_coef']}, "
            f"alpha={cfg['adv_alpha']}, norm={cfg['attack_norm']}",
            flush=True,
        )
    rows = []
    for seed in SEEDS:
        for mode, config in configs:
            row = run_arm(
                cebra=cebra,
                mode=mode,
                config=config,
                seed=seed,
                arrays=arrays,
                out=out,
            )
            rows.append(row)
            save_json(out / "results_so_far.json", rows)
    summarize(rows=rows, configs=configs, out=out)

if __name__ == "__main__":
    main()
