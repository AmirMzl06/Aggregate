"""
Multi-session generalization of the C-CO12 NORMAL vs PNHARD vs ACORN script.

For every monkey+task group in PERICH_DAYS, randomly pick up to 5 target
sessions (fewer if the group doesn't have 5). For EACH target session:
  - build a foreign neuron pool from the REST of that SAME group's sessions
    (capped to whatever is available, warns if short of 860)
  - train CLEAN CEBRA + PNHARD CEBRA (extra_negative_fraction=0.5,
    extra_negative_candidate_multiplier=1, 30_000 iterations, behavior labels)
  - train ACORN (adversarial, epsilon=0.5, 3_000 iterations, behavior labels)
  - decoder + test R2 for all three models
  - forward Jacobian (.npy + heatmap .png) for all three models
  - NO embedding/PCA plots this time

Outputs:
  OUT_DIR/{group}/{session}/models/{normal,pnhard,acorn}.pt
  OUT_DIR/{group}/{session}/decoder/{normal,pnhard,acorn}_decoder.pt
  OUT_DIR/{group}/{session}/jacobian/{normal,pnhard,acorn}_train_jf.npy + .png
  OUT_DIR/{group}/{session}/foreign_pool_mapping.csv
  OUT_DIR/all_sessions_r2_summary.csv   <- rewritten after every session
"""

from __future__ import annotations

import csv
import gc
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# PATHS / FORK SWITCHING
# =============================================================================

ROOT = Path(__file__).resolve().parent
PNHARD_CEBRA_DIR = ROOT / "CEBRA-NPHard"

if not PNHARD_CEBRA_DIR.exists():
    raise FileNotFoundError(f"CEBRA-PNHard fork not found: {PNHARD_CEBRA_DIR}")

from utils.constants import CEBRA_DIR as ACORN_CEBRA_DIR
ACORN_CEBRA_DIR = Path(ACORN_CEBRA_DIR).resolve()

# these get reassigned by switch_to_pnhard_fork()/switch_to_acorn_fork() below;
# every function that talks to CEBRA (build_*, compute_train_jacobian, ...)
# reads these module-level names, so it always uses whichever fork was
# switched to most recently.
cebra = None
CEBRA = None


def clear_cebra_modules():
    for name in list(sys.modules):
        if name == "cebra" or name.startswith("cebra."):
            del sys.modules[name]
    importlib.invalidate_caches()


def import_pnhard_cebra():
    clear_cebra_modules()
    for p in (str(PNHARD_CEBRA_DIR), str(ACORN_CEBRA_DIR)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(PNHARD_CEBRA_DIR))
    try:
        import cebra as _cebra
        from cebra import CEBRA as _CEBRA
        import cebra.attribution  # noqa: F401
    except Exception as exc:
        raise SystemExit(f"cannot import CEBRA-NPHard from {PNHARD_CEBRA_DIR}. error: {exc}")

    params = inspect.signature(_CEBRA.__init__).parameters
    required = {
        "extra_negatives", "extra_negative_fraction",
        "extra_negative_candidate_multiplier", "extra_negative_normalize",
    }
    missing = required.difference(params)
    if missing:
        raise RuntimeError(f"Wrong fork loaded. Missing PNHard args: {sorted(missing)}")
    return _cebra, _CEBRA


def import_acorn_cebra():
    clear_cebra_modules()
    for p in (str(PNHARD_CEBRA_DIR), str(ACORN_CEBRA_DIR)):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, str(ACORN_CEBRA_DIR))
    import cebra as _cebra
    from cebra import CEBRA as _CEBRA
    import cebra.attribution

    params = inspect.signature(_CEBRA.__init__).parameters
    required = {"training_mode", "adv_epsilon", "adv_alpha", "adv_steps", "attack_norm"}
    missing = required.difference(params)
    if missing:
        raise RuntimeError(f"ACORN API missing: {sorted(missing)}")
    return _cebra, _CEBRA


def switch_to_pnhard_fork():
    global cebra, CEBRA
    cebra, CEBRA = import_pnhard_cebra()
    print("Using CEBRA-NPHard:", cebra.__file__)


def switch_to_acorn_fork():
    global cebra, CEBRA
    cebra, CEBRA = import_acorn_cebra()
    print("Using standard ACORN CEBRA:", cebra.__file__)


# =============================================================================
# CONFIG
# =============================================================================

PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")

PERICH_DAYS = {
    "C-CO": range(53),
    "C-RT": range(15),
    "J-CO": range(3),
    "T-CO": range(6),
    "T-RT": range(6),
    "M-CO": range(22),
    "M-RT": range(6),
}

N_TARGETS_PER_GROUP = 5       # capped to len(group) if the group is smaller
FOREIGN_POOL_NEURONS = 860    # target; capped to whatever's available, with a warning

SEED = 42

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"
OFFSET = 1
CONDITIONAL = "time_delta"

CLEAN_NPHARD_MAX_ITER = 30_000
EXTRA_NEGATIVE_FRACTION = 0.5
PNHARD_CANDIDATE_MULTIPLIER = 1   # =1 -> effectively "negpos" (no oversampling of
                                   # candidates), but WHICH foreign neurons fill the
                                   # neg-pair slot still changes every iteration.
PNHARD_NORMALIZE = True

DECODER_HIDDEN_DIM = 64
DECODER_DROPOUT = 0.4
DECODER_EPOCHS = 2500
DECODER_BATCH_SIZE = 256
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_PRINT_EVERY = 500

ATTR_CHUNKS = 16
ATTR_LEN = 128
ATTR_BATCH = 16

ACORN_MAX_ITER = 3_000
ACORN_EPSILON = 0.5
ACORN_ALPHA = ACORN_EPSILON / 5.0
ACORN_STEPS = 10
ACORN_ATTACK_NORM = "linf"

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

OUT_DIR = ROOT / "CEBRA_PNHard_AllGroups_NegPos05"
SUMMARY_CSV = OUT_DIR / "all_sessions_r2_summary.csv"


# =============================================================================
# BASIC HELPERS
# =============================================================================

def seed_all(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def session_path(name: str) -> Path:
    return PERICH_DATA_DIR / f"{name}.npz"


# =============================================================================
# DATA -- one group loaded/cached once, reused for every target in that group
# =============================================================================

def load_group_sessions(group: str, day_range) -> Dict[int, dict]:
    """Loads every usable session of `group` ONCE. Each entry has X_train,
    X_test, Y_train, Y_test -- used both as a potential TARGET session and
    as a source of foreign neurons for other targets in the same group."""
    sessions: Dict[int, dict] = {}
    for day in day_range:
        name = f"{group}{day}"
        path = session_path(name)
        if not path.exists():
            print(f"[SKIP missing] {name}")
            continue
        try:
            d = np.load(path, allow_pickle=True)
            X_train = np.asarray(d["train_data"], dtype=np.float32)
            X_test = np.asarray(d["valid_data"], dtype=np.float32)
            Y_train = np.asarray(d["train_label"], dtype=np.float32)
            Y_test = np.asarray(d["valid_label"], dtype=np.float32)
            if Y_train.ndim == 1:
                Y_train = Y_train[:, None]
            if Y_test.ndim == 1:
                Y_test = Y_test[:, None]
            if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
                raise ValueError(f"bad shapes {X_train.shape} {X_test.shape}")
            if X_train.shape[1] == 0:
                raise ValueError("0 neurons")
            for nm, arr in (("X_train", X_train), ("X_test", X_test),
                            ("Y_train", Y_train), ("Y_test", Y_test)):
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"{nm} contains NaN/Inf")
        except Exception as e:
            print(f"[SKIP bad] {name}: {e}")
            continue

        sessions[day] = dict(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test)
        print(f"[OK] {name}: train={X_train.shape} test={X_test.shape} "
              f"Y_train={Y_train.shape}")

    return sessions


# =============================================================================
# FOREIGN POOL (from the REST of the same group)
# =============================================================================

ForeignNeuron = Tuple[int, int]


def choose_random_foreign_neurons(other_sessions: Dict[int, dict], n_select: int,
                                   seed: int) -> List[ForeignNeuron]:
    pool: List[ForeignNeuron] = []
    for day, s in other_sessions.items():
        pool.extend((day, j) for j in range(s["X_train"].shape[1]))

    if len(pool) == 0:
        return []
    if len(pool) < n_select:
        print(f"[WARN] only {len(pool)} foreign neurons available "
              f"(< requested {n_select}) -- using all of them.")
        n_select = len(pool)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n_select, replace=False)
    return [pool[int(i)] for i in idx]


def save_mapping(mapping: Sequence[ForeignNeuron], path: Path, coordinate_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([coordinate_name, "foreign_session_day", "foreign_neuron_index"])
        for j, (day, neuron_idx) in enumerate(mapping):
            w.writerow([j, day, neuron_idx])
    print("saved:", path)


def build_foreign_matrix(other_sessions: Dict[int, dict], mapping: Sequence[ForeignNeuron],
                         split: str) -> np.ndarray:
    if split not in ("train", "test"):
        raise ValueError("split must be train/test.")
    key = "X_train" if split == "train" else "X_test"

    traces = [other_sessions[day][key][:, neuron_idx].astype(np.float32)
              for day, neuron_idx in mapping]
    T_min = min(len(x) for x in traces)
    X = np.column_stack([x[:T_min] for x in traces]).astype(np.float32)

    n_sessions = len(set(day for day, _ in mapping))
    print(f"foreign_{split}_raw={X.shape} | T_min={T_min} | "
          f"neurons={len(mapping)} | sessions={n_sessions}")
    return X


# =============================================================================
# CEBRA BUILDERS (use whichever fork is currently active via `CEBRA`)
# =============================================================================

def common_cebra_kwargs(max_iterations):
    return dict(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        conditional=CONDITIONAL,
        max_iterations=max_iterations,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        device=DEVICE,
        verbose=True,
    )


def build_normal_cebra():
    return CEBRA(**common_cebra_kwargs(CLEAN_NPHARD_MAX_ITER))


def build_pnhard_cebra(foreign_raw):
    return CEBRA(
        **common_cebra_kwargs(CLEAN_NPHARD_MAX_ITER),
        extra_negatives=foreign_raw,
        extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
        extra_negative_candidate_multiplier=PNHARD_CANDIDATE_MULTIPLIER,
        extra_negative_normalize=PNHARD_NORMALIZE,
    )


def build_acorn_cebra():
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        conditional=CONDITIONAL,
        max_iterations=ACORN_MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="adversarial",
        adv_epsilon=ACORN_EPSILON,
        adv_alpha=ACORN_ALPHA,
        adv_steps=ACORN_STEPS,
        attack_norm=ACORN_ATTACK_NORM,
        device=DEVICE,
        verbose=True,
    )


def save_model(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    print("saved model:", path)


def embed(model, X):
    return np.asarray(model.transform(X.astype(np.float32)), dtype=np.float32)


# =============================================================================
# DECODER / R2
# =============================================================================

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


def align_embedding_labels(Z, Y):
    n = min(len(Z), len(Y))
    return Z[:n], Y[:n]


def train_decoder_and_r2(display, Z_train, Z_test, Y_train, Y_test, save_path: Path):
    Z_train, Ytr = align_embedding_labels(Z_train, Y_train)
    Z_test, Yte = align_embedding_labels(Z_test, Y_test)
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    if Yte.ndim == 1:
        Yte = Yte[:, None]

    seed_all(SEED)
    decoder = TwoLayerMLP(
        input_dim=Z_train.shape[1], hidden_dim=DECODER_HIDDEN_DIM,
        output_dim=Ytr.shape[1], dropout_rate=DECODER_DROPOUT,
    ).to(TORCH_DEVICE)

    ds = TensorDataset(torch.from_numpy(Z_train.astype(np.float32)),
                       torch.from_numpy(Ytr.astype(np.float32)))
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(ds, batch_size=DECODER_BATCH_SIZE, shuffle=True,
                        drop_last=False, num_workers=0, generator=g)

    opt = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    print(f"\nDecoder {display}: {DECODER_EPOCHS} epochs")
    for epoch in range(1, DECODER_EPOCHS + 1):
        decoder.train()
        for xb, yb in loader:
            xb, yb = xb.to(TORCH_DEVICE), yb.to(TORCH_DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(decoder(xb), yb)
            loss.backward()
            opt.step()
        if epoch == 1 or epoch % DECODER_PRINT_EVERY == 0 or epoch == DECODER_EPOCHS:
            print(f"{display}: decoder epoch {epoch}/{DECODER_EPOCHS}")

    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.from_numpy(Z_test.astype(np.float32)).to(TORCH_DEVICE)).cpu().numpy()

    r2_each = np.asarray(r2_score(Yte, pred, multioutput="raw_values"), dtype=float)
    mean_r2 = float(np.mean(r2_each))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decoder.state_dict(), save_path)

    print(f"{display} TEST R2 per target:", r2_each)
    print(f"{display} TEST Mean R2 = {mean_r2:.6f}")
    return r2_each, mean_r2


# =============================================================================
# JACOBIAN
# =============================================================================

def compute_train_jacobian(model, X_train, display):
    print(f"\nComputing CEBRA Jacobian: {display}")
    net = model.solver_.model
    if isinstance(net, nn.ModuleList):
        if len(net) != 1:
            raise RuntimeError("Expected single-session encoder.")
        net = net[0]

    device = next(net.parameters()).device
    net.eval()
    n_neurons = X_train.shape[1]

    if len(X_train) <= ATTR_LEN + 1:
        print(f"[WARN] {display}: only {len(X_train)} train samples, "
              f"< ATTR_LEN+1={ATTR_LEN+1} -- skipping Jacobian.")
        return None

    starts = np.linspace(0, len(X_train) - ATTR_LEN - 1, ATTR_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
    total = 0

    for ci, start in enumerate(starts, 1):
        chunk = X_train[start:start + ATTR_LEN]
        inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)

        method = cebra.attribution.init(
            name="jacobian-based-batched", model=net, input_data=inp,
            output_dimension=LATENT_DIM,
        )
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH)

        jf = np.abs(np.asarray(result["jf"]))
        jf = np.squeeze(jf)
        if jf.shape != (LATENT_DIM, n_neurons):
            if jf.ndim < 2:
                raise RuntimeError(f"Unexpected jf shape: {jf.shape}")
            jf = np.mean(jf, axis=tuple(range(jf.ndim - 2)))
        if jf.shape != (LATENT_DIM, n_neurons):
            raise RuntimeError(f"Expected {(LATENT_DIM, n_neurons)}, got {jf.shape}")

        jf_sum += jf.astype(np.float64) * len(chunk)
        total += len(chunk)
        print(f"{display}: Jacobian chunk {ci}/{len(starts)}")

        del inp, method, result, jf
        cleanup()

    return (jf_sum / total).astype(np.float32)


def save_jacobian(J, jacobian_dir: Path, tag: str):
    if J is None:
        return
    jacobian_dir.mkdir(parents=True, exist_ok=True)
    np.save(jacobian_dir / f"{tag}_train_jf.npy", J)

    plt.figure(figsize=(12, 8))
    plt.imshow(J, aspect="auto")
    plt.colorbar(label="Absolute forward Jacobian")
    plt.xlabel("Input neuron")
    plt.ylabel("Latent dimension")
    plt.title(f"{tag.upper()} -- Train Forward Jacobian")
    plt.tight_layout()
    path = jacobian_dir / f"{tag}_train_jacobian.png"
    plt.savefig(path, dpi=250, bbox_inches="tight")
    plt.close()
    print("saved:", path)


# =============================================================================
# ONE TARGET SESSION -- full pipeline
# =============================================================================

def run_one_target_session(group: str, target_day: int, group_sessions: Dict[int, dict]):
    session_name = f"{group}{target_day}"
    session_dir = OUT_DIR / group / session_name
    models_dir = session_dir / "models"
    decoder_dir = session_dir / "decoder"
    jacobian_dir = session_dir / "jacobian"

    print("\n" + "=" * 100)
    print(f"TARGET SESSION: {session_name}  (group={group})")
    print("=" * 100)

    target = group_sessions[target_day]
    X_train, X_test = target["X_train"], target["X_test"]
    Y_train, Y_test = target["Y_train"], target["Y_test"]
    print(f"X_train={X_train.shape} X_test={X_test.shape} "
          f"Y_train={Y_train.shape} Y_test={Y_test.shape}")

    other_sessions = {d: s for d, s in group_sessions.items() if d != target_day}
    print(f"other sessions available in {group}: {len(other_sessions)}")

    rows = []

    mapping = choose_random_foreign_neurons(
        other_sessions, n_select=FOREIGN_POOL_NEURONS, seed=SEED + target_day)

    if len(mapping) == 0:
        print(f"[WARN] {session_name}: no foreign neurons available at all "
              f"-- skipping PNHARD for this session (CLEAN + ACORN still run).")
        foreign_train_raw = None
    else:
        save_mapping(mapping, session_dir / "foreign_pool_mapping.csv",
                    "foreign_pool_coordinate")
        foreign_train_raw = build_foreign_matrix(other_sessions, mapping, split="train")

    # ---------------- PNHard fork: NORMAL + PNHARD ----------------
    switch_to_pnhard_fork()

    seed_all(SEED)
    normal = build_normal_cebra()
    normal.fit(X_train, Y_train)
    save_model(normal, models_dir / "normal_cebra.pt")
    normal_r2, normal_mean = train_decoder_and_r2(
        "NORMAL", embed(normal, X_train), embed(normal, X_test),
        Y_train, Y_test, decoder_dir / "normal_decoder.pt")
    normal_J = compute_train_jacobian(normal, X_train, "NORMAL")
    save_jacobian(normal_J, jacobian_dir, "normal")
    rows.append(dict(group=group, session=session_name, model="NORMAL",
                     mean_r2=normal_mean, r2_per_target=normal_r2.tolist(), status="ok"))
    del normal
    cleanup()

    if foreign_train_raw is not None:
        seed_all(SEED)
        pnhard = build_pnhard_cebra(foreign_train_raw)
        pnhard.fit(X_train, Y_train)
        save_model(pnhard, models_dir / "pnhard_cebra.pt")
        pnhard_r2, pnhard_mean = train_decoder_and_r2(
            "PNHARD", embed(pnhard, X_train), embed(pnhard, X_test),
            Y_train, Y_test, decoder_dir / "pnhard_decoder.pt")
        pnhard_J = compute_train_jacobian(pnhard, X_train, "PNHARD")
        save_jacobian(pnhard_J, jacobian_dir, "pnhard")
        rows.append(dict(group=group, session=session_name, model="PNHARD",
                         mean_r2=pnhard_mean, r2_per_target=pnhard_r2.tolist(), status="ok"))
        del pnhard
        cleanup()
    else:
        rows.append(dict(group=group, session=session_name, model="PNHARD",
                         mean_r2=float("nan"), r2_per_target=[], status="skipped_no_foreign_pool"))

    # ---------------- ACORN fork ----------------
    switch_to_acorn_fork()

    seed_all(SEED)
    acorn = build_acorn_cebra()
    print(f"ACORN | eps={ACORN_EPSILON} | alpha={ACORN_ALPHA} | "
          f"steps={ACORN_STEPS} | iters={ACORN_MAX_ITER} | norm={ACORN_ATTACK_NORM}")
    acorn.fit(X_train, Y_train)
    save_model(acorn, models_dir / "acorn.pt")
    acorn_r2, acorn_mean = train_decoder_and_r2(
        "ACORN", embed(acorn, X_train), embed(acorn, X_test),
        Y_train, Y_test, decoder_dir / "acorn_decoder.pt")
    acorn_J = compute_train_jacobian(acorn, X_train, "ACORN")
    save_jacobian(acorn_J, jacobian_dir, "acorn")
    rows.append(dict(group=group, session=session_name, model="ACORN",
                     mean_r2=acorn_mean, r2_per_target=acorn_r2.tolist(), status="ok"))
    del acorn
    cleanup()

    return rows


# =============================================================================
# SUMMARY CSV (rewritten after every session)
# =============================================================================

def write_summary(all_rows: List[dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    max_targets = max((len(r["r2_per_target"]) for r in all_rows), default=0)
    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "session", "model", "status", "mean_r2"]
                   + [f"r2_target_{i}" for i in range(max_targets)])
        for r in all_rows:
            padded = list(r["r2_per_target"]) + [""] * (max_targets - len(r["r2_per_target"]))
            w.writerow([r["group"], r["session"], r["model"], r["status"], r["mean_r2"]] + padded)
    print("updated summary:", SUMMARY_CSV)


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)

    all_rows: List[dict] = []

    for group_idx, (group, day_range) in enumerate(PERICH_DAYS.items()):
        print("\n" + "#" * 110)
        print(f"GROUP: {group}")
        print("#" * 110)

        group_sessions = load_group_sessions(group, day_range)
        if len(group_sessions) < 2:
            print(f"[SKIP GROUP] {group}: fewer than 2 usable sessions, "
                  f"can't build any foreign pool at all.")
            continue

        available_days = sorted(group_sessions.keys())
        n_targets = min(N_TARGETS_PER_GROUP, len(available_days))
        rng = np.random.default_rng(SEED + group_idx)
        target_days = sorted(rng.choice(available_days, size=n_targets, replace=False).tolist())
        print(f"{group}: {len(available_days)} usable sessions -> "
              f"{n_targets} random target(s): {[f'{group}{d}' for d in target_days]}")

        for target_day in target_days:
            try:
                rows = run_one_target_session(group, target_day, group_sessions)
                all_rows.extend(rows)
            except Exception as e:
                print(f"[ERROR] {group}{target_day} failed: {e}")
                print("Continuing with the next session...")
                cleanup()
                continue

            write_summary(all_rows)  # crash-safe: rewritten after every session

    print("\n" + "#" * 110)
    print("ALL GROUPS DONE")
    print("#" * 110)
    print("summary:", SUMMARY_CSV)


if __name__ == "__main__":
    main()








# """
# C-CO12: NORMAL CEBRA vs PNHARD CEBRA + final ACORN.

# NORMAL + PNHARD:
#   * behavior CEBRA: fit(X_train, Y_train)
#   * PNHARD fork: ./CEBRA-PNHard
#   * raw 860-neuron foreign pool
#   * every PNHard step randomly selects 86 foreign columns
#   * extra_negative_normalize=True
#   * extra_negative_fraction=0.10
#   * extra_negative_candidate_multiplier=3
#   * embedding tests:
#       - C-CO12 test vs fake
#       - C-CO12 test vs cross-session
#   * same TwoLayerMLP decoder, test R2
#   * CEBRA attribution Jacobian on C-CO12 train

# Then:
#   * standard ACORN, epsilon=0.5, behavior labels
#   * same decoder, test R2

# Final line section prints R2 for all three models side by side.
# """

# from __future__ import annotations

# import csv
# import gc
# import importlib
# import inspect
# import random
# import sys
# from pathlib import Path
# from typing import Dict, List, Sequence, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.decomposition import PCA
# from sklearn.metrics import r2_score

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# # =============================================================================
# # PATHS / IMPORTS
# # =============================================================================

# ROOT = Path(__file__).resolve().parent
# PNHARD_CEBRA_DIR = ROOT / "CEBRA-NPHard"

# if not PNHARD_CEBRA_DIR.exists():
#     raise FileNotFoundError(f"CEBRA-PNHard fork not found: {PNHARD_CEBRA_DIR}")

# from utils.constants import CEBRA_DIR as ACORN_CEBRA_DIR
# ACORN_CEBRA_DIR = Path(ACORN_CEBRA_DIR).resolve()


# def clear_cebra_modules():
#     for name in list(sys.modules):
#         if name == "cebra" or name.startswith("cebra."):
#             del sys.modules[name]
#     importlib.invalidate_caches()

# def import_pnhard_cebra():
#     clear_cebra_modules()
#     for p in (
#         str(PNHARD_CEBRA_DIR),
#         str(ACORN_CEBRA_DIR),
#     ):
#         while p in sys.path:
#             sys.path.remove(p)
#     sys.path.insert(
#         0,
#         str(PNHARD_CEBRA_DIR),
#     )
#     try:
#         import cebra
#         from cebra import CEBRA
#         import cebra.attribution
#     except Exception as exc:
#         raise SystemExit(
#             f"cannot import CEBRA-NPHard from "
#             f"{PNHARD_CEBRA_DIR}. error: {exc}"
#         )

#     print("\nUsing CEBRA-NPHard:")
#     print(cebra.__file__)

#     params = inspect.signature(
#         CEBRA.__init__
#     ).parameters
#     required = {
#         "extra_negatives",
#         "extra_negative_fraction",
#         "extra_negative_candidate_multiplier",
#         "extra_negative_normalize",
#     }
#     missing = required.difference(params)
#     if missing:
#         raise RuntimeError(
#             "Wrong fork loaded. Missing PNHard args: "
#             f"{sorted(missing)}"
#         )
#     print("PNHard import/API check: OK")
#     return cebra, CEBRA

# cebra, CEBRA = import_pnhard_cebra()

# # =============================================================================
# # CONFIG
# # =============================================================================

# PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# DATASET_NAME = "C-CO"
# TARGET_DAY = 12
# TARGET_SESSION = f"{DATASET_NAME}{TARGET_DAY}"

# N_NEURONS = 86
# N_CCO_SESSIONS = 53
# FOREIGN_POOL_NEURONS = 860
# CROSS_TEST_NEURONS = 86

# SEED = 42

# LATENT_DIM = 64
# HIDDEN = 64
# BATCH_SIZE = 2048
# MAX_ITER = 3000
# TEMPERATURE = 0.4
# MODEL_ARCH = "offset36-model-more-dropout"
# DEVICE = "cuda_if_available"
# OFFSET = 1
# CONDITIONAL = "time_delta"

# EXTRA_NEGATIVE_FRACTION = 0.85
# PNHARD_CANDIDATE_MULTIPLIER = 1#3
# PNHARD_NORMALIZE = True

# DECODER_HIDDEN_DIM = 64
# DECODER_DROPOUT = 0.4
# DECODER_EPOCHS = 2500
# DECODER_BATCH_SIZE = 256
# DECODER_LR = 1e-3
# DECODER_WEIGHT_DECAY = 1e-4
# DECODER_PRINT_EVERY = 500

# ATTR_CHUNKS = 16
# ATTR_LEN = 128
# ATTR_BATCH = 16

# ACORN_EPSILON = 0.5
# ACORN_ALPHA = ACORN_EPSILON / 5.0
# ACORN_STEPS = 10
# ACORN_ATTACK_NORM = "linf"

# TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# EPS = 1e-8

# OUT_DIR = ROOT / "CEBRA_CCO12_Behavior_PNHard_860"
# MODELS_DIR = OUT_DIR / "models"
# EMB_DIR = OUT_DIR / "embeddings"
# PLOTS_DIR = OUT_DIR / "plots"
# DECODER_DIR = OUT_DIR / "decoder"
# JACOBIAN_DIR = OUT_DIR / "jacobian"

# FOREIGN_MAP_CSV = OUT_DIR / "foreign_860_mapping.csv"
# CROSS_TEST_MAP_CSV = OUT_DIR / "cross_test_86_mapping.csv"
# R2_CSV = DECODER_DIR / "all_three_models_r2.csv"


# # =============================================================================
# # BASIC HELPERS
# # =============================================================================

# def ensure_dirs():
#     for d in (OUT_DIR, MODELS_DIR, EMB_DIR, PLOTS_DIR, DECODER_DIR, JACOBIAN_DIR):
#         d.mkdir(parents=True, exist_ok=True)


# def seed_all(seed=SEED):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def session_path(name: str) -> Path:
#     return PERICH_DATA_DIR / f"{name}.npz"


# # =============================================================================
# # DATA
# # =============================================================================

# def load_target_session():
#     path = session_path(TARGET_SESSION)
#     if not path.exists():
#         raise FileNotFoundError(path)

#     d = np.load(path, allow_pickle=True)
#     X_train = np.asarray(d["train_data"], dtype=np.float32)
#     X_test = np.asarray(d["valid_data"], dtype=np.float32)
#     Y_train = np.asarray(d["train_label"], dtype=np.float32)
#     Y_test = np.asarray(d["valid_label"], dtype=np.float32)

#     if Y_train.ndim == 1:
#         Y_train = Y_train[:, None]
#     if Y_test.ndim == 1:
#         Y_test = Y_test[:, None]

#     if X_train.shape[1] != N_NEURONS:
#         raise ValueError(f"Expected {N_NEURONS} neurons, got {X_train.shape[1]}")
#     if len(X_train) != len(Y_train) or len(X_test) != len(Y_test):
#         raise ValueError("Neural/label length mismatch.")

#     for name, arr in (
#         ("X_train", X_train), ("X_test", X_test),
#         ("Y_train", Y_train), ("Y_test", Y_test),
#     ):
#         if not np.isfinite(arr).all():
#             raise RuntimeError(f"{name} contains NaN/Inf.")

#     print("\nC-CO12")
#     print("X_train:", X_train.shape)
#     print("X_test :", X_test.shape)
#     print("Y_train:", Y_train.shape)
#     print("Y_test :", Y_test.shape)

#     return X_train, X_test, Y_train, Y_test


# def load_other_session(name: str):
#     d = np.load(session_path(name), allow_pickle=True)
#     tr = np.asarray(d["train_data"], dtype=np.float32)
#     te = np.asarray(d["valid_data"], dtype=np.float32)
#     if tr.ndim != 2 or te.ndim != 2 or tr.shape[1] != te.shape[1]:
#         raise ValueError(f"Bad shapes in {name}: {tr.shape}, {te.shape}")
#     if not np.isfinite(tr).all() or not np.isfinite(te).all():
#         raise RuntimeError(f"{name} contains NaN/Inf.")
#     return tr, te


# def load_other_sessions():
#     sessions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
#     for day in range(N_CCO_SESSIONS):
#         if day == TARGET_DAY:
#             continue
#         name = f"{DATASET_NAME}{day}"
#         path = session_path(name)
#         if not path.exists():
#             print(f"[SKIP missing] {name}")
#             continue
#         try:
#             tr, te = load_other_session(name)
#         except Exception as e:
#             print(f"[SKIP bad] {name}: {e}")
#             continue
#         if tr.shape[1] == 0:
#             continue
#         sessions[day] = (tr, te)
#         print(f"[OTHER OK] {name}: train={tr.shape}, test={te.shape}")
#     if not sessions:
#         raise RuntimeError("No usable other C-CO sessions.")
#     return sessions


# # =============================================================================
# # RAW 860-NEURON FOREIGN POOL
# # =============================================================================

# ForeignNeuron = Tuple[int, int]


# def choose_random_foreign_neurons(
#     sessions,
#     n_select: int,
#     seed: int,
# ) -> List[ForeignNeuron]:
#     pool: List[ForeignNeuron] = []
#     for day, (tr, _) in sessions.items():
#         pool.extend((day, j) for j in range(tr.shape[1]))

#     if len(pool) < n_select:
#         raise RuntimeError(
#             f"Only {len(pool)} foreign neurons available; need {n_select}."
#         )

#     rng = np.random.default_rng(seed)
#     idx = rng.choice(len(pool), size=n_select, replace=False)
#     return [pool[int(i)] for i in idx]


# def save_mapping(
#     mapping: Sequence[ForeignNeuron],
#     path: Path,
#     coordinate_name: str,
# ):
#     with path.open("w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow([coordinate_name, "foreign_session", "foreign_neuron_index"])
#         for j, (day, neuron_idx) in enumerate(mapping):
#             w.writerow([j, f"{DATASET_NAME}{day}", neuron_idx])
#     print("saved:", path)


# def build_foreign_matrix(sessions, mapping, split: str):
#     if split not in ("train", "test"):
#         raise ValueError("split must be train/test.")

#     split_idx = 0 if split == "train" else 1
#     traces = []

#     for day, neuron_idx in mapping:
#         X = sessions[day][split_idx]
#         traces.append(X[:, neuron_idx].astype(np.float32))

#     # Minimum time across ALL 860 selected neurons.
#     T_min = min(len(x) for x in traces)
#     X = np.column_stack([x[:T_min] for x in traces]).astype(np.float32)

#     if X.shape != (T_min, len(mapping)):
#         raise RuntimeError(f"Unexpected foreign shape: {X.shape}")

#     n_sessions = len(set(day for day, _ in mapping))
#     print(
#         f"foreign_{split}_raw={X.shape} | T_min={T_min} | "
#         f"neurons={len(mapping)} | sessions={n_sessions}"
#     )
#     return X


# def choose_cross_test_86(foreign_test_860, mapping_860, seed=SEED + 300):
#     rng = np.random.default_rng(seed)
#     idx = rng.choice(
#         foreign_test_860.shape[1],
#         size=CROSS_TEST_NEURONS,
#         replace=False,
#     )
#     idx = np.sort(idx)
#     X86 = foreign_test_860[:, idx].astype(np.float32)
#     mapping86 = [mapping_860[int(i)] for i in idx]
#     return X86, mapping86


# # =============================================================================
# # MOMENT MATCHING FOR TEST VISUALIZATIONS
# # =============================================================================

# def column_stats(X):
#     mu = X.mean(axis=0, keepdims=True).astype(np.float32)
#     sd = X.std(axis=0, keepdims=True).astype(np.float32)
#     return mu, np.maximum(sd, EPS)


# def match_neuron_by_neuron(source, target):
#     source = np.asarray(source, dtype=np.float32)
#     target = np.asarray(target, dtype=np.float32)
#     if source.shape[1] != target.shape[1]:
#         raise ValueError(f"Column mismatch: {source.shape} vs {target.shape}")
#     sm, ss = column_stats(source)
#     tm, ts = column_stats(target)
#     return (((source - sm) / ss) * ts + tm).astype(np.float32)


# def print_match_check(name, X, target):
#     xm, xs = column_stats(X)
#     tm, ts = column_stats(target)
#     print(
#         f"{name}: max |dmean|={np.max(np.abs(xm-tm)):.6f} | "
#         f"max |dstd|={np.max(np.abs(xs-ts)):.6f}"
#     )


# def build_fake(X_test, seed=SEED + 200):
#     rng = np.random.default_rng(seed)
#     raw = rng.normal(size=X_test.shape).astype(np.float32)
#     return match_neuron_by_neuron(raw, X_test)


# # =============================================================================
# # NORMAL / PNHARD CEBRA
# # =============================================================================

# def common_cebra_kwargs():
#     return dict(
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         model_architecture=MODEL_ARCH,
#         time_offsets=OFFSET,
#         conditional=CONDITIONAL,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=HIDDEN,
#         device=DEVICE,
#         verbose=True,
#     )


# def build_normal_cebra():
#     return CEBRA(**common_cebra_kwargs())


# def build_pnhard_cebra(foreign_raw_860):
#     return CEBRA(
#         **common_cebra_kwargs(),
#         extra_negatives=foreign_raw_860,              # RAW (T_min, 860)
#         extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
#         extra_negative_candidate_multiplier=PNHARD_CANDIDATE_MULTIPLIER,
#         extra_negative_normalize=PNHARD_NORMALIZE,
#     )


# def save_model(model, filename):
#     path = MODELS_DIR / filename
#     model.save(path)
#     print("saved model:", path)


# def embed(model, X):
#     return np.asarray(model.transform(X.astype(np.float32)), dtype=np.float32)


# def save_embedding(X, filename):
#     np.save(EMB_DIR / filename, np.asarray(X, dtype=np.float32))


# def train_normal_pnhard(X_train, Y_train, foreign_raw_860):
#     print("\n" + "#" * 100)
#     print("TRAIN 1 -- NORMAL CEBRA + LABEL")
#     print("#" * 100)
#     seed_all(SEED)
#     normal = build_normal_cebra()
#     normal.fit(X_train, Y_train)
#     save_model(normal, "normal_cebra_behavior.pt")

#     k = int(round(EXTRA_NEGATIVE_FRACTION * BATCH_SIZE))
#     k = max(0, min(k, BATCH_SIZE - 1))
#     M = PNHARD_CANDIDATE_MULTIPLIER * k

#     print("\n" + "#" * 100)
#     print("TRAIN 2 -- PNHARD CEBRA + LABEL")
#     print("#" * 100)
#     print(
#         f"raw foreign pool={foreign_raw_860.shape} | "
#         f"k={k} | multiplier={PNHARD_CANDIDATE_MULTIPLIER} | "
#         f"candidates={M} | internal normalize={PNHARD_NORMALIZE}"
#     )

#     seed_all(SEED)
#     pnhard = build_pnhard_cebra(foreign_raw_860)
#     pnhard.fit(X_train, Y_train)
#     save_model(pnhard, "pnhard_cebra_behavior.pt")
#     return normal, pnhard


# # =============================================================================
# # PCA / EMBEDDING TESTS
# # =============================================================================

# def plot_pca_comparison(emb_a, label_a, emb_b, label_b, title, out2d, out3d):
#     pca = PCA(n_components=3)
#     pca.fit(np.concatenate([emb_a, emb_b], axis=0))
#     A = pca.transform(emb_a)
#     B = pca.transform(emb_b)
#     v = pca.explained_variance_ratio_

#     plt.figure(figsize=(7, 6))
#     plt.scatter(A[:, 0], A[:, 1], s=6, alpha=0.5, label=label_a)
#     plt.scatter(B[:, 0], B[:, 1], s=6, alpha=0.5, label=label_b)
#     plt.xlabel(f"PC1 ({100*v[0]:.1f}%)")
#     plt.ylabel(f"PC2 ({100*v[1]:.1f}%)")
#     plt.title(title + " -- PCA 2D")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out2d, dpi=220)
#     plt.close()

#     fig = plt.figure(figsize=(8, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=6, alpha=0.5, label=label_a)
#     ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=6, alpha=0.5, label=label_b)
#     ax.set_xlabel(f"PC1 ({100*v[0]:.1f}%)")
#     ax.set_ylabel(f"PC2 ({100*v[1]:.1f}%)")
#     ax.set_zlabel(f"PC3 ({100*v[2]:.1f}%)")
#     ax.set_title(title + " -- PCA 3D")
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig(out3d, dpi=220)
#     plt.close()

#     print("saved:", out2d)
#     print("saved:", out3d)


# def run_fake_test(normal, pnhard, X_test):
#     print("\n" + "#" * 100)
#     print("TEST 1 -- C-CO12 TEST vs FAKE")
#     print("#" * 100)

#     fake = build_fake(X_test)
#     print_match_check("fake vs C-CO12 test", fake, X_test)

#     for tag, display, model in (
#         ("normal_cebra", "NORMAL CEBRA", normal),
#         ("pnhard", "PNHARD CEBRA", pnhard),
#     ):
#         real_emb = embed(model, X_test)
#         fake_emb = embed(model, fake)
#         save_embedding(real_emb, f"{tag}_cco12_test.npy")
#         save_embedding(fake_emb, f"{tag}_fake.npy")
#         plot_pca_comparison(
#             real_emb, "C-CO12 test",
#             fake_emb, "fake, neuron-wise matched",
#             f"{display} -- C-CO12 test vs fake",
#             PLOTS_DIR / f"{tag}_cco12_vs_fake_pca2d.png",
#             PLOTS_DIR / f"{tag}_cco12_vs_fake_pca3d.png",
#         )


# def run_cross_session_test(normal, pnhard, X_test, foreign_test_860, mapping_860):
#     print("\n" + "#" * 100)
#     print("TEST 2 -- C-CO12 TEST vs CROSS SESSION")
#     print("#" * 100)

#     cross_raw, map86 = choose_cross_test_86(foreign_test_860, mapping_860)
#     save_mapping(map86, CROSS_TEST_MAP_CSV, "cross_test_coordinate")

#     T = min(len(X_test), len(cross_raw))
#     X_real = X_test[:T]
#     X_cross = match_neuron_by_neuron(cross_raw[:T], X_real)
#     print_match_check("cross-session 86 vs C-CO12 test", X_cross, X_real)

#     for tag, display, model in (
#         ("normal_cebra", "NORMAL CEBRA", normal),
#         ("pnhard", "PNHARD CEBRA", pnhard),
#     ):
#         real_emb = embed(model, X_real)
#         cross_emb = embed(model, X_cross)
#         save_embedding(real_emb, f"{tag}_cco12_test_cross_equalT.npy")
#         save_embedding(cross_emb, f"{tag}_cross_session_test.npy")
#         plot_pca_comparison(
#             real_emb, "C-CO12 test",
#             cross_emb, "cross-session 86, neuron-wise matched",
#             f"{display} -- C-CO12 test vs cross session",
#             PLOTS_DIR / f"{tag}_cco12_vs_cross_session_pca2d.png",
#             PLOTS_DIR / f"{tag}_cco12_vs_cross_session_pca3d.png",
#         )


# # =============================================================================
# # DECODER / R2
# # =============================================================================

# class TwoLayerMLP(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, output_dim),
#         )
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)

#     def forward(self, x):
#         return self.net(x)


# def align_embedding_labels(Z, Y):
#     n = min(len(Z), len(Y))
#     return Z[:n], Y[:n]


# def train_decoder_and_r2(display, Z_train, Z_test, Y_train, Y_test, save_path):
#     Z_train, Ytr = align_embedding_labels(Z_train, Y_train)
#     Z_test, Yte = align_embedding_labels(Z_test, Y_test)
#     if Ytr.ndim == 1:
#         Ytr = Ytr[:, None]
#     if Yte.ndim == 1:
#         Yte = Yte[:, None]

#     seed_all(SEED)
#     decoder = TwoLayerMLP(
#         input_dim=Z_train.shape[1],
#         hidden_dim=DECODER_HIDDEN_DIM,
#         output_dim=Ytr.shape[1],
#         dropout_rate=DECODER_DROPOUT,
#     ).to(TORCH_DEVICE)

#     ds = TensorDataset(
#         torch.from_numpy(Z_train.astype(np.float32)),
#         torch.from_numpy(Ytr.astype(np.float32)),
#     )
#     g = torch.Generator().manual_seed(SEED)
#     loader = DataLoader(
#         ds,
#         batch_size=DECODER_BATCH_SIZE,
#         shuffle=True,
#         drop_last=False,
#         num_workers=0,
#         generator=g,
#     )

#     opt = torch.optim.Adam(
#         decoder.parameters(),
#         lr=DECODER_LR,
#         weight_decay=DECODER_WEIGHT_DECAY,
#     )
#     loss_fn = nn.MSELoss()

#     print(f"\nDecoder {display}: {DECODER_EPOCHS} epochs")
#     for epoch in range(1, DECODER_EPOCHS + 1):
#         decoder.train()
#         for xb, yb in loader:
#             xb, yb = xb.to(TORCH_DEVICE), yb.to(TORCH_DEVICE)
#             opt.zero_grad(set_to_none=True)
#             loss = loss_fn(decoder(xb), yb)
#             loss.backward()
#             opt.step()
#         if epoch == 1 or epoch % DECODER_PRINT_EVERY == 0 or epoch == DECODER_EPOCHS:
#             print(f"{display}: decoder epoch {epoch}/{DECODER_EPOCHS}")

#     decoder.eval()
#     with torch.no_grad():
#         pred = decoder(
#             torch.from_numpy(Z_test.astype(np.float32)).to(TORCH_DEVICE)
#         ).cpu().numpy()

#     r2_each = np.asarray(
#         r2_score(Yte, pred, multioutput="raw_values"),
#         dtype=float,
#     )
#     mean_r2 = float(np.mean(r2_each))
#     torch.save(decoder.state_dict(), save_path)

#     print(f"{display} TEST R2 per target:", r2_each)
#     print(f"{display} TEST Mean R2 = {mean_r2:.6f}")
#     return r2_each, mean_r2


# def run_normal_pnhard_decoders(normal, pnhard, X_train, X_test, Y_train, Y_test):
#     normal_r2, normal_mean = train_decoder_and_r2(
#         "NORMAL CEBRA",
#         embed(normal, X_train),
#         embed(normal, X_test),
#         Y_train, Y_test,
#         DECODER_DIR / "normal_cebra_decoder.pt",
#     )
#     pnhard_r2, pnhard_mean = train_decoder_and_r2(
#         "PNHARD CEBRA",
#         embed(pnhard, X_train),
#         embed(pnhard, X_test),
#         Y_train, Y_test,
#         DECODER_DIR / "pnhard_cebra_decoder.pt",
#     )
#     return normal_r2, normal_mean, pnhard_r2, pnhard_mean


# # =============================================================================
# # CEBRA ATTRIBUTION JACOBIAN -- TRAIN SET
# # =============================================================================

# def compute_train_jacobian(model, X_train, display):
#     print(f"\nComputing CEBRA Jacobian: {display}")
#     net = model.solver_.model
#     if isinstance(net, nn.ModuleList):
#         if len(net) != 1:
#             raise RuntimeError("Expected single-session encoder.")
#         net = net[0]

#     device = next(net.parameters()).device
#     net.eval()
#     n_neurons = X_train.shape[1]

#     starts = np.linspace(
#         0,
#         len(X_train) - ATTR_LEN - 1,
#         ATTR_CHUNKS,
#         dtype=int,
#     )

#     jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
#     total = 0

#     for ci, start in enumerate(starts, 1):
#         chunk = X_train[start:start + ATTR_LEN]
#         inp = torch.tensor(
#             chunk,
#             dtype=torch.float32,
#             device=device,
#             requires_grad=True,
#         )

#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=net,
#             input_data=inp,
#             output_dimension=LATENT_DIM,
#         )
#         with torch.enable_grad():
#             result = method.compute_attribution_map(batch_size=ATTR_BATCH)

#         jf = np.abs(np.asarray(result["jf"]))
#         jf = np.squeeze(jf)

#         if jf.shape != (LATENT_DIM, n_neurons):
#             if jf.ndim < 2:
#                 raise RuntimeError(f"Unexpected jf shape: {jf.shape}")
#             jf = np.mean(jf, axis=tuple(range(jf.ndim - 2)))

#         if jf.shape != (LATENT_DIM, n_neurons):
#             raise RuntimeError(
#                 f"Expected {(LATENT_DIM, n_neurons)}, got {jf.shape}"
#             )

#         jf_sum += jf.astype(np.float64) * len(chunk)
#         total += len(chunk)
#         print(f"{display}: Jacobian chunk {ci}/{len(starts)}")

#         del inp, method, result, jf
#         cleanup()

#     return (jf_sum / total).astype(np.float32), starts


# def plot_jacobian(matrix, title, path, vmax):
#     plt.figure(figsize=(12, 8))
#     im = plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=vmax)
#     plt.colorbar(im, label="Absolute forward Jacobian")
#     plt.xlabel("Input neuron")
#     plt.ylabel("Latent dimension")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print("saved:", path)


# def run_jacobians(normal, pnhard, X_train):
#     normal_J, starts1 = compute_train_jacobian(normal, X_train, "NORMAL CEBRA")
#     pnhard_J, starts2 = compute_train_jacobian(pnhard, X_train, "PNHARD CEBRA")
#     if not np.array_equal(starts1, starts2):
#         raise RuntimeError("Jacobian chunks differ.")

#     np.save(JACOBIAN_DIR / "normal_cebra_train_jf.npy", normal_J)
#     np.save(JACOBIAN_DIR / "pnhard_cebra_train_jf.npy", pnhard_J)
#     np.save(JACOBIAN_DIR / "train_jacobian_chunk_starts.npy", starts1)

#     vmax = float(max(normal_J.max(), pnhard_J.max()))
#     if not np.isfinite(vmax) or vmax <= 0:
#         vmax = 1.0

#     plot_jacobian(
#         normal_J,
#         "NORMAL CEBRA -- Train Forward Jacobian",
#         JACOBIAN_DIR / "normal_cebra_train_jacobian.png",
#         vmax,
#     )
#     plot_jacobian(
#         pnhard_J,
#         "PNHARD CEBRA -- Train Forward Jacobian",
#         JACOBIAN_DIR / "pnhard_cebra_train_jacobian.png",
#         vmax,
#     )

#     fig, ax = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
#     im = ax[0].imshow(normal_J, aspect="auto", vmin=0.0, vmax=vmax)
#     ax[0].set_title("NORMAL CEBRA")
#     ax[0].set_xlabel("Input neuron")
#     ax[0].set_ylabel("Latent dimension")
#     ax[1].imshow(pnhard_J, aspect="auto", vmin=0.0, vmax=vmax)
#     ax[1].set_title("PNHARD CEBRA")
#     ax[1].set_xlabel("Input neuron")
#     fig.suptitle("Train Forward Jacobian -- CEBRA Attribution")
#     cbar = fig.colorbar(im, ax=ax, shrink=0.9)
#     cbar.set_label("Absolute forward Jacobian")
#     path = JACOBIAN_DIR / "normal_vs_pnhard_train_jacobian.png"
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print("saved:", path)


# # =============================================================================
# # STANDARD ACORN -- TRAINED LAST
# # =============================================================================

# def import_acorn_cebra():
#     clear_cebra_modules()
#     for p in (str(PNHARD_CEBRA_DIR), str(ACORN_CEBRA_DIR)):
#         while p in sys.path:
#             sys.path.remove(p)
#     sys.path.insert(0, str(ACORN_CEBRA_DIR))

#     import cebra as acorn_cebra
#     from cebra import CEBRA as ACORN_CEBRA

#     print("\nUsing standard ACORN CEBRA:")
#     print(acorn_cebra.__file__)

#     params = inspect.signature(ACORN_CEBRA.__init__).parameters
#     required = {
#         "training_mode", "adv_epsilon", "adv_alpha", "adv_steps", "attack_norm"
#     }
#     missing = required.difference(params)
#     if missing:
#         raise RuntimeError(f"ACORN API missing: {sorted(missing)}")

#     return ACORN_CEBRA


# def run_acorn(X_train, X_test, Y_train, Y_test):
#     ACORN_CEBRA = import_acorn_cebra()

#     model = ACORN_CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMPERATURE,
#         model_architecture=MODEL_ARCH,
#         time_offsets=OFFSET,
#         conditional=CONDITIONAL,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=HIDDEN,
#         training_mode="adversarial",
#         adv_epsilon=ACORN_EPSILON,
#         adv_alpha=ACORN_ALPHA,
#         adv_steps=ACORN_STEPS,
#         attack_norm=ACORN_ATTACK_NORM,
#         device=DEVICE,
#         verbose=True,
#     )

#     print("\n" + "#" * 100)
#     print("TRAIN 3 -- STANDARD ACORN + LABEL")
#     print("#" * 100)
#     print(
#         f"eps={ACORN_EPSILON} | alpha={ACORN_ALPHA} | "
#         f"steps={ACORN_STEPS} | norm={ACORN_ATTACK_NORM}"
#     )

#     seed_all(SEED)
#     model.fit(X_train, Y_train)
#     save_model(model, "acorn_eps0p5_behavior.pt")

#     r2_each, mean_r2 = train_decoder_and_r2(
#         "ACORN eps=0.5",
#         np.asarray(model.transform(X_train), dtype=np.float32),
#         np.asarray(model.transform(X_test), dtype=np.float32),
#         Y_train, Y_test,
#         DECODER_DIR / "acorn_eps0p5_decoder.pt",
#     )
#     return r2_each, mean_r2


# # =============================================================================
# # FINAL R2
# # =============================================================================

# def save_and_print_all_r2(
#     normal_r2, normal_mean,
#     pnhard_r2, pnhard_mean,
#     acorn_r2, acorn_mean,
# ):
#     rows = [
#         ("NORMAL CEBRA", float(normal_mean), np.asarray(normal_r2)),
#         ("PNHARD CEBRA", float(pnhard_mean), np.asarray(pnhard_r2)),
#         ("ACORN eps=0.5", float(acorn_mean), np.asarray(acorn_r2)),
#     ]

#     n_targets = max(len(x[2]) for x in rows)
#     with R2_CSV.open("w", newline="") as f:
#         w = csv.writer(f)
#         w.writerow(["model", "mean_r2"] + [f"r2_target_{i}" for i in range(n_targets)])
#         for name, mean_r2, r2s in rows:
#             w.writerow([name, mean_r2] + r2s.tolist())

#     print("saved:", R2_CSV)

#     width = 20
#     n_print = min(len(x[2]) for x in rows)

#     print("\n" + "=" * 82)
#     print("FINAL TEST R2 -- ALL THREE MODELS SIDE BY SIDE")
#     print("=" * 82)
#     print(f"{'Metric':<14}" + "".join(f"{x[0]:>{width}}" for x in rows))
#     print("-" * 82)

#     for i in range(n_print):
#         print(
#             f"{('Target ' + str(i)):<14}"
#             + "".join(f"{x[2][i]:>{width}.6f}" for x in rows)
#         )

#     print("-" * 82)
#     print(
#         f"{'MEAN R2':<14}"
#         + "".join(f"{x[1]:>{width}.6f}" for x in rows)
#     )
#     print("=" * 82)


# # =============================================================================
# # MAIN
# # =============================================================================

# def main():
#     ensure_dirs()
#     seed_all(SEED)

#     X_train, X_test, Y_train, Y_test = load_target_session()

#     # Build the SAME 860 foreign neuron identities for train/test.
#     sessions = load_other_sessions()
#     mapping_860 = choose_random_foreign_neurons(
#         sessions,
#         n_select=FOREIGN_POOL_NEURONS,
#         seed=SEED + 100,
#     )
#     save_mapping(mapping_860, FOREIGN_MAP_CSV, "foreign_pool_coordinate")

#     foreign_train_raw_860 = build_foreign_matrix(
#         sessions, mapping_860, split="train"
#     )
#     foreign_test_raw_860 = build_foreign_matrix(
#         sessions, mapping_860, split="test"
#     )

#     # IMPORTANT: no external training normalization here.
#     normal, pnhard = train_normal_pnhard(
#         X_train,
#         Y_train,
#         foreign_train_raw_860,
#     )

#     run_fake_test(normal, pnhard, X_test)
#     run_cross_session_test(
#         normal,
#         pnhard,
#         X_test,
#         foreign_test_raw_860,
#         mapping_860,
#     )

#     normal_r2, normal_mean, pnhard_r2, pnhard_mean = (
#         run_normal_pnhard_decoders(
#             normal, pnhard,
#             X_train, X_test,
#             Y_train, Y_test,
#         )
#     )

#     run_jacobians(normal, pnhard, X_train)

#     # Switch forks only after all PNHard/normal work is complete.
#     del normal
#     del pnhard
#     cleanup()

#     acorn_r2, acorn_mean = run_acorn(
#         X_train, X_test, Y_train, Y_test
#     )

#     # REQUIRED final output immediately before script ends.
#     save_and_print_all_r2(
#         normal_r2, normal_mean,
#         pnhard_r2, pnhard_mean,
#         acorn_r2, acorn_mean,
#     )


# if __name__ == "__main__":
#     main()
