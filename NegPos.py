from __future__ import annotations
import csv
import inspect
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# ============================================================
# USE THE MODIFIED CROSS-SESSION-NEGATIVE CEBRA FORK
# ============================================================

CEBRA_DIR = Path(__file__).resolve().parent / "CEBRA-NegPos"

if not CEBRA_DIR.exists():
    raise FileNotFoundError(f"CEBRA fork not found: {CEBRA_DIR}")

# Remove any previously imported CEBRA package.
for _m in list(sys.modules):
    if _m == "cebra" or _m.startswith("cebra."):
        del sys.modules[_m]

sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra import CEBRA

print("\nUsing CEBRA from:")
print(cebra.__file__)

from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================================
# CONFIG
# =====================================================================

PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw")

DATASET_NAME = "C-CO"
TARGET_DAY = 12
TARGET_SESSION = f"{DATASET_NAME}{TARGET_DAY}"

N_CCO_SESSIONS = 53          # C-CO0 ... C-CO52
N_NEURONS = 86               # C-CO12 input dimensionality
N_FOREIGN_NEURONS = 86       # must equal target input dimensionality

# None => all C-CO sessions except C-CO12 are candidates.
# Or e.g. [0, 1, 2, 3, 4, 5]
OTHER_SESSION_IDS = None

SEED = 42

# Same basic CEBRA setup as the previous C-CO12 representation script.
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 5000
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
TIME_OFFSET = 1
DEVICE = "cuda_if_available"

# 10% of the negative bank is replaced, NOT appended.
EXTRA_NEGATIVE_FRACTION = 0.10

OUT_DIR = Path("crossneg_experiment")
MODELS_DIR = OUT_DIR / "models"
EMB_DIR = OUT_DIR / "embeddings"
PLOTS_DIR = OUT_DIR / "plots"
FOREIGN_MAP_CSV = OUT_DIR / "foreign_neuron_map.csv"

EPS_STD = 1e-8


# =====================================================================
# REPRODUCIBILITY / IO
# =====================================================================

def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    for p in (OUT_DIR, MODELS_DIR, EMB_DIR, PLOTS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def session_path(session_name: str) -> Path:
    return PERICH_DATA_DIR / f"{session_name}.npz"


def load_session(session_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return raw (train_data, valid_data) as float32."""
    path = session_path(session_name)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    if "train_data" not in data or "valid_data" not in data:
        raise KeyError(f"{path} must contain train_data and valid_data")

    x_train = np.asarray(data["train_data"], dtype=np.float32)
    x_test = np.asarray(data["valid_data"], dtype=np.float32)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError(
            f"{session_name}: expected 2D neural arrays, got "
            f"train={x_train.shape}, test={x_test.shape}"
        )
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"{session_name}: train/test neuron counts differ: "
            f"{x_train.shape[1]} vs {x_test.shape[1]}"
        )
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError(f"{session_name}: NaN/Inf in neural data")

    return x_train, x_test


# =====================================================================
# NORMALIZATION / MARGINAL MATCHING
# =====================================================================

def neuron_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std."""
    mu = x.mean(axis=0, keepdims=True).astype(np.float32)
    sd = x.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(sd, EPS_STD)
    return mu, sd


def match_neuron_marginals(
    source: np.ndarray,
    target_reference: np.ndarray,
) -> np.ndarray:
    """
    Match source[:, j] to target_reference[:, j] neuron-by-neuron:

        z_j = (source_j - mu_source_j) / sd_source_j
        mapped_j = z_j * sd_target_j + mu_target_j

    This makes foreign coordinates live on the same marginal scale/offset as
    the corresponding C-CO12 coordinates, instead of exposing a trivial
    session-specific mean/variance cue to the shared encoder.
    """
    if source.ndim != 2 or target_reference.ndim != 2:
        raise ValueError("source and target_reference must both be 2D")
    if source.shape[1] != target_reference.shape[1]:
        raise ValueError(
            f"feature mismatch: source={source.shape}, "
            f"target_reference={target_reference.shape}"
        )

    src_mu, src_sd = neuron_stats(source)
    tgt_mu, tgt_sd = neuron_stats(target_reference)

    z = (source - src_mu) / src_sd
    mapped = z * tgt_sd + tgt_mu
    return mapped.astype(np.float32)


def print_marginal_match_diagnostics(
    name: str,
    x: np.ndarray,
    target: np.ndarray,
) -> None:
    x_mu, x_sd = neuron_stats(x)
    t_mu, t_sd = neuron_stats(target)

    max_mean_diff = float(np.max(np.abs(x_mu - t_mu)))
    max_std_diff = float(np.max(np.abs(x_sd - t_sd)))

    print(
        f"{name}: shape={x.shape} | "
        f"max neuron mean diff={max_mean_diff:.6g} | "
        f"max neuron std diff={max_std_diff:.6g}"
    )


# =====================================================================
# BUILD ONE 86-D FOREIGN "SESSION" FROM MANY OTHER SESSIONS
# =====================================================================

ForeignNeuron = Tuple[int, int]  # (session_day, neuron_index)


def get_other_session_ids() -> List[int]:
    if OTHER_SESSION_IDS is not None:
        ids = [int(x) for x in OTHER_SESSION_IDS if int(x) != TARGET_DAY]
    else:
        ids = [d for d in range(N_CCO_SESSIONS) if d != TARGET_DAY]

    return ids


def load_usable_foreign_sessions() -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all candidate other sessions that have usable train/test arrays.
    Returns day -> (train_data, valid_data).
    """
    usable: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for day in get_other_session_ids():
        name = f"{DATASET_NAME}{day}"
        path = session_path(name)

        if not path.exists():
            print(f"[SKIP missing] {name}")
            continue

        try:
            tr, te = load_session(name)
        except Exception as e:
            print(f"[SKIP bad] {name}: {e}")
            continue

        if tr.shape[1] < 1:
            print(f"[SKIP no neurons] {name}")
            continue

        usable[day] = (tr, te)
        print(
            f"[FOREIGN OK] {name}: train={tr.shape}, test={te.shape}"
        )

    if not usable:
        raise RuntimeError("No usable foreign C-CO sessions found.")

    return usable


def choose_foreign_neurons(
    sessions: Dict[int, Tuple[np.ndarray, np.ndarray]],
    n_total: int = N_FOREIGN_NEURONS,
    seed: int = SEED,
) -> List[ForeignNeuron]:
    """
    Select exactly n_total foreign neuron identities.

    Selection is spread across sessions in rounds:
      - shuffle session order
      - pick one not-yet-used neuron from each session
      - repeat until 86 coordinates are collected

    This avoids letting a single high-neuron-count session dominate the
    86-dimensional foreign matrix.
    """
    rng = np.random.default_rng(seed)

    days = list(sessions.keys())
    if not days:
        raise RuntimeError("No foreign sessions available.")

    available = {
        day: list(range(sessions[day][0].shape[1]))
        for day in days
    }
    for day in days:
        rng.shuffle(available[day])

    total_available = sum(len(v) for v in available.values())
    if total_available < n_total:
        raise RuntimeError(
            f"Only {total_available} total foreign neurons available; "
            f"need {n_total}."
        )

    selected: List[ForeignNeuron] = []

    while len(selected) < n_total:
        order = days.copy()
        rng.shuffle(order)
        made_progress = False

        for day in order:
            if len(selected) >= n_total:
                break
            if not available[day]:
                continue

            neuron_idx = available[day].pop()
            selected.append((day, int(neuron_idx)))
            made_progress = True

        if not made_progress:
            raise RuntimeError(
                f"Could not select {n_total} unique foreign neurons."
            )

    assert len(selected) == n_total
    return selected


def save_foreign_mapping(mapping: Sequence[ForeignNeuron]) -> None:
    ensure_dirs()
    with open(FOREIGN_MAP_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["input_coordinate", "source_session", "source_neuron_index"]
        )
        for coord, (day, neuron_idx) in enumerate(mapping):
            writer.writerow(
                [coord, f"{DATASET_NAME}{day}", neuron_idx]
            )
    print("saved foreign mapping:", FOREIGN_MAP_CSV)


def assemble_foreign_matrix(
    sessions: Dict[int, Tuple[np.ndarray, np.ndarray]],
    mapping: Sequence[ForeignNeuron],
    split: str,
) -> np.ndarray:
    """
    Stack the selected neurons as columns.

    split="train": use train_data
    split="test":  use valid_data

    Each selected neuron can come from a different session. We therefore use
    the shortest time length among all contributing sessions and truncate every
    trace to that T_min before column-stacking.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    split_idx = 0 if split == "train" else 1

    lengths = []
    traces = []

    for day, neuron_idx in mapping:
        x = sessions[day][split_idx]

        if neuron_idx >= x.shape[1]:
            raise IndexError(
                f"{DATASET_NAME}{day}: neuron {neuron_idx} out of bounds "
                f"for {split} shape {x.shape}"
            )

        lengths.append(x.shape[0])
        traces.append(x[:, neuron_idx])

    min_t = int(min(lengths))

    matrix = np.column_stack(
        [trace[:min_t] for trace in traces]
    ).astype(np.float32)

    if matrix.shape != (min_t, len(mapping)):
        raise RuntimeError(
            f"Unexpected foreign {split} shape: {matrix.shape}"
        )

    print(
        f"foreign_{split}_raw: {matrix.shape} | "
        f"shortest T={min_t} | "
        f"contributing sessions={len(set(day for day, _ in mapping))}"
    )
    return matrix


# =====================================================================
# FAKE DATASET
# =====================================================================

def build_fake_like_test(
    x_test: np.ndarray,
    seed: int = SEED + 1000,
) -> np.ndarray:
    """
    Independent Gaussian per neuron, with EACH C-CO12 test neuron's own
    mean/std. No cross-neuron correlation and no temporal structure.

    Shape is exactly the same as C-CO12 test.
    """
    rng = np.random.default_rng(seed)
    mu, sd = neuron_stats(x_test)

    fake = rng.normal(
        loc=mu,
        scale=sd,
        size=x_test.shape,
    ).astype(np.float32)

    return fake


# =====================================================================
# CEBRA
# =====================================================================

def verify_modified_cebra() -> None:
    print("\nUsing CEBRA from:")
    print(cebra.__file__)

    params = inspect.signature(CEBRA.__init__).parameters
    required = {"extra_negatives", "extra_negative_fraction"}
    missing = required.difference(params)

    if missing:
        raise RuntimeError(
            "This is not the modified cross-session-negative CEBRA fork. "
            f"Missing CEBRA.__init__ arguments: {sorted(missing)}"
        )

    print("[OK] modified cross-session-negative CEBRA API detected")


def common_cebra_kwargs() -> dict:
    return dict(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=TIME_OFFSET,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        device=DEVICE,
        verbose=True,
    )


def build_vanilla_cebra() -> CEBRA:
    return CEBRA(
        **common_cebra_kwargs(),
        extra_negatives=None,
        extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
    )


def build_crossneg_cebra(foreign_train: np.ndarray) -> CEBRA:
    return CEBRA(
        **common_cebra_kwargs(),
        extra_negatives=foreign_train.astype(np.float32),
        extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
    )


def model_path(name: str) -> Path:
    return MODELS_DIR / f"{name}.pt"


def save_model(model: CEBRA, name: str) -> None:
    path = model_path(name)
    model.save(path)
    print("saved model:", path)


def embed(model: CEBRA, x: np.ndarray) -> np.ndarray:
    return np.asarray(
        model.transform(np.asarray(x, dtype=np.float32)),
        dtype=np.float32,
    )


def save_embedding(name: str, x: np.ndarray) -> None:
    path = EMB_DIR / f"{name}.npy"
    np.save(path, np.asarray(x, dtype=np.float32))
    print("saved embedding:", path, x.shape)


# =====================================================================
# PLOTS
# =====================================================================

def plot_pca_pair(
    emb_a: np.ndarray,
    label_a: str,
    emb_b: np.ndarray,
    label_b: str,
    title: str,
    out_path: Path,
    point_size: float = 7,
    alpha: float = 0.5,
) -> None:
    """
    Fit ONE PCA on the union, then project both groups into the SAME axes.
    """
    combined = np.concatenate([emb_a, emb_b], axis=0)
    pca = PCA(n_components=2, random_state=SEED)
    projected = pca.fit_transform(combined)

    n_a = len(emb_a)
    a = projected[:n_a]
    b = projected[n_a:]

    var = pca.explained_variance_ratio_

    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(
        a[:, 0], a[:, 1],
        s=point_size, alpha=alpha,
        label=label_a,
    )
    plt.scatter(
        b[:, 0], b[:, 1],
        s=point_size, alpha=alpha,
        label=label_b,
    )
    plt.xlabel(f"PC1 ({100 * var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100 * var[1]:.1f}%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print("saved plot:", out_path)


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    ensure_dirs()
    seed_all()
    verify_modified_cebra()

    # ---------------------------------------------------------------
    # 1) TARGET: C-CO12
    # ---------------------------------------------------------------
    x_train, x_test = load_session(TARGET_SESSION)

    print("\n" + "=" * 100)
    print("TARGET SESSION")
    print("=" * 100)
    print(f"{TARGET_SESSION} train: {x_train.shape}")
    print(f"{TARGET_SESSION} test : {x_test.shape}")

    if x_train.shape[1] != N_NEURONS:
        raise RuntimeError(
            f"Expected {N_NEURONS} neurons in {TARGET_SESSION}, "
            f"got {x_train.shape[1]}"
        )

    # ---------------------------------------------------------------
    # 2) SAME 86 FOREIGN NEURON IDENTITIES FOR TRAIN AND TEST
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BUILD FOREIGN 86-NEURON MATRIX")
    print("=" * 100)

    foreign_sessions = load_usable_foreign_sessions()
    foreign_mapping = choose_foreign_neurons(
        foreign_sessions,
        n_total=N_FOREIGN_NEURONS,
        seed=SEED + 2000,
    )
    save_foreign_mapping(foreign_mapping)

    print("\nSelected foreign input coordinates:")
    for coord, (day, neuron_idx) in enumerate(foreign_mapping):
        print(
            f"  coord {coord:02d} <- "
            f"{DATASET_NAME}{day}, neuron {neuron_idx}"
        )

    foreign_train_raw = assemble_foreign_matrix(
        foreign_sessions,
        foreign_mapping,
        split="train",
    )
    foreign_test_raw = assemble_foreign_matrix(
        foreign_sessions,
        foreign_mapping,
        split="test",
    )

    # ---------------------------------------------------------------
    # 3) CRITICAL NEURON-BY-NEURON SCALE MATCHING
    # ---------------------------------------------------------------
    # Training foreign negatives are matched ONLY using training splits.
    foreign_train = match_neuron_marginals(
        foreign_train_raw,
        target_reference=x_train,
    )

    # For the requested test visualization, foreign valid_data is matched
    # neuron-by-neuron to the C-CO12 test distribution.
    foreign_test = match_neuron_marginals(
        foreign_test_raw,
        target_reference=x_test,
    )

    print("\nMarginal matching diagnostics:")
    print_marginal_match_diagnostics(
        "foreign_train matched to C-CO12 train",
        foreign_train,
        x_train,
    )
    print_marginal_match_diagnostics(
        "foreign_test matched to C-CO12 test",
        foreign_test,
        x_test,
    )

    # ---------------------------------------------------------------
    # 4) TRAIN TWO CLEAN MODELS
    # ---------------------------------------------------------------
    print("\n" + "#" * 100)
    print("MODEL A -- VANILLA CLEAN CEBRA")
    print("#" * 100)

    seed_all(SEED)
    vanilla = build_vanilla_cebra()
    vanilla.fit(x_train)
    save_model(vanilla, "cebra_vanilla")

    print("\n" + "#" * 100)
    print("MODEL B -- CLEAN CEBRA + CROSS-SESSION NEGATIVES")
    print("#" * 100)
    k_expected = int(round(EXTRA_NEGATIVE_FRACTION * BATCH_SIZE))
    k_expected = max(0, min(k_expected, BATCH_SIZE - 1))
    print(
        f"extra_negative_fraction={EXTRA_NEGATIVE_FRACTION} | "
        f"batch_size={BATCH_SIZE} | expected foreign negatives/step={k_expected}"
    )
    print("foreign training matrix:", foreign_train.shape)

    seed_all(SEED)
    crossneg = build_crossneg_cebra(foreign_train)
    crossneg.fit(x_train)
    save_model(crossneg, "cebra_cross_session_negatives")

    models = [
        ("vanilla", vanilla),
        ("crossneg", crossneg),
    ]

    # ---------------------------------------------------------------
    # 5) TEST CONDITION 1:
    #    C-CO12 test vs neuron-wise marginal-matched fake
    # ---------------------------------------------------------------
    print("\n" + "#" * 100)
    print("TEST 1 -- C-CO12 TEST vs FAKE TEST")
    print("#" * 100)

    fake_test = build_fake_like_test(x_test)

    print_marginal_match_diagnostics(
        "fake_test matched to C-CO12 test",
        fake_test,
        x_test,
    )

    for model_name, model in models:
        real_emb = embed(model, x_test)
        fake_emb = embed(model, fake_test)

        save_embedding(
            f"{model_name}_cco12_test",
            real_emb,
        )
        save_embedding(
            f"{model_name}_fake_test",
            fake_emb,
        )

        plot_pca_pair(
            real_emb,
            f"{TARGET_SESSION} test",
            fake_emb,
            "fake test (per-neuron marginal matched)",
            title=f"{model_name.upper()} | C-CO12 test vs fake",
            out_path=PLOTS_DIR / f"{model_name}_test_vs_fake.png",
        )

    # ---------------------------------------------------------------
    # 6) TEST CONDITION 2:
    #    C-CO12 test vs 86 neurons from other sessions' VALID sets.
    #
    #    The foreign matrix already uses the minimum time length among
    #    contributing sessions. For a perfectly balanced PCA comparison,
    #    also crop C-CO12 test to the same number of rows.
    # ---------------------------------------------------------------
    print("\n" + "#" * 100)
    print("TEST 2 -- C-CO12 TEST vs OTHER-SESSIONS 86 NEURONS")
    print("#" * 100)

    eval_t = min(x_test.shape[0], foreign_test.shape[0])
    x_test_equal = x_test[:eval_t]
    foreign_test_equal = foreign_test[:eval_t]

    print(
        f"equalized test length: {eval_t} | "
        f"C-CO12={x_test_equal.shape} | foreign={foreign_test_equal.shape}"
    )

    for model_name, model in models:
        real_emb = embed(model, x_test_equal)
        foreign_emb = embed(model, foreign_test_equal)

        save_embedding(
            f"{model_name}_cco12_test_equalT",
            real_emb,
        )
        save_embedding(
            f"{model_name}_foreign86_test_equalT",
            foreign_emb,
        )

        plot_pca_pair(
            real_emb,
            f"{TARGET_SESSION} test",
            foreign_emb,
            "86 foreign neurons (other-session valid sets)",
            title=f"{model_name.upper()} | C-CO12 test vs foreign 86",
            out_path=PLOTS_DIR / f"{model_name}_test_vs_foreign86.png",
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Models:")
    print(" ", model_path("cebra_vanilla"))
    print(" ", model_path("cebra_cross_session_negatives"))
    print("Foreign neuron mapping:")
    print(" ", FOREIGN_MAP_CSV)
    print("Plots:")
    print(" ", PLOTS_DIR / "vanilla_test_vs_fake.png")
    print(" ", PLOTS_DIR / "crossneg_test_vs_fake.png")
    print(" ", PLOTS_DIR / "vanilla_test_vs_foreign86.png")
    print(" ", PLOTS_DIR / "crossneg_test_vs_foreign86.png")


if __name__ == "__main__":
    main()
