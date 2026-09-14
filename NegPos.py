"""
C-CO12 BEHAVIOR-CEBRA experiment

1) NORMAL CEBRA + labels
2) CEBRA-NegPos + labels + random other-session negatives
3) Same two embedding tests:
      - C-CO12 test vs fake
      - C-CO12 test vs 86 random neurons from other C-CO sessions
   Each comparison saves PCA 2D + PCA 3D.
4) Train the same TwoLayerMLP decoder for NORMAL CEBRA and NEGPOS
   and report test R^2.
5) Finally train ACORN with epsilon=0.5 using the ORIGINAL CEBRA_DIR,
   train the same decoder, and ONLY print its R^2 (no ACORN plots).

IMPORTANT:
- All three encoders are BEHAVIOR CEBRA:
      model.fit(X_train, Y_train)
- `conditional="time_delta"` is set explicitly.
- Foreign negatives do NOT need labels in standard CEBRA behavior sampling:
  labels condition the positive sampling; negatives are random/uniform.
"""

from __future__ import annotations

import csv
import gc
import importlib
import inspect
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =====================================================================
# PROJECT / CEBRA PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parent

# NegPos fork used for NORMAL CEBRA and NEGPOS.
NEGPOS_CEBRA_DIR = ROOT / "CEBRA-NegPos"

if not NEGPOS_CEBRA_DIR.exists():
    raise FileNotFoundError(
        f"CEBRA-NegPos fork not found: {NEGPOS_CEBRA_DIR}"
    )

# Original ACORN-capable fork.
# This is the same CEBRA_DIR used in the previous ACORN scripts.
from utils.constants import CEBRA_DIR as ACORN_CEBRA_DIR
ACORN_CEBRA_DIR = Path(ACORN_CEBRA_DIR).resolve()


def clear_cebra_modules():
    """Remove an already-imported CEBRA fork from Python module cache."""
    for name in list(sys.modules):
        if name == "cebra" or name.startswith("cebra."):
            del sys.modules[name]
    importlib.invalidate_caches()


def import_negpos_cebra():
    clear_cebra_modules()

    # Avoid accidentally prioritizing the ACORN fork.
    for p in (str(NEGPOS_CEBRA_DIR), str(ACORN_CEBRA_DIR)):
        while p in sys.path:
            sys.path.remove(p)

    sys.path.insert(0, str(NEGPOS_CEBRA_DIR))

    import cebra
    from cebra import CEBRA

    print("\nUsing CEBRA-NegPos fork:")
    print(cebra.__file__)

    params = inspect.signature(CEBRA.__init__).parameters
    required = {"extra_negatives", "extra_negative_fraction"}

    missing = required.difference(params)
    if missing:
        raise RuntimeError(
            "Wrong CEBRA fork loaded. Missing NegPos arguments: "
            f"{sorted(missing)}"
        )

    return cebra, CEBRA


cebra, CEBRA = import_negpos_cebra()


# =====================================================================
# CONFIG
# =====================================================================

PERICH_DATA_DIR = Path(
    "/data/hossein/mm_project/perich_data_valid_final_raw/"
)

DATASET_NAME = "C-CO"
TARGET_DAY = 12
TARGET_SESSION = f"{DATASET_NAME}{TARGET_DAY}"

N_NEURONS = 86
N_CCO_SESSIONS = 53

# None -> use all available C-CO sessions except C-CO12.
OTHER_SESSION_IDS = None

SEED = 42

# Shared encoder hyperparameters
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"
OFFSET = 1

# Explicit behavior-conditioned CEBRA.
CONDITIONAL = "time_delta"

# NegPos
EXTRA_NEGATIVE_FRACTION = 0.10

# Decoder
DECODER_HIDDEN_DIM = 64
DECODER_DROPOUT = 0.4
DECODER_EPOCHS = 2500
DECODER_BATCH_SIZE = 256
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_PRINT_EVERY = 500

# ACORN
ACORN_EPSILON = 0.5
ACORN_ALPHA = ACORN_EPSILON / 5.0   # 0.1
ACORN_STEPS = 10
ACORN_ATTACK_NORM = "linf"

TORCH_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPS = 1e-8

# Outputs for Normal CEBRA + NegPos only.
OUT_DIR = ROOT / "CEBRA_CCO12_Behavior_NegPos"
MODELS_DIR = OUT_DIR / "models"
EMB_DIR = OUT_DIR / "embeddings"
PLOTS_DIR = OUT_DIR / "plots"
DECODER_DIR = OUT_DIR / "decoder"

FOREIGN_MAP_CSV = OUT_DIR / "foreign_86_mapping.csv"
R2_CSV = DECODER_DIR / "normal_negpos_r2.csv"


# =====================================================================
# HELPERS
# =====================================================================

def ensure_dirs():
    for d in (
        OUT_DIR,
        MODELS_DIR,
        EMB_DIR,
        PLOTS_DIR,
        DECODER_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def session_path(session_name: str) -> Path:
    return PERICH_DATA_DIR / f"{session_name}.npz"


# =====================================================================
# DATA
# =====================================================================

def load_target_session():
    path = session_path(TARGET_SESSION)

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)

    X_train = np.asarray(
        data["train_data"],
        dtype=np.float32,
    )
    X_test = np.asarray(
        data["valid_data"],
        dtype=np.float32,
    )

    Y_train = np.asarray(
        data["train_label"],
        dtype=np.float32,
    )
    Y_test = np.asarray(
        data["valid_label"],
        dtype=np.float32,
    )

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"Expected 2D neural arrays; got "
            f"{X_train.shape=} and {X_test.shape=}"
        )

    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]

    if Y_test.ndim == 1:
        Y_test = Y_test[:, None]

    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError(
            f"Train neural/label mismatch: "
            f"{X_train.shape[0]} vs {Y_train.shape[0]}"
        )

    if X_test.shape[0] != Y_test.shape[0]:
        raise ValueError(
            f"Test neural/label mismatch: "
            f"{X_test.shape[0]} vs {Y_test.shape[0]}"
        )

    if X_train.shape[1] != N_NEURONS:
        raise ValueError(
            f"Expected {N_NEURONS} C-CO12 neurons, "
            f"got {X_train.shape[1]}"
        )

    for name, arr in (
        ("X_train", X_train),
        ("X_test", X_test),
        ("Y_train", Y_train),
        ("Y_test", Y_test),
    ):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{name} contains NaN/Inf.")

    print("\nC-CO12:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)

    return X_train, X_test, Y_train, Y_test


def load_other_session_neural(session_name: str):
    path = session_path(session_name)

    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)

    X_train = np.asarray(
        data["train_data"],
        dtype=np.float32,
    )
    X_test = np.asarray(
        data["valid_data"],
        dtype=np.float32,
    )

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            f"{session_name}: expected 2D arrays; "
            f"train={X_train.shape}, test={X_test.shape}"
        )

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"{session_name}: train/test neuron mismatch."
        )

    if not np.isfinite(X_train).all():
        raise RuntimeError(
            f"{session_name}: train contains NaN/Inf."
        )

    if not np.isfinite(X_test).all():
        raise RuntimeError(
            f"{session_name}: test contains NaN/Inf."
        )

    return X_train, X_test


# =====================================================================
# NEURON-WISE NORMALIZATION
# =====================================================================

def column_stats(X: np.ndarray):
    mu = X.mean(
        axis=0,
        keepdims=True,
    ).astype(np.float32)

    sd = X.std(
        axis=0,
        keepdims=True,
    ).astype(np.float32)

    sd = np.maximum(sd, EPS)

    return mu, sd


def match_neuron_by_neuron(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """
    For every coordinate j:

        matched_j =
            (source_j - mean(source_j)) / std(source_j)
            * std(target_j)
            + mean(target_j)

    Therefore source coordinate j gets the same finite-sample
    mean/std as target coordinate j (up to floating-point error).
    """
    source = np.asarray(
        source,
        dtype=np.float32,
    )

    target = np.asarray(
        target,
        dtype=np.float32,
    )

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D.")

    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"Neuron mismatch: source={source.shape}, "
            f"target={target.shape}"
        )

    src_mu, src_sd = column_stats(source)
    tgt_mu, tgt_sd = column_stats(target)

    Z = (source - src_mu) / src_sd
    matched = Z * tgt_sd + tgt_mu

    return matched.astype(np.float32)


def print_match_check(
    name: str,
    X: np.ndarray,
    target: np.ndarray,
):
    X_mu, X_sd = column_stats(X)
    T_mu, T_sd = column_stats(target)

    mean_err = float(
        np.max(np.abs(X_mu - T_mu))
    )

    std_err = float(
        np.max(np.abs(X_sd - T_sd))
    )

    print(
        f"{name}: {X.shape} | "
        f"max mean error={mean_err:.6f} | "
        f"max std error={std_err:.6f}"
    )


# =====================================================================
# SELECT EXACTLY 86 RANDOM FOREIGN NEURONS
# =====================================================================

ForeignNeuron = Tuple[int, int]


def get_other_session_ids():
    if OTHER_SESSION_IDS is not None:
        return [
            int(day)
            for day in OTHER_SESSION_IDS
            if int(day) != TARGET_DAY
        ]

    return [
        day
        for day in range(N_CCO_SESSIONS)
        if day != TARGET_DAY
    ]


def load_other_sessions():
    sessions: Dict[
        int,
        Tuple[np.ndarray, np.ndarray],
    ] = {}

    for day in get_other_session_ids():
        session_name = f"{DATASET_NAME}{day}"
        path = session_path(session_name)

        if not path.exists():
            print(f"[SKIP missing] {session_name}")
            continue

        try:
            X_train, X_test = load_other_session_neural(
                session_name
            )
        except Exception as e:
            print(f"[SKIP bad] {session_name}: {e}")
            continue

        if X_train.shape[1] == 0:
            print(f"[SKIP empty] {session_name}")
            continue

        sessions[day] = (
            X_train,
            X_test,
        )

        print(
            f"[OTHER OK] {session_name}: "
            f"train={X_train.shape}, "
            f"test={X_test.shape}"
        )

    if not sessions:
        raise RuntimeError(
            "No usable other C-CO sessions found."
        )

    return sessions


def choose_86_random_foreign_neurons(
    sessions,
    seed=SEED + 100,
) -> List[ForeignNeuron]:
    pool: List[ForeignNeuron] = []

    for day, (X_train, _) in sessions.items():
        for neuron_idx in range(X_train.shape[1]):
            pool.append(
                (day, neuron_idx)
            )

    if len(pool) < N_NEURONS:
        raise RuntimeError(
            f"Only {len(pool)} foreign neurons available; "
            f"need {N_NEURONS}."
        )

    rng = np.random.default_rng(seed)

    idx = rng.choice(
        len(pool),
        size=N_NEURONS,
        replace=False,
    )

    return [
        pool[int(i)]
        for i in idx
    ]


def save_foreign_mapping(
    mapping: Sequence[ForeignNeuron],
):
    with FOREIGN_MAP_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "cco12_coordinate",
            "foreign_session",
            "foreign_neuron_index",
        ])

        for j, (day, neuron_idx) in enumerate(mapping):
            writer.writerow([
                j,
                f"{DATASET_NAME}{day}",
                neuron_idx,
            ])

    print("saved:", FOREIGN_MAP_CSV)


def build_foreign_matrix(
    sessions,
    mapping,
    split: str,
):
    """
    Build exactly one (T_min, 86) foreign matrix.

    The same 86 neuron identities are used for train and test.
    Train uses each source session's train_data.
    Test uses each source session's valid_data.
    """
    if split not in ("train", "test"):
        raise ValueError(
            "split must be train or test."
        )

    array_index = (
        0 if split == "train" else 1
    )

    traces = []

    for day, neuron_idx in mapping:
        X = sessions[day][array_index]

        if neuron_idx >= X.shape[1]:
            raise RuntimeError(
                f"C-CO{day} neuron {neuron_idx} "
                f"missing in {split}."
            )

        traces.append(
            X[:, neuron_idx].astype(np.float32)
        )

    T_min = min(
        len(trace)
        for trace in traces
    )

    foreign = np.column_stack([
        trace[:T_min]
        for trace in traces
    ]).astype(np.float32)

    if foreign.shape != (
        T_min,
        N_NEURONS,
    ):
        raise RuntimeError(
            f"Unexpected foreign shape: {foreign.shape}"
        )

    used_sessions = len(
        set(day for day, _ in mapping)
    )

    print(
        f"foreign_{split}_raw: {foreign.shape} | "
        f"T_min={T_min} | "
        f"selected neurons from {used_sessions} sessions"
    )

    return foreign


# =====================================================================
# FAKE
# =====================================================================

def build_fake_neuronwise(
    X_test: np.ndarray,
    seed=SEED + 200,
):
    rng = np.random.default_rng(seed)

    fake0 = rng.normal(
        size=X_test.shape
    ).astype(np.float32)

    return match_neuron_by_neuron(
        fake0,
        X_test,
    )


# =====================================================================
# NORMAL + NEGPOS CEBRA
# =====================================================================

def common_cebra_kwargs():
    return dict(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        conditional=CONDITIONAL,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        device=DEVICE,
        verbose=True,
    )


def build_normal_cebra():
    """
    Normal behavior CEBRA.

    Uses the NegPos fork with extra_negatives=None, so no foreign
    negatives are injected.
    """
    return CEBRA(
        **common_cebra_kwargs(),
        extra_negatives=None,
        extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
    )


def build_negpos_cebra(
    foreign_train_norm,
):
    """
    Behavior CEBRA on C-CO12 labels, with 10% of its random negatives
    replaced by foreign-session neural samples.
    """
    return CEBRA(
        **common_cebra_kwargs(),
        extra_negatives=foreign_train_norm,
        extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
    )


def save_cebra_model(
    model,
    filename,
):
    path = MODELS_DIR / filename
    model.save(path)
    print("saved model:", path)


def embed(
    model,
    X,
):
    return np.asarray(
        model.transform(
            X.astype(np.float32)
        ),
        dtype=np.float32,
    )


def save_embedding(
    emb,
    filename,
):
    path = EMB_DIR / filename
    np.save(
        path,
        np.asarray(
            emb,
            dtype=np.float32,
        ),
    )


# =====================================================================
# PCA PLOTS
# =====================================================================

def plot_pca_comparison(
    emb_a,
    label_a,
    emb_b,
    label_b,
    title,
    out_2d,
    out_3d,
):
    """
    Fit ONE PCA on the union of both embeddings.

    The exact same PCA basis is used for the 2D and 3D plots.
    """
    combined = np.concatenate(
        [emb_a, emb_b],
        axis=0,
    )

    pca = PCA(
        n_components=3
    )

    pca.fit(combined)

    A = pca.transform(emb_a)
    B = pca.transform(emb_b)

    var = pca.explained_variance_ratio_

    # 2D
    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        A[:, 0],
        A[:, 1],
        s=6,
        c="red",
        alpha=0.5,
        label=label_a,
    )

    plt.scatter(
        B[:, 0],
        B[:, 1],
        s=6,
        c="blue",
        alpha=0.5,
        label=label_b,
    )

    plt.xlabel(
        f"PC1 ({var[0] * 100:.1f}%)"
    )
    plt.ylabel(
        f"PC2 ({var[1] * 100:.1f}%)"
    )

    plt.title(
        title + " -- PCA 2D"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        out_2d,
        dpi=220,
    )
    plt.close()

    print("saved:", out_2d)

    # 3D
    fig = plt.figure(
        figsize=(8, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.scatter(
        A[:, 0],
        A[:, 1],
        A[:, 2],
        s=6,
        c="red",
        alpha=0.5,
        label=label_a,
    )

    ax.scatter(
        B[:, 0],
        B[:, 1],
        B[:, 2],
        s=6,
        c="blue",
        alpha=0.5,
        label=label_b,
    )

    ax.set_xlabel(
        f"PC1 ({var[0] * 100:.1f}%)"
    )
    ax.set_ylabel(
        f"PC2 ({var[1] * 100:.1f}%)"
    )
    ax.set_zlabel(
        f"PC3 ({var[2] * 100:.1f}%)"
    )

    ax.set_title(
        title + " -- PCA 3D"
    )

    ax.legend()

    plt.tight_layout()
    plt.savefig(
        out_3d,
        dpi=220,
    )
    plt.close()

    print("saved:", out_3d)


# =====================================================================
# DECODER
# =====================================================================

class TwoLayerMLP(nn.Module):

    def __init__(
        self,
        input_dim=32,
        hidden_dim=64,
        output_dim=2,
        dropout_rate=0.4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(
                dropout_rate
            ),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(
                layer,
                nn.Linear,
            ):
                nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu",
                )

                if layer.bias is not None:
                    nn.init.constant_(
                        layer.bias,
                        0,
                    )

    def forward(self, x):
        return self.net(x)


def align_embedding_and_labels(
    embedding,
    labels,
):
    n = min(
        len(embedding),
        len(labels),
    )

    return (
        embedding[:n],
        labels[:n],
    )


def train_decoder_and_r2(
    display_name,
    train_emb,
    test_emb,
    Y_train,
    Y_test,
    save_decoder_path=None,
):
    train_emb, Ytr = align_embedding_and_labels(
        train_emb,
        Y_train,
    )

    test_emb, Yte = align_embedding_and_labels(
        test_emb,
        Y_test,
    )

    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]

    if Yte.ndim == 1:
        Yte = Yte[:, None]

    seed_all(SEED)

    decoder = TwoLayerMLP(
        input_dim=train_emb.shape[1],
        hidden_dim=DECODER_HIDDEN_DIM,
        output_dim=Ytr.shape[1],
        dropout_rate=DECODER_DROPOUT,
    ).to(TORCH_DEVICE)

    dataset = TensorDataset(
        torch.from_numpy(
            train_emb.astype(np.float32)
        ),
        torch.from_numpy(
            Ytr.astype(np.float32)
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=DECODER_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=DECODER_LR,
        weight_decay=DECODER_WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    print(
        f"\nDecoder: {display_name} | "
        f"epochs={DECODER_EPOCHS} | "
        f"input={train_emb.shape[1]} | "
        f"output={Ytr.shape[1]}"
    )

    for epoch in range(
        1,
        DECODER_EPOCHS + 1,
    ):
        decoder.train()

        for xb, yb in loader:
            xb = xb.to(
                TORCH_DEVICE
            )
            yb = yb.to(
                TORCH_DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = decoder(xb)

            loss = criterion(
                pred,
                yb,
            )

            loss.backward()
            optimizer.step()

        if (
            epoch == 1
            or epoch % DECODER_PRINT_EVERY == 0
            or epoch == DECODER_EPOCHS
        ):
            print(
                f"{display_name}: "
                f"decoder epoch "
                f"{epoch}/{DECODER_EPOCHS}"
            )

    decoder.eval()

    with torch.no_grad():
        pred = decoder(
            torch.from_numpy(
                test_emb.astype(np.float32)
            ).to(TORCH_DEVICE)
        ).cpu().numpy()

    r2_each = np.asarray(
        r2_score(
            Yte,
            pred,
            multioutput="raw_values",
        ),
        dtype=float,
    )

    mean_r2 = float(
        np.mean(r2_each)
    )

    if save_decoder_path is not None:
        torch.save(
            decoder.state_dict(),
            save_decoder_path,
        )

    print(
        f"\n{display_name} R2 per target:",
        r2_each,
    )
    print(
        f"{display_name} Mean R2 = "
        f"{mean_r2:.6f}"
    )

    return r2_each, mean_r2


def save_normal_negpos_r2(
    normal_r2,
    normal_mean,
    negpos_r2,
    negpos_mean,
):
    max_targets = max(
        len(normal_r2),
        len(negpos_r2),
    )

    with R2_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            ["model", "mean_r2"]
            + [
                f"r2_target_{i}"
                for i in range(max_targets)
            ]
        )

        writer.writerow(
            ["Normal CEBRA", normal_mean]
            + normal_r2.tolist()
        )

        writer.writerow(
            ["NegPos", negpos_mean]
            + negpos_r2.tolist()
        )

    print("saved:", R2_CSV)


# =====================================================================
# NORMAL + NEGPOS TRAINING
# =====================================================================

def train_normal_and_negpos(
    X_train,
    Y_train,
    foreign_train_norm,
):
    print("\n" + "#" * 100)
    print("TRAIN 1 -- NORMAL CEBRA + LABEL")
    print("#" * 100)

    seed_all(SEED)

    normal_model = build_normal_cebra()

    # BEHAVIOR CEBRA: labels are passed to CEBRA.
    normal_model.fit(
        X_train,
        Y_train,
    )

    save_cebra_model(
        normal_model,
        "cebra_normal_behavior.pt",
    )

    print("\n" + "#" * 100)
    print("TRAIN 2 -- NEGPOS CEBRA + LABEL")
    print("#" * 100)

    k = round(
        EXTRA_NEGATIVE_FRACTION
        * BATCH_SIZE
    )

    print(
        f"batch_size={BATCH_SIZE} | "
        f"foreign negative fraction="
        f"{EXTRA_NEGATIVE_FRACTION} | "
        f"~{k} foreign negatives/batch"
    )

    seed_all(SEED)

    negpos_model = build_negpos_cebra(
        foreign_train_norm
    )

    # BEHAVIOR CEBRA: C-CO12 labels condition the normal CEBRA sampling.
    # The foreign neural matrix is used only as replacement negatives.
    negpos_model.fit(
        X_train,
        Y_train,
    )

    save_cebra_model(
        negpos_model,
        "cebra_negpos_behavior.pt",
    )

    return normal_model, negpos_model


# =====================================================================
# EMBEDDING TESTS
# =====================================================================

def test_fake(
    normal_model,
    negpos_model,
    X_test,
):
    print("\n" + "#" * 100)
    print("TEST 1 -- C-CO12 TEST vs FAKE")
    print("#" * 100)

    fake = build_fake_neuronwise(
        X_test
    )

    print_match_check(
        "fake vs C-CO12 test",
        fake,
        X_test,
    )

    model_specs = (
        (
            "normal_cebra",
            "NORMAL CEBRA",
            normal_model,
        ),
        (
            "negpos",
            "NEGPOS",
            negpos_model,
        ),
    )

    for file_tag, display_name, model in model_specs:
        real_emb = embed(
            model,
            X_test,
        )

        fake_emb = embed(
            model,
            fake,
        )

        save_embedding(
            real_emb,
            f"{file_tag}_cco12_test.npy",
        )

        save_embedding(
            fake_emb,
            f"{file_tag}_fake.npy",
        )

        plot_pca_comparison(
            real_emb,
            f"{TARGET_SESSION} test",
            fake_emb,
            "fake, neuron-wise matched",
            title=(
                f"{display_name} -- "
                f"C-CO12 test vs fake"
            ),
            out_2d=(
                PLOTS_DIR
                / f"{file_tag}_cco12_vs_fake_pca2d.png"
            ),
            out_3d=(
                PLOTS_DIR
                / f"{file_tag}_cco12_vs_fake_pca3d.png"
            ),
        )


def test_other86(
    normal_model,
    negpos_model,
    X_test,
    foreign_test_raw,
):
    print("\n" + "#" * 100)
    print(
        "TEST 2 -- C-CO12 TEST vs "
        "86 OTHER-SESSION NEURONS"
    )
    print("#" * 100)

    T = min(
        X_test.shape[0],
        foreign_test_raw.shape[0],
    )

    X_real = X_test[:T]
    X_other_raw = foreign_test_raw[:T]

    # IMPORTANT: normalize AFTER final equal-time crop.
    X_other = match_neuron_by_neuron(
        X_other_raw,
        X_real,
    )

    print(
        f"final T={T} | "
        f"C-CO12={X_real.shape} | "
        f"other86={X_other.shape}"
    )

    print_match_check(
        "other86 vs cropped C-CO12 test",
        X_other,
        X_real,
    )

    model_specs = (
        (
            "normal_cebra",
            "NORMAL CEBRA",
            normal_model,
        ),
        (
            "negpos",
            "NEGPOS",
            negpos_model,
        ),
    )

    for file_tag, display_name, model in model_specs:
        real_emb = embed(
            model,
            X_real,
        )

        other_emb = embed(
            model,
            X_other,
        )

        save_embedding(
            real_emb,
            f"{file_tag}_cco12_test_other86_equalT.npy",
        )

        save_embedding(
            other_emb,
            f"{file_tag}_other86_test.npy",
        )

        plot_pca_comparison(
            real_emb,
            f"{TARGET_SESSION} test",
            other_emb,
            "other-session 86, neuron-wise matched",
            title=(
                f"{display_name} -- "
                f"C-CO12 test vs other-session 86"
            ),
            out_2d=(
                PLOTS_DIR
                / f"{file_tag}_cco12_vs_other86_pca2d.png"
            ),
            out_3d=(
                PLOTS_DIR
                / f"{file_tag}_cco12_vs_other86_pca3d.png"
            ),
        )


# =====================================================================
# NORMAL + NEGPOS DECODERS
# =====================================================================

def run_normal_negpos_decoders(
    normal_model,
    negpos_model,
    X_train,
    X_test,
    Y_train,
    Y_test,
):
    print("\n" + "#" * 100)
    print("DECODER R2 -- NORMAL CEBRA vs NEGPOS")
    print("#" * 100)

    normal_train_emb = embed(
        normal_model,
        X_train,
    )

    normal_test_emb = embed(
        normal_model,
        X_test,
    )

    negpos_train_emb = embed(
        negpos_model,
        X_train,
    )

    negpos_test_emb = embed(
        negpos_model,
        X_test,
    )

    normal_r2, normal_mean = train_decoder_and_r2(
        "NORMAL CEBRA",
        normal_train_emb,
        normal_test_emb,
        Y_train,
        Y_test,
        save_decoder_path=(
            DECODER_DIR
            / "normal_cebra_decoder.pt"
        ),
    )

    negpos_r2, negpos_mean = train_decoder_and_r2(
        "NEGPOS",
        negpos_train_emb,
        negpos_test_emb,
        Y_train,
        Y_test,
        save_decoder_path=(
            DECODER_DIR
            / "negpos_decoder.pt"
        ),
    )

    save_normal_negpos_r2(
        normal_r2,
        normal_mean,
        negpos_r2,
        negpos_mean,
    )


# =====================================================================
# ACORN -- ORIGINAL CEBRA_DIR
# =====================================================================

def import_acorn_cebra_class():
    """
    Switch from CEBRA-NegPos to the original ACORN-capable CEBRA_DIR.
    """
    clear_cebra_modules()

    for p in (
        str(NEGPOS_CEBRA_DIR),
        str(ACORN_CEBRA_DIR),
    ):
        while p in sys.path:
            sys.path.remove(p)

    sys.path.insert(
        0,
        str(ACORN_CEBRA_DIR),
    )

    import cebra as acorn_cebra
    from cebra import CEBRA as ACORN_CEBRA

    print("\nUsing ACORN/original CEBRA_DIR:")
    print(acorn_cebra.__file__)

    params = inspect.signature(
        ACORN_CEBRA.__init__
    ).parameters

    required = {
        "training_mode",
        "adv_epsilon",
        "adv_alpha",
        "adv_steps",
        "attack_norm",
    }

    missing = required.difference(
        params
    )

    if missing:
        raise RuntimeError(
            "Loaded CEBRA_DIR does not expose the expected "
            f"ACORN API. Missing: {sorted(missing)}"
        )

    return ACORN_CEBRA


def build_acorn(
    ACORN_CEBRA,
):
    return ACORN_CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        conditional=CONDITIONAL,
        max_iterations=MAX_ITER,
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


def run_acorn_r2_only(
    X_train,
    X_test,
    Y_train,
    Y_test,
):
    """
    Train ACORN eps=0.5 + same decoder.
    No PCA, no plots, no saved embedding, no ACORN result file.
    Only its R^2 is printed.
    """
    print("\n" + "#" * 100)
    print("ACORN eps=0.5 -- R2 ONLY")
    print("#" * 100)

    ACORN_CEBRA = import_acorn_cebra_class()

    seed_all(SEED)

    acorn = build_acorn(
        ACORN_CEBRA
    )

    print(
        f"ACORN epsilon={ACORN_EPSILON} | "
        f"alpha={ACORN_ALPHA} | "
        f"steps={ACORN_STEPS} | "
        f"norm={ACORN_ATTACK_NORM} | "
        f"offset={OFFSET}"
    )

    # BEHAVIOR ACORN/CEBRA: labels are passed here as well.
    acorn.fit(
        X_train,
        Y_train,
    )

    train_emb = np.asarray(
        acorn.transform(
            X_train.astype(np.float32)
        ),
        dtype=np.float32,
    )

    test_emb = np.asarray(
        acorn.transform(
            X_test.astype(np.float32)
        ),
        dtype=np.float32,
    )

    # No ACORN decoder/model files are saved.
    _, mean_r2 = train_decoder_and_r2(
        "ACORN eps=0.5",
        train_emb,
        test_emb,
        Y_train,
        Y_test,
        save_decoder_path=None,
    )

    print("\n" + "=" * 100)
    print(
        f"ACORN eps=0.5 FINAL MEAN R2 = "
        f"{mean_r2:.6f}"
    )
    print("=" * 100)


# =====================================================================
# MAIN
# =====================================================================

def main():
    ensure_dirs()
    seed_all(SEED)

    # ---------------------------------------------------------------
    # Target data + labels
    # ---------------------------------------------------------------
    X_train, X_test, Y_train, Y_test = (
        load_target_session()
    )

    # ---------------------------------------------------------------
    # Foreign neural negatives
    # ---------------------------------------------------------------
    other_sessions = load_other_sessions()

    mapping = choose_86_random_foreign_neurons(
        other_sessions,
        seed=SEED + 100,
    )

    save_foreign_mapping(
        mapping
    )

    print("\nSelected 86 foreign neurons:")
    for j, (day, neuron_idx) in enumerate(mapping):
        print(
            f"  coordinate {j:02d} <- "
            f"C-CO{day} neuron {neuron_idx}"
        )

    foreign_train_raw = build_foreign_matrix(
        other_sessions,
        mapping,
        split="train",
    )

    foreign_test_raw = build_foreign_matrix(
        other_sessions,
        mapping,
        split="test",
    )

    # Train foreign negatives are neuron-wise matched to C-CO12 TRAIN.
    foreign_train_norm = match_neuron_by_neuron(
        foreign_train_raw,
        X_train,
    )

    print_match_check(
        "foreign_train_norm vs C-CO12 train",
        foreign_train_norm,
        X_train,
    )

    # ---------------------------------------------------------------
    # Train NORMAL CEBRA + NEGPOS, BOTH WITH LABELS
    # ---------------------------------------------------------------
    normal_model, negpos_model = train_normal_and_negpos(
        X_train,
        Y_train,
        foreign_train_norm,
    )

    # ---------------------------------------------------------------
    # Embedding tests / plots
    # ---------------------------------------------------------------
    test_fake(
        normal_model,
        negpos_model,
        X_test,
    )

    test_other86(
        normal_model,
        negpos_model,
        X_test,
        foreign_test_raw,
    )

    # ---------------------------------------------------------------
    # Decoders / R2
    # ---------------------------------------------------------------
    run_normal_negpos_decoders(
        normal_model,
        negpos_model,
        X_train,
        X_test,
        Y_train,
        Y_test,
    )

    # We are done with the NegPos fork before switching CEBRA imports.
    del normal_model
    del negpos_model
    cleanup()

    # ---------------------------------------------------------------
    # ACORN eps=0.5 -- only R2
    # ---------------------------------------------------------------
    run_acorn_r2_only(
        X_train,
        X_test,
        Y_train,
        Y_test,
    )


if __name__ == "__main__":
    main()
