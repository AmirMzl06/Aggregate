"""
C-CO12 BEHAVIOR-CEBRA experiment:
NORMAL CEBRA vs NEGPOS + final ACORN.
"""

from __future__ import annotations

import csv
import gc
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

ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "CEBRA-NegPos"

if not CEBRA_DIR.exists():
    raise FileNotFoundError(f"CEBRA-NegPos fork not found: {CEBRA_DIR}")

for _m in list(sys.modules):
    if _m == "cebra" or _m.startswith("cebra."):
        del sys.modules[_m]

while str(CEBRA_DIR) in sys.path:
    sys.path.remove(str(CEBRA_DIR))
sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra import CEBRA
import cebra.attribution

print("\nUsing CEBRA-NegPos fork:")
print(cebra.__file__)

params = inspect.signature(CEBRA.__init__).parameters
required = {"extra_negatives", "extra_negative_fraction"}
missing = required.difference(params)
if missing:
    raise RuntimeError("Wrong CEBRA fork loaded. Missing NegPos arguments: " f"{sorted(missing)}")

PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
DATASET_NAME = "C-CO"
TARGET_DAY = 0
TARGET_SESSION = f"{DATASET_NAME}{TARGET_DAY}"
N_NEURONS = None
N_CCO_SESSIONS = 53
OTHER_SESSION_IDS = None
SEED = 42

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"
OFFSET = 1
CONDITIONAL = "time_delta"
EXTRA_NEGATIVE_FRACTION = 0.85

ACORN_EPSILON = 0.5
ACORN_ALPHA = ACORN_EPSILON / 5.0
ACORN_STEPS = 10
ACORN_ATTACK_NORM = "linf"

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
SAVE_JACOBIAN_ARRAYS = True

TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8

OUT_DIR = ROOT / "CEBRA_CCO12_Behavior_NegPos_Jacobian"
MODELS_DIR = OUT_DIR / "models"
EMB_DIR = OUT_DIR / "embeddings"
PLOTS_DIR = OUT_DIR / "plots"
DECODER_DIR = OUT_DIR / "decoder"
JACOBIAN_DIR = OUT_DIR / "jacobian"
FOREIGN_MAP_CSV = OUT_DIR / "foreign_86_mapping.csv"
R2_CSV = DECODER_DIR / "normal_negpos_r2.csv"

def ensure_dirs():
    for d in (OUT_DIR, MODELS_DIR, EMB_DIR, PLOTS_DIR, DECODER_DIR, JACOBIAN_DIR):
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

def load_target_session():
    path = session_path(TARGET_SESSION)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    X_train = np.asarray(data["train_data"], dtype=np.float32)
    X_test = np.asarray(data["valid_data"], dtype=np.float32)
    Y_train = np.asarray(data["train_label"], dtype=np.float32)
    Y_test = np.asarray(data["valid_label"], dtype=np.float32)
    # global N_NEURONS
   
    # N_NEURONS = X_train.shape[1]
   
    # print("Detected neurons:", N_NEURONS)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(f"Expected 2D neural arrays; got {X_train.shape=} and {X_test.shape=}")
    
    global N_NEURONS
    N_NEURONS = X_train.shape[1]
    print("Detected neurons:", N_NEURONS)
    
    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]
    if Y_test.ndim == 1:
        Y_test = Y_test[:, None]
    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError(f"Train neural/label mismatch: {X_train.shape[0]} vs {Y_train.shape[0]}")
    if X_test.shape[0] != Y_test.shape[0]:
        raise ValueError(f"Test neural/label mismatch: {X_test.shape[0]} vs {Y_test.shape[0]}")
    if X_train.shape[1] != N_NEURONS:
        raise ValueError(f"Expected {N_NEURONS} C-CO12 neurons, got {X_train.shape[1]}")
    for name, arr in (("X_train", X_train), ("X_test", X_test), ("Y_train", Y_train), ("Y_test", Y_test)):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{name} contains NaN/Inf.")
    print("\nC-CO12:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)
    # global N_NEURONS
    # N_NEURONS = X_train.shape[1]
    # print("Detected neurons:", N_NEURONS)
    return X_train, X_test, Y_train, Y_test

def load_other_session_neural(session_name: str):
    path = session_path(session_name)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    X_train = np.asarray(data["train_data"], dtype=np.float32)
    X_test = np.asarray(data["valid_data"], dtype=np.float32)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(f"{session_name}: expected 2D arrays; train={X_train.shape}, test={X_test.shape}")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"{session_name}: train/test neuron mismatch.")
    if not np.isfinite(X_train).all():
        raise RuntimeError(f"{session_name}: train contains NaN/Inf.")
    if not np.isfinite(X_test).all():
        raise RuntimeError(f"{session_name}: test contains NaN/Inf.")
    return X_train, X_test

def column_stats(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sd = X.std(axis=0, keepdims=True).astype(np.float32)
    sd = np.maximum(sd, EPS)
    return mu, sd

def match_neuron_by_neuron(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D.")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"Neuron mismatch: source={source.shape}, target={target.shape}")
    src_mu, src_sd = column_stats(source)
    tgt_mu, tgt_sd = column_stats(target)
    Z = (source - src_mu) / src_sd
    matched = Z * tgt_sd + tgt_mu
    return matched.astype(np.float32)

def print_match_check(name: str, X: np.ndarray, target: np.ndarray):
    X_mu, X_sd = column_stats(X)
    T_mu, T_sd = column_stats(target)
    mean_err = float(np.max(np.abs(X_mu - T_mu)))
    std_err = float(np.max(np.abs(X_sd - T_sd)))
    print(f"{name}: {X.shape} | " f"max mean error={mean_err:.6f} | " f"max std error={std_err:.6f}")

ForeignNeuron = Tuple[int, int]

def get_other_session_ids():
    if OTHER_SESSION_IDS is not None:
        return [int(day) for day in OTHER_SESSION_IDS if int(day) != TARGET_DAY]
    return [day for day in range(N_CCO_SESSIONS) if day != TARGET_DAY]

def load_other_sessions():
    sessions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for day in get_other_session_ids():
        session_name = f"{DATASET_NAME}{day}"
        path = session_path(session_name)
        if not path.exists():
            print(f"[SKIP missing] {session_name}")
            continue
        try:
            X_train, X_test = load_other_session_neural(session_name)
        except Exception as e:
            print(f"[SKIP bad] {session_name}: {e}")
            continue
        if X_train.shape[1] == 0:
            print(f"[SKIP empty] {session_name}")
            continue
        sessions[day] = (X_train, X_test)
        print(f"[OTHER OK] {session_name}: " f"train={X_train.shape}, " f"test={X_test.shape}")
    if not sessions:
        raise RuntimeError("No usable other C-CO sessions found.")
    return sessions

def choose_86_random_foreign_neurons(sessions, seed=SEED + 100) -> List[ForeignNeuron]:
    pool: List[ForeignNeuron] = []
    for day, (X_train, _) in sessions.items():
        for neuron_idx in range(X_train.shape[1]):
            pool.append((day, neuron_idx))
    if len(pool) < N_NEURONS:
        raise RuntimeError(f"Only {len(pool)} foreign neurons available; " f"need {N_NEURONS}.")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=N_NEURONS, replace=False)
    return [pool[int(i)] for i in idx]

def save_foreign_mapping(mapping: Sequence[ForeignNeuron]):
    with FOREIGN_MAP_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cco12_coordinate", "foreign_session", "foreign_neuron_index"])
        for j, (day, neuron_idx) in enumerate(mapping):
            writer.writerow([j, f"{DATASET_NAME}{day}", neuron_idx])
    print("saved:", FOREIGN_MAP_CSV)

def build_foreign_matrix(sessions, mapping, split: str):
    if split not in ("train", "test"):
        raise ValueError("split must be train or test.")
    array_index = 0 if split == "train" else 1
    traces = []
    for day, neuron_idx in mapping:
        X = sessions[day][array_index]
        if neuron_idx >= X.shape[1]:
            raise RuntimeError(f"C-CO{day} neuron {neuron_idx} missing in {split}.")
        traces.append(X[:, neuron_idx].astype(np.float32))
    T_min = min(len(trace) for trace in traces)
    foreign = np.column_stack([trace[:T_min] for trace in traces]).astype(np.float32)
    if foreign.shape != (T_min, N_NEURONS):
        raise RuntimeError(f"Unexpected foreign shape: {foreign.shape}")
    used_sessions = len(set(day for day, _ in mapping))
    print(f"foreign_{split}_raw: {foreign.shape} | " f"T_min={T_min} | " f"selected neurons from {used_sessions} sessions")
    return foreign

def build_fake_neuronwise(X_test: np.ndarray, seed=SEED + 200):
    rng = np.random.default_rng(seed)
    fake0 = rng.normal(size=X_test.shape).astype(np.float32)
    return match_neuron_by_neuron(fake0, X_test)

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
    return CEBRA(
        **common_cebra_kwargs(),
        extra_negatives=None,
        extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
    )

def build_negpos_cebra(foreign_train_norm):
    return CEBRA(
        **common_cebra_kwargs(),
        extra_negatives=foreign_train_norm,
        extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
    )

def save_cebra_model(model, filename):
    path = MODELS_DIR / filename
    model.save(path)
    print("saved model:", path)

def embed(model, X):
    return np.asarray(model.transform(X.astype(np.float32)), dtype=np.float32)

def save_embedding(emb, filename):
    path = EMB_DIR / filename
    np.save(path, np.asarray(emb, dtype=np.float32))

def plot_pca_comparison(emb_a, label_a, emb_b, label_b, title, out_2d, out_3d):
    combined = np.concatenate([emb_a, emb_b], axis=0)
    pca = PCA(n_components=3)
    pca.fit(combined)
    A = pca.transform(emb_a)
    B = pca.transform(emb_b)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(7, 6))
    plt.scatter(A[:, 0], A[:, 1], s=6, c="red", alpha=0.5, label=label_a)
    plt.scatter(B[:, 0], B[:, 1], s=6, c="blue", alpha=0.5, label=label_b)
    plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    plt.title(title + " -- PCA 2D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_2d, dpi=220)
    plt.close()
    print("saved:", out_2d)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], s=6, c="red", alpha=0.5, label=label_a)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], s=6, c="blue", alpha=0.5, label=label_b)
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var[2] * 100:.1f}%)")
    ax.set_title(title + " -- PCA 3D")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_3d, dpi=220)
    plt.close()
    print("saved:", out_3d)

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=2, dropout_rate=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()
    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    def forward(self, x):
        return self.net(x)

def align_embedding_and_labels(embedding, labels):
    n = min(len(embedding), len(labels))
    return (embedding[:n], labels[:n])

def train_decoder_and_r2(display_name, train_emb, test_emb, Y_train, Y_test, save_decoder_path=None):
    train_emb, Ytr = align_embedding_and_labels(train_emb, Y_train)
    test_emb, Yte = align_embedding_and_labels(test_emb, Y_test)
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
        torch.from_numpy(train_emb.astype(np.float32)),
        torch.from_numpy(Ytr.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=DECODER_BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0, generator=generator)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=DECODER_LR, weight_decay=DECODER_WEIGHT_DECAY)
    criterion = nn.MSELoss()
    print(f"\nDecoder: {display_name} | " f"epochs={DECODER_EPOCHS} | " f"input={train_emb.shape[1]} | " f"output={Ytr.shape[1]}")
    for epoch in range(1, DECODER_EPOCHS + 1):
        decoder.train()
        for xb, yb in loader:
            xb = xb.to(TORCH_DEVICE)
            yb = yb.to(TORCH_DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = decoder(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if epoch == 1 or epoch % DECODER_PRINT_EVERY == 0 or epoch == DECODER_EPOCHS:
            print(f"{display_name}: " f"decoder epoch " f"{epoch}/{DECODER_EPOCHS}")
    decoder.eval()
    with torch.no_grad():
        pred = decoder(torch.from_numpy(test_emb.astype(np.float32)).to(TORCH_DEVICE)).cpu().numpy()
    r2_each = np.asarray(r2_score(Yte, pred, multioutput="raw_values"), dtype=float)
    mean_r2 = float(np.mean(r2_each))
    if save_decoder_path is not None:
        torch.save(decoder.state_dict(), save_decoder_path)
    print(f"\n{display_name} R2 per target:", r2_each)
    print(f"{display_name} Mean R2 = " f"{mean_r2:.6f}")
    return r2_each, mean_r2

def save_normal_negpos_r2(normal_r2, normal_mean, negpos_r2, negpos_mean):
    max_targets = max(len(normal_r2), len(negpos_r2))
    with R2_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mean_r2"] + [f"r2_target_{i}" for i in range(max_targets)])
        writer.writerow(["Normal CEBRA", normal_mean] + normal_r2.tolist())
        writer.writerow(["NegPos", negpos_mean] + negpos_r2.tolist())
    print("saved:", R2_CSV)

def train_normal_and_negpos(X_train, Y_train, foreign_train_norm):
    print("\n" + "#" * 100)
    print("TRAIN 1 -- NORMAL CEBRA + LABEL")
    print("#" * 100)
    seed_all(SEED)
    normal_model = build_normal_cebra()
    normal_model.fit(X_train, Y_train)
    save_cebra_model(normal_model, "cebra_normal_behavior.pt")

    print("\n" + "#" * 100)
    print("TRAIN 2 -- NEGPOS CEBRA + LABEL")
    print("#" * 100)
    k = round(EXTRA_NEGATIVE_FRACTION * BATCH_SIZE)
    print(f"batch_size={BATCH_SIZE} | " f"foreign negative fraction=" f"{EXTRA_NEGATIVE_FRACTION} | " f"~{k} foreign negatives/batch")
    seed_all(SEED)
    negpos_model = build_negpos_cebra(foreign_train_norm)
    negpos_model.fit(X_train, Y_train)
    save_cebra_model(negpos_model, "cebra_negpos_behavior.pt")
    return normal_model, negpos_model

def test_fake(normal_model, negpos_model, X_test):
    print("\n" + "#" * 100)
    print("TEST 1 -- C-CO12 TEST vs FAKE")
    print("#" * 100)
    fake = build_fake_neuronwise(X_test)
    print_match_check("fake vs C-CO12 test", fake, X_test)
    model_specs = (("normal_cebra", "NORMAL CEBRA", normal_model), ("negpos", "NEGPOS", negpos_model))
    for file_tag, display_name, model in model_specs:
        real_emb = embed(model, X_test)
        fake_emb = embed(model, fake)
        save_embedding(real_emb, f"{file_tag}_cco12_test.npy")
        save_embedding(fake_emb, f"{file_tag}_fake.npy")
        plot_pca_comparison(
            real_emb, f"{TARGET_SESSION} test",
            fake_emb, "fake, neuron-wise matched",
            title=f"{display_name} -- C-CO12 test vs fake",
            out_2d=PLOTS_DIR / f"{file_tag}_cco12_vs_fake_pca2d.png",
            out_3d=PLOTS_DIR / f"{file_tag}_cco12_vs_fake_pca3d.png",
        )

def test_other86(normal_model, negpos_model, X_test, foreign_test_raw):
    print("\n" + "#" * 100)
    print("TEST 2 -- C-CO12 TEST vs 86 OTHER-SESSION NEURONS")
    print("#" * 100)
    T = min(X_test.shape[0], foreign_test_raw.shape[0])
    X_real = X_test[:T]
    X_other_raw = foreign_test_raw[:T]
    X_other = match_neuron_by_neuron(X_other_raw, X_real)
    print(f"final T={T} | " f"C-CO12={X_real.shape} | " f"other86={X_other.shape}")
    print_match_check("other86 vs cropped C-CO12 test", X_other, X_real)
    model_specs = (("normal_cebra", "NORMAL CEBRA", normal_model), ("negpos", "NEGPOS", negpos_model))
    for file_tag, display_name, model in model_specs:
        real_emb = embed(model, X_real)
        other_emb = embed(model, X_other)
        save_embedding(real_emb, f"{file_tag}_cco12_test_other86_equalT.npy")
        save_embedding(other_emb, f"{file_tag}_other86_test.npy")
        plot_pca_comparison(
            real_emb, f"{TARGET_SESSION} test",
            other_emb, "other-session 86, neuron-wise matched",
            title=f"{display_name} -- C-CO12 test vs other-session 86",
            out_2d=PLOTS_DIR / f"{file_tag}_cco12_vs_other86_pca2d.png",
            out_3d=PLOTS_DIR / f"{file_tag}_cco12_vs_other86_pca3d.png",
        )

def run_normal_negpos_decoders(normal_model, negpos_model, X_train, X_test, Y_train, Y_test):
    print("\n" + "#" * 100)
    print("DECODER R2 -- NORMAL CEBRA vs NEGPOS")
    print("#" * 100)
    normal_train_emb = embed(normal_model, X_train)
    normal_test_emb = embed(normal_model, X_test)
    negpos_train_emb = embed(negpos_model, X_train)
    negpos_test_emb = embed(negpos_model, X_test)
    normal_r2, normal_mean = train_decoder_and_r2(
        "NORMAL CEBRA", normal_train_emb, normal_test_emb, Y_train, Y_test,
        save_decoder_path=DECODER_DIR / "normal_cebra_decoder.pt",
    )
    negpos_r2, negpos_mean = train_decoder_and_r2(
        "NEGPOS", negpos_train_emb, negpos_test_emb, Y_train, Y_test,
        save_decoder_path=DECODER_DIR / "negpos_decoder.pt",
    )
    save_normal_negpos_r2(normal_r2, normal_mean, negpos_r2, negpos_mean)
    return normal_r2, normal_mean, negpos_r2, negpos_mean

def compute_cebra_train_jacobian(model, X_train, display_name):
    print("\n" + "-" * 90)
    print(f"{display_name} -- CEBRA TRAIN JACOBIAN")
    print("-" * 90)
    if len(X_train) <= ATTR_LEN:
        raise ValueError(f"X_train has only {len(X_train)} samples, " f"but ATTR_LEN={ATTR_LEN}.")
    net = model.solver_.model
    if isinstance(net, nn.ModuleList):
        if len(net) != 1:
            raise RuntimeError("Expected a single-session CEBRA encoder for attribution.")
        net = net[0]
    device = next(net.parameters()).device
    net.eval()
    n_neurons = X_train.shape[1]
    starts = np.linspace(0, len(X_train) - ATTR_LEN - 1, ATTR_CHUNKS, dtype=int)
    jf_sum = np.zeros((LATENT_DIM, n_neurons), dtype=np.float64)
    total = 0
    for chunk_idx, start in enumerate(starts, start=1):
        chunk = X_train[start:start + ATTR_LEN]
        inp = torch.tensor(chunk, dtype=torch.float32, device=device, requires_grad=True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM,
        )
        with torch.enable_grad():
            result = method.compute_attribution_map(batch_size=ATTR_BATCH)
        if "jf" not in result:
            raise RuntimeError("CEBRA attribution result does not contain key 'jf'. " f"Available keys: {list(result.keys())}")
        jf = np.abs(np.asarray(result["jf"]))
        jf = np.squeeze(jf)
        if jf.shape != (LATENT_DIM, n_neurons):
            if jf.ndim < 2:
                raise RuntimeError(f"Unexpected CEBRA Jacobian shape: {jf.shape}")
            jf = np.mean(jf, axis=tuple(range(jf.ndim - 2)))
        if jf.shape != (LATENT_DIM, n_neurons):
            raise RuntimeError("After reduction, expected CEBRA attribution Jacobian " f"shape {(LATENT_DIM, n_neurons)}, got {jf.shape}.")
        jf_sum += jf.astype(np.float64) * len(chunk)
        total += len(chunk)
        print(f"{display_name}: attribution chunk " f"{chunk_idx}/{len(starts)} | " f"start={start} | " f"jf shape={jf.shape}")
        del inp, method, result, jf
        cleanup()
    jf_mean = (jf_sum / float(total)).astype(np.float32)
    print(f"{display_name}: final Jacobian shape = " f"{jf_mean.shape}")
    return jf_mean, starts

def plot_single_jacobian(matrix, title, out_path, vmax):
    plt.figure(figsize=(12, 8))
    im = plt.imshow(matrix, aspect="auto", vmin=0.0, vmax=vmax)
    plt.colorbar(im, label="Absolute forward Jacobian")
    plt.xlabel("Input neuron")
    plt.ylabel("Latent dimension")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", out_path)

def plot_jacobian_comparison(normal_J, negpos_J):
    vmax = float(max(np.max(normal_J), np.max(negpos_J)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    normal_path = JACOBIAN_DIR / "normal_cebra_train_cebra_jacobian.png"
    negpos_path = JACOBIAN_DIR / "negpos_train_cebra_jacobian.png"
    plot_single_jacobian(normal_J, "NORMAL CEBRA -- Train Forward Jacobian", normal_path, vmax)
    plot_single_jacobian(negpos_J, "NEGPOS -- Train Forward Jacobian", negpos_path, vmax)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    im = axes[0].imshow(normal_J, aspect="auto", vmin=0.0, vmax=vmax)
    axes[0].set_title("NORMAL CEBRA")
    axes[0].set_xlabel("Input neuron")
    axes[0].set_ylabel("Latent dimension")
    axes[1].imshow(negpos_J, aspect="auto", vmin=0.0, vmax=vmax)
    axes[1].set_title("NEGPOS")
    axes[1].set_xlabel("Input neuron")
    fig.suptitle("Train Forward Jacobian -- CEBRA Attribution")
    cbar = fig.colorbar(im, ax=axes, shrink=0.9)
    cbar.set_label("Absolute forward Jacobian")
    out_path = JACOBIAN_DIR / "normal_vs_negpos_train_cebra_jacobian.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_path)

def run_train_jacobians(normal_model, negpos_model, X_train):
    print("\n" + "#" * 100)
    print("TRAIN-SET JACOBIANS -- CEBRA ATTRIBUTION")
    print("#" * 100)
    normal_J, normal_starts = compute_cebra_train_jacobian(normal_model, X_train, "NORMAL CEBRA")
    negpos_J, negpos_starts = compute_cebra_train_jacobian(negpos_model, X_train, "NEGPOS")
    if not np.array_equal(normal_starts, negpos_starts):
        raise RuntimeError("Normal CEBRA and NegPos attribution used different train chunks.")
    if SAVE_JACOBIAN_ARRAYS:
        np.save(JACOBIAN_DIR / "normal_cebra_train_cebra_jf.npy", normal_J)
        np.save(JACOBIAN_DIR / "negpos_train_cebra_jf.npy", negpos_J)
        np.save(JACOBIAN_DIR / "cebra_attribution_train_chunk_starts.npy", normal_starts)
    plot_jacobian_comparison(normal_J, negpos_J)

def _clear_all_cebra_modules():
    for module_name in list(sys.modules):
        if module_name == "cebra" or module_name.startswith("cebra."):
            del sys.modules[module_name]

def import_acorn_cebra_class():
    from utils.constants import CEBRA_DIR as ACORN_CEBRA_DIR
    acorn_dir = Path(ACORN_CEBRA_DIR).resolve()
    if not acorn_dir.exists():
        raise FileNotFoundError(f"ACORN CEBRA_DIR does not exist: {acorn_dir}")
    _clear_all_cebra_modules()
    while str(CEBRA_DIR) in sys.path:
        sys.path.remove(str(CEBRA_DIR))
    while str(acorn_dir) in sys.path:
        sys.path.remove(str(acorn_dir))
    sys.path.insert(0, str(acorn_dir))
    import cebra as acorn_cebra
    from cebra import CEBRA as ACORN_CEBRA
    print("\nUsing ACORN CEBRA:")
    print(acorn_cebra.__file__)
    params = inspect.signature(ACORN_CEBRA.__init__).parameters
    required = {"training_mode", "adv_epsilon", "adv_alpha", "adv_steps", "attack_norm"}
    missing = required.difference(params)
    if missing:
        raise RuntimeError("The CEBRA checkout from utils.constants.CEBRA_DIR " "does not expose the expected ACORN API. " f"Missing: {sorted(missing)}")
    return ACORN_CEBRA

def build_acorn(ACORN_CEBRA):
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

def train_acorn_and_decoder(X_train, X_test, Y_train, Y_test):
    print("\n" + "#" * 100)
    print("TRAIN 3 -- ACORN + LABEL")
    print("#" * 100)
    ACORN_CEBRA = import_acorn_cebra_class()
    seed_all(SEED)
    acorn_model = build_acorn(ACORN_CEBRA)
    print(f"ACORN epsilon={ACORN_EPSILON} | " f"alpha={ACORN_ALPHA} | " f"steps={ACORN_STEPS} | " f"norm={ACORN_ATTACK_NORM}")
    acorn_model.fit(X_train, Y_train)
    acorn_model_path = MODELS_DIR / "acorn_eps0p5_behavior.pt"
    acorn_model.save(acorn_model_path)
    print("saved model:", acorn_model_path)
    acorn_train_emb = np.asarray(acorn_model.transform(X_train.astype(np.float32)), dtype=np.float32)
    acorn_test_emb = np.asarray(acorn_model.transform(X_test.astype(np.float32)), dtype=np.float32)
    acorn_r2, acorn_mean = train_decoder_and_r2(
        "ACORN eps=0.5",
        acorn_train_emb,
        acorn_test_emb,
        Y_train,
        Y_test,
        save_decoder_path=DECODER_DIR / "acorn_eps0p5_decoder.pt",
    )
    return acorn_r2, acorn_mean

def save_all_three_r2(normal_r2, normal_mean, negpos_r2, negpos_mean, acorn_r2, acorn_mean):
    path = DECODER_DIR / "normal_negpos_acorn_r2.csv"
    rows = [
        ("Normal CEBRA", float(normal_mean), np.asarray(normal_r2, dtype=float)),
        ("NegPos", float(negpos_mean), np.asarray(negpos_r2, dtype=float)),
        ("ACORN eps=0.5", float(acorn_mean), np.asarray(acorn_r2, dtype=float)),
    ]
    n_targets = max(len(row[2]) for row in rows)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mean_r2"] + [f"r2_target_{i}" for i in range(n_targets)])
        for name, mean_r2, r2_each in rows:
            writer.writerow([name, mean_r2] + r2_each.tolist())
    print("saved:", path)

def print_all_three_r2(normal_r2, normal_mean, negpos_r2, negpos_mean, acorn_r2, acorn_mean):
    names = ("NORMAL CEBRA", "NEGPOS", "ACORN eps=0.5")
    arrays = (
        np.asarray(normal_r2, dtype=float),
        np.asarray(negpos_r2, dtype=float),
        np.asarray(acorn_r2, dtype=float),
    )
    means = (float(normal_mean), float(negpos_mean), float(acorn_mean))
    n_targets = min(len(arr) for arr in arrays)
    width = 20
    print("\n")
    print("=" * 80)
    print("FINAL TEST R2 -- ALL THREE MODELS")
    print("=" * 80)
    print(f"{'Metric':<14}" + "".join(f"{name:>{width}}" for name in names))
    print("-" * 80)
    for i in range(n_targets):
        print(f"{('Target ' + str(i)):<14}" + "".join(f"{arr[i]:>{width}.6f}" for arr in arrays))
    print("-" * 80)
    print(f"{'MEAN R2':<14}" + "".join(f"{value:>{width}.6f}" for value in means))
    print("=" * 80)

def main():
    ensure_dirs()
    seed_all(SEED)
    X_train, X_test, Y_train, Y_test = load_target_session()
    other_sessions = load_other_sessions()
    mapping = choose_86_random_foreign_neurons(other_sessions, seed=SEED + 100)
    save_foreign_mapping(mapping)
    print("\nSelected 86 foreign neurons:")
    for j, (day, neuron_idx) in enumerate(mapping):
        print(f"  coordinate {j:02d} <- " f"C-CO{day} neuron {neuron_idx}")
    foreign_train_raw = build_foreign_matrix(other_sessions, mapping, split="train")
    foreign_test_raw = build_foreign_matrix(other_sessions, mapping, split="test")
    foreign_train_norm = match_neuron_by_neuron(foreign_train_raw, X_train)
    print_match_check("foreign_train_norm vs C-CO12 train", foreign_train_norm, X_train)
    normal_model, negpos_model = train_normal_and_negpos(X_train, Y_train, foreign_train_norm)
    test_fake(normal_model, negpos_model, X_test)
    test_other86(normal_model, negpos_model, X_test, foreign_test_raw)
    (normal_r2, normal_mean, negpos_r2, negpos_mean) = run_normal_negpos_decoders(
        normal_model, negpos_model, X_train, X_test, Y_train, Y_test,
    )
    run_train_jacobians(normal_model, negpos_model, X_train)
    del normal_model
    del negpos_model
    cleanup()
    acorn_r2, acorn_mean = train_acorn_and_decoder(X_train, X_test, Y_train, Y_test)
    save_all_three_r2(normal_r2, normal_mean, negpos_r2, negpos_mean, acorn_r2, acorn_mean)
    print_all_three_r2(normal_r2, normal_mean, negpos_r2, negpos_mean, acorn_r2, acorn_mean)

if __name__ == "__main__":
    main()
