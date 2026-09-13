import os
import sys
import random
import numpy as np
import torch

from utils.constants import CEBRA_DIR
sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d projection


# =====================================================================
# CONFIG
# =====================================================================
PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
DATASET_NAME = "C-CO"
TARGET_DAY = 12
TARGET_SESSION = f"{DATASET_NAME}{TARGET_DAY}"
N_NEURONS = 86              # expected neuron count for C-CO12 (sanity-checked at load time)
N_CCO_SESSIONS = 53         # C-CO has sessions 0..52

# which other C-CO sessions to pull neurons from for the cross-session test.
# None -> every C-CO session except TARGET_DAY.
OTHER_SESSION_IDS = None

SEED = 42
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 5000
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

OFFSET = 1                  # time_offsets, as requested
ADV_EPSILON = 0.3           # FIXED epsilon, as requested (not derived from min_l2_distance)
ADV_ALPHA = ADV_EPSILON / 5.0
ADV_STEPS = 10
ATTACK_NORM = "linf"

MODELS_DIR = "models"
EMB_DIR = "embeddings"
PLOTS_DIR = "plots"


def ensure_dirs():
    for d in (MODELS_DIR, EMB_DIR, PLOTS_DIR):
        os.makedirs(d, exist_ok=True)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# DATA
# =====================================================================
def session_path(session_name):
    return os.path.join(PERICH_DATA_DIR, f"{session_name}.npz")


def load_session_raw(session_name):
    """Returns (X_train, X_test), RAW -- intentionally NOT z-scored, since
    the fake-dataset step needs genuine per-neuron mean/std from this data."""
    path = session_path(session_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)
    return X_train, X_test


# =====================================================================
# CEBRA model build / save / load
# =====================================================================
def build_cebra(adversarial=False):
    seed_all(SEED)
    if adversarial:
        return CEBRA(
            batch_size=BATCH_SIZE,
            temperature=TEMPERATURE,
            model_architecture=MODEL_ARCH,
            time_offsets=OFFSET,
            max_iterations=MAX_ITER,
            output_dimension=LATENT_DIM,
            num_hidden_units=HIDDEN,
            training_mode="adversarial",
            adv_alpha=ADV_ALPHA,
            adv_epsilon=ADV_EPSILON,
            adv_steps=ADV_STEPS,
            attack_norm=ATTACK_NORM,
            device=DEVICE,
            verbose=True,
        )
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="clean",
        device=DEVICE,
        verbose=True,
    )


def model_path(name):
    return os.path.join(MODELS_DIR, f"{name}.pt")


def save_model(model, name):
    ensure_dirs()
    model.save(model_path(name))
    print("saved model:", model_path(name))


def load_model(name):
    path = model_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- train it first.")
    model = CEBRA.load(path)
    print("loaded model:", path)
    return model


# =====================================================================
# embeddings
# =====================================================================
def emb_path(name):
    return os.path.join(EMB_DIR, f"{name}.npy")


def save_embedding(emb, name):
    ensure_dirs()
    arr = np.asarray(emb, dtype=np.float32)
    np.save(emb_path(name), arr)
    print("saved embedding:", emb_path(name), arr.shape)


def embed(model, X):
    return np.asarray(model.transform(X.astype(np.float32)), dtype=np.float32)


# =====================================================================
# plotting
# =====================================================================
def plot_path(name):
    return os.path.join(PLOTS_DIR, name)


def plot_pca_comparison(emb_a, label_a, color_a, emb_b, label_b, color_b,
                        title, out_2d, out_3d=None, point_size=6, alpha=0.5):
    """Fits ONE PCA on the UNION of emb_a/emb_b, so both clouds are projected
    into the SAME space -- fitting two separate PCAs (one per group) would
    make a red-vs-blue comparison meaningless, since the axes wouldn't
    correspond to the same directions."""
    n_comp = 3 if out_3d is not None else 2
    combined = np.concatenate([emb_a, emb_b], axis=0)
    pca = PCA(n_components=n_comp)
    pca.fit(combined)
    proj_a = pca.transform(emb_a)
    proj_b = pca.transform(emb_b)
    var = pca.explained_variance_ratio_

    # ---- 2D ----
    plt.figure(figsize=(7, 6))
    plt.scatter(proj_a[:, 0], proj_a[:, 1], s=point_size, c=color_a, alpha=alpha, label=label_a)
    plt.scatter(proj_b[:, 0], proj_b[:, 1], s=point_size, c=color_b, alpha=alpha, label=label_b)
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.title(title + " -- PCA 2D")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_2d, dpi=200)
    plt.close()
    print("saved:", out_2d)

    # ---- 3D (optional) ----
    if out_3d is not None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(proj_a[:, 0], proj_a[:, 1], proj_a[:, 2], s=point_size, c=color_a, alpha=alpha, label=label_a)
        ax.scatter(proj_b[:, 0], proj_b[:, 1], proj_b[:, 2], s=point_size, c=color_b, alpha=alpha, label=label_b)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(title + " -- PCA 3D")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_3d, dpi=200)
        plt.close()
        print("saved:", out_3d)


# =====================================================================
# STAGE 1 -- train CLEAN and ACORN on C-CO12
# =====================================================================
def stage1_train_models():
    print("\n" + "#" * 100)
    print("STAGE 1 -- TRAIN CLEAN + ACORN ON", TARGET_SESSION)
    print("#" * 100)

    X_train, X_test = load_session_raw(TARGET_SESSION)
    print(f"{TARGET_SESSION} neural train: {X_train.shape} | test: {X_test.shape}")
    if X_train.shape[1] != N_NEURONS:
        print(f"[WARN] expected {N_NEURONS} neurons, got {X_train.shape[1]} -- "
              f"update N_NEURONS at the top of this file if this is expected.")

    print("\n--- CLEAN ---")
    seed_all()
    clean_model = build_cebra(adversarial=False)
    clean_model.fit(X_train)
    save_model(clean_model, "clean")

    print(f"\n--- ACORN | epsilon={ADV_EPSILON} | alpha={ADV_ALPHA} | offset={OFFSET} ---")
    seed_all()
    acorn_model = build_cebra(adversarial=True)
    acorn_model.fit(X_train)
    save_model(acorn_model, "acorn")

    return clean_model, acorn_model, X_train, X_test


# =====================================================================
# STAGE 2+3 -- fake dataset, embeddings, real-vs-fake PCA plots
# =====================================================================
def build_fake_like(X_test, seed=SEED):
    """One independent Gaussian per neuron: mean_i/std_i taken from the REAL
    test set. No cross-neuron correlation, no temporal structure -- this is
    a "marginal-matched, structure-destroyed" null dataset. Same number of
    rows as X_test, for a like-for-like comparison in the PCA plots."""
    rng = np.random.default_rng(seed)
    mu = X_test.mean(axis=0)
    sigma = X_test.std(axis=0)
    fake = rng.normal(loc=mu, scale=sigma, size=X_test.shape).astype(np.float32)
    return fake


def stage2_fake_and_embed(clean_model, acorn_model, X_test):
    print("\n" + "#" * 100)
    print("STAGE 2+3 -- FAKE DATASET, EMBEDDINGS, REAL-VS-FAKE PCA")
    print("#" * 100)

    fake = build_fake_like(X_test)
    mean_diff = float(np.max(np.abs(fake.mean(0) - X_test.mean(0))))
    std_diff = float(np.max(np.abs(fake.std(0) - X_test.std(0))))
    print(f"fake dataset: {fake.shape} | max |mean diff|={mean_diff:.4f} "
          f"| max |std diff|={std_diff:.4f} (both should be small -- sampling noise only)")

    for name, model in (("clean", clean_model), ("acorn", acorn_model)):
        print(f"\n--- {name.upper()} ---")

        real_emb = embed(model, X_test)
        fake_emb = embed(model, fake)

        save_embedding(real_emb, f"{name}_real")
        save_embedding(fake_emb, f"{name}_fake")

        plot_pca_comparison(
            real_emb, f"real {TARGET_SESSION} test", "red",
            fake_emb, "fake (marginal-matched)", "blue",
            title=f"{name.upper()} -- real vs fake",
            out_2d=plot_path(f"{name}_fake_pca2d.png"),
            out_3d=plot_path(f"{name}_fake_pca3d.png"),
        )


# =====================================================================
# STAGE 4 -- cross-session
# =====================================================================
def get_other_session_ids():
    if OTHER_SESSION_IDS is not None:
        return list(OTHER_SESSION_IDS)
    return [d for d in range(N_CCO_SESSIONS) if d != TARGET_DAY]


def select_neurons(X, n_neurons, seed):
    """Random (seeded) subset of n_neurons columns if X has more than that;
    X unchanged if it has exactly n_neurons; None (skip) if it has fewer."""
    total = X.shape[1]
    if total < n_neurons:
        return None
    if total == n_neurons:
        return X
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=n_neurons, replace=False))
    return X[:, idx]


def build_other_corpus(n_neurons=N_NEURONS, seed=SEED):
    """Uses train+test combined from each OTHER C-CO session (not C-CO12's
    own split, reserved for the "real" side of the comparison), picks
    n_neurons columns from each, and concatenates them in time."""
    parts, used, skipped = [], [], []

    for day in get_other_session_ids():
        session = f"{DATASET_NAME}{day}"
        if not os.path.exists(session_path(session)):
            print(f"[SKIP] {session}: file not found")
            skipped.append(session)
            continue
        try:
            X_train, X_test = load_session_raw(session)
        except Exception as e:
            print(f"[SKIP] {session}: {e}")
            skipped.append(session)
            continue

        X_all = np.concatenate([X_train, X_test], axis=0)
        X_sel = select_neurons(X_all, n_neurons, seed=seed + day)
        if X_sel is None:
            print(f"[SKIP] {session}: only {X_all.shape[1]} neurons (< {n_neurons})")
            skipped.append(session)
            continue

        parts.append(X_sel.astype(np.float32))
        used.append(session)
        print(f"[OK]   {session}: {X_sel.shape}")

    if not parts:
        raise RuntimeError("No usable other-C-CO sessions found.")

    corpus = np.concatenate(parts, axis=0)
    print(f"\nother_corpus: {corpus.shape} from {len(used)} sessions ({len(skipped)} skipped)")
    print("used sessions:", used)
    if skipped:
        print("skipped sessions:", skipped)
    return corpus


def stage4_cross_session(clean_model, acorn_model, X_test):
    print("\n" + "#" * 100)
    print("STAGE 4 -- CROSS-SESSION")
    print("#" * 100)

    other_corpus = build_other_corpus()

    for name, model in (("clean", clean_model), ("acorn", acorn_model)):
        print(f"\n--- {name.upper()} ---")

        real_emb = embed(model, X_test)
        other_emb = embed(model, other_corpus)

        plot_pca_comparison(
            real_emb, f"{TARGET_SESSION} test", "red",
            other_emb, "other C-CO sessions", "blue",
            title=f"{name.upper()} -- cross-session",
            out_2d=plot_path(f"{name}_cross_session.png"),
            out_3d=None,
        )


# =====================================================================
# MAIN
# =====================================================================
def main():
    ensure_dirs()

    clean_model, acorn_model, X_train, X_test = stage1_train_models()
    stage2_fake_and_embed(clean_model, acorn_model, X_test)
    stage4_cross_session(clean_model, acorn_model, X_test)

    print("\n" + "#" * 100)
    print("ALL DONE")
    print("#" * 100)
    print("models:     ", os.path.abspath(MODELS_DIR))
    print("embeddings: ", os.path.abspath(EMB_DIR))
    print("plots:      ", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    main()
