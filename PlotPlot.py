import os
import sys
import gc
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

PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
DATASET_NAME = "C-CO"
TARGET_DAY = 12
TARGET_SESSION = f"{DATASET_NAME}{TARGET_DAY}"
N_NEURONS = 86
N_CCO_SESSIONS = 53
SEED = 42
MODELS_DIR = "models"
PLOTS_DIR = "plots_cross_session"
CLEAN_MODEL_PATH = os.path.join(MODELS_DIR, "clean.pt")
ACORN_MODEL_PATH = os.path.join(MODELS_DIR, "acorn.pt")
os.makedirs(PLOTS_DIR, exist_ok=True)

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all()

def session_path(session_name):
    return os.path.join(PERICH_DATA_DIR, f"{session_name}.npz")

def load_test(session_name):
    path = session_path(session_name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    X_test = data["valid_data"].astype(np.float32)
    return X_test

def load_target_test():
    X_target = load_test(TARGET_SESSION)
    print(f"{TARGET_SESSION} test:", X_target.shape)
    if X_target.shape[1] != N_NEURONS:
        raise RuntimeError(f"{TARGET_SESSION} has " f"{X_target.shape[1]} neurons, " f"expected {N_NEURONS}.")
    if not np.isfinite(X_target).all():
        raise RuntimeError(f"{TARGET_SESSION} test " "contains NaN or Inf.")
    return X_target

def get_other_session_ids():
    return [day for day in range(N_CCO_SESSIONS) if day != TARGET_DAY]

def build_cross_session_test_corpus():
    print("\n" + "=" * 100)
    print("BUILDING CROSS-SESSION TEST CORPUS")
    print("=" * 100)
    parts = []
    used = []
    skipped = []
    for day in get_other_session_ids():
        session = f"{DATASET_NAME}{day}"
        path = session_path(session)
        if not os.path.exists(path):
            print(f"[SKIP] {session}: " "file not found")
            skipped.append(session)
            continue
        try:
            X_test = load_test(session)
        except Exception as exc:
            print(f"[SKIP] {session}: " f"{exc}")
            skipped.append(session)
            continue
        if X_test.shape[0] == 0:
            print(f"[SKIP] {session}: " "empty test set")
            skipped.append(session)
            continue
        if X_test.shape[1] == 0:
            print(f"[SKIP] {session}: " "zero neurons")
            skipped.append(session)
            continue
        if not np.isfinite(X_test).all():
            print(f"[SKIP] {session}: " "NaN/Inf in test")
            skipped.append(session)
            continue
        parts.append(X_test.astype(np.float32))
        used.append(session)
        print(f"[OK]   {session}: " f"{X_test.shape}")
    if not parts:
        raise RuntimeError("No usable cross-session " "test sets found.")
    corpus = np.concatenate(parts, axis=0).astype(np.float32)
    print("\nCross-session test corpus:", corpus.shape)
    print("Used sessions:", used)
    if skipped:
        print("Skipped sessions:", skipped)
    return corpus

def select_random_neurons(corpus, n_neurons, seed=SEED):
    total_neurons = corpus.shape[1]
    if total_neurons < n_neurons:
        raise RuntimeError(f"Cross-session corpus has " f"only {total_neurons} neurons, " f"cannot select {n_neurons}.")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total_neurons, size=n_neurons, replace=False))
    selected = corpus[:, idx].astype(np.float32)
    print("\nRandomly selected neurons:", n_neurons)
    print("Selected indices:", idx)
    print("Selected corpus:", selected.shape)
    return selected, idx

def match_mean_std_to_target(X_source, X_target):
    if X_source.shape[1] != X_target.shape[1]:
        raise RuntimeError("Source and target must have " "the same number of neurons.")
    source_mean = X_source.mean(axis=0)
    source_std = X_source.std(axis=0)
    target_mean = X_target.mean(axis=0)
    target_std = X_target.std(axis=0)
    source_std_safe = source_std.copy()
    source_std_safe[source_std_safe < 1e-8] = 1.0
    X_matched = ((X_source - source_mean) / source_std_safe) * target_std
    X_matched += target_mean
    X_matched = X_matched.astype(np.float32)
    return X_matched

def print_statistics(X_target, X_cross):
    target_mean = X_target.mean(axis=0)
    target_std = X_target.std(axis=0)
    cross_mean = X_cross.mean(axis=0)
    cross_std = X_cross.std(axis=0)
    mean_diff = np.max(np.abs(target_mean - cross_mean))
    std_diff = np.max(np.abs(target_std - cross_std))
    print("\n" + "=" * 100)
    print("MEAN / STD MATCH CHECK")
    print("=" * 100)
    print("Max absolute mean diff:", f"{mean_diff:.10f}")
    print("Max absolute std diff :", f"{std_diff:.10f}")
    print("\nFirst 10 neurons:")
    print("Target mean:", target_mean[:10])
    print("Cross  mean:", cross_mean[:10])
    print("\nTarget std:", target_std[:10])
    print("Cross  std:", cross_std[:10])

def load_model(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    model = CEBRA.load(path)
    print(f"\nLoaded {name}:", path)
    return model

def embed(model, X):
    return np.asarray(model.transform(X.astype(np.float32)), dtype=np.float32)

def plot_pca_comparison(emb_target, emb_cross, model_name):
    combined = np.concatenate([emb_target, emb_cross], axis=0)
    pca = PCA(n_components=3)
    pca.fit(combined)
    proj_target = pca.transform(emb_target)
    proj_cross = pca.transform(emb_cross)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 7))
    plt.scatter(proj_target[:, 0], proj_target[:, 1], s=6, c="red", alpha=0.5, label=f"{TARGET_SESSION} test")
    plt.scatter(proj_cross[:, 0], proj_cross[:, 1], s=6, c="blue", alpha=0.5, label="Cross-session test")
    plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    plt.title(f"{model_name.upper()} -- " "C-CO12 vs cross-session\n" "PCA 2D")
    plt.legend()
    plt.tight_layout()
    path_2d = os.path.join(PLOTS_DIR, f"{model_name}_cross_session_2d.png")
    plt.savefig(path_2d, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path_2d)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(proj_target[:, 0], proj_target[:, 1], proj_target[:, 2], s=6, c="red", alpha=0.5, label=f"{TARGET_SESSION} test")
    ax.scatter(proj_cross[:, 0], proj_cross[:, 1], proj_cross[:, 2], s=6, c="blue", alpha=0.5, label="Cross-session test")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var[2] * 100:.1f}%)")
    ax.set_title(f"{model_name.upper()} -- " "C-CO12 vs cross-session\n" "PCA 3D")
    ax.legend()
    plt.tight_layout()
    path_3d = os.path.join(PLOTS_DIR, f"{model_name}_cross_session_3d.png")
    plt.savefig(path_3d, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path_3d)
    return path_2d, path_3d

def main():
    print("\n" + "#" * 120)
    print("CROSS-SESSION EMBEDDING TEST")
    print(f"TARGET SESSION = {TARGET_SESSION}")
    print(f"N NEURONS      = {N_NEURONS}")
    print("SOURCE         = OTHER C-CO TEST SETS ONLY")
    print("NORMALIZATION  = TARGET C-CO12 MEAN / STD")
    print("#" * 120)

    X_target_test = load_target_test()
    other_test_corpus = build_cross_session_test_corpus()
    X_cross_random, selected_idx = select_random_neurons(other_test_corpus, N_NEURONS, seed=SEED)
    X_cross_matched = match_mean_std_to_target(X_cross_random, X_target_test)
    print_statistics(X_target_test, X_cross_matched)

    np.save(os.path.join(PLOTS_DIR, "cross_session_test_random86_meanstd_matched.npy"), X_cross_matched)
    np.save(os.path.join(PLOTS_DIR, "selected_neuron_indices.npy"), selected_idx)

    clean_model = load_model(CLEAN_MODEL_PATH, "CLEAN")
    acorn_model = load_model(ACORN_MODEL_PATH, "ACORN")

    print("\n" + "=" * 100)
    print("CLEAN EMBEDDINGS")
    print("=" * 100)
    clean_target_emb = embed(clean_model, X_target_test)
    clean_cross_emb = embed(clean_model, X_cross_matched)
    print("CLEAN target embedding:", clean_target_emb.shape)
    print("CLEAN cross embedding:", clean_cross_emb.shape)
    np.save(os.path.join(PLOTS_DIR, "clean_target_test_embedding.npy"), clean_target_emb)
    np.save(os.path.join(PLOTS_DIR, "clean_cross_session_embedding.npy"), clean_cross_emb)
    plot_pca_comparison(clean_target_emb, clean_cross_emb, "clean")

    print("\n" + "=" * 100)
    print("ACORN EMBEDDINGS")
    print("=" * 100)
    acorn_target_emb = embed(acorn_model, X_target_test)
    acorn_cross_emb = embed(acorn_model, X_cross_matched)
    print("ACORN target embedding:", acorn_target_emb.shape)
    print("ACORN cross embedding:", acorn_cross_emb.shape)
    np.save(os.path.join(PLOTS_DIR, "acorn_target_test_embedding.npy"), acorn_target_emb)
    np.save(os.path.join(PLOTS_DIR, "acorn_cross_session_embedding.npy"), acorn_cross_emb)
    plot_pca_comparison(acorn_target_emb, acorn_cross_emb, "acorn")

    del clean_model
    del acorn_model
    del clean_target_emb
    del clean_cross_emb
    del acorn_target_emb
    del acorn_cross_emb
    del X_target_test
    del other_test_corpus
    del X_cross_random
    del X_cross_matched
    gc.collect()

    print("\n" + "#" * 120)
    print("DONE")
    print("#" * 120)
    print("Models loaded from:")
    print(CLEAN_MODEL_PATH)
    print(ACORN_MODEL_PATH)
    print("\nPlots saved in:")
    print(PLOTS_DIR)

if __name__ == "__main__":
    main()
