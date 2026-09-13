import os
import sys
import random
import numpy as np
import torch
from utils.constants import CEBRA_DIR

sys.path.insert(0, str(CEBRA_DIR))
from cebra import CEBRA
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
TARGET_SESSION = "C-CO12"
N_NEURONS = 86
SEED = 42
MODELS_DIR = "models"
PLOTS_DIR = "plots_Fake"

def session_path(session):
    return os.path.join(PERICH_DATA_DIR, session + ".npz")

def load_session(session):
    data = np.load(session_path(session), allow_pickle=True)
    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)
    return X_train, X_test

def load_model(name):
    path = os.path.join(MODELS_DIR, name + ".pt")
    model = CEBRA.load(path)
    print("loaded:", path)
    return model

def embed(model, X):
    return np.asarray(model.transform(X.astype(np.float32)))

def build_fake_test(X_ref):
    """
    Independent Gaussian fake data.
    Neuron-wise exact mean/std matching.
    """
    rng = np.random.default_rng(SEED)
    X_fake = rng.normal(loc=0.0, scale=1.0, size=X_ref.shape).astype(np.float64)
    fake_mu = X_fake.mean(axis=0, keepdims=True)
    fake_std = X_fake.std(axis=0, keepdims=True)
    ref_mu = X_ref.mean(axis=0, keepdims=True)
    ref_std = X_ref.std(axis=0, keepdims=True)
    fake_std = np.where(fake_std < 1e-12, 1.0, fake_std)
    X_fake = (X_fake - fake_mu) / fake_std
    X_fake = X_fake * ref_std + ref_mu
    return X_fake.astype(np.float32)

def check_matching(X_fake, X_ref):
    fake_mu = X_fake.mean(axis=0)
    fake_std = X_fake.std(axis=0)
    ref_mu = X_ref.mean(axis=0)
    ref_std = X_ref.std(axis=0)
    mean_diff = np.abs(fake_mu - ref_mu)
    std_diff = np.abs(fake_std - ref_std)
    print("\nNeuron-wise matching check")
    print("max |mean_fake - mean_CCO12|:", mean_diff.max())
    print("max |std_fake - std_CCO12|:", std_diff.max())
    print("\nFirst 10 neurons:")
    for i in range(min(10, X_ref.shape[1])):
        print(
            f"Neuron {i:02d} | "
            f"CCO12 mean={ref_mu[i]:.6f}, fake mean={fake_mu[i]:.6f} | "
            f"CCO12 std={ref_std[i]:.6f}, fake std={fake_std[i]:.6f}"
        )

def pca_plot(real_emb, fake_emb, title, filename):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    data = np.concatenate([real_emb, fake_emb], axis=0)
    pca = PCA(n_components=3)
    pca.fit(data)
    Z_real = pca.transform(real_emb)
    Z_fake = pca.transform(fake_emb)
    var = pca.explained_variance_ratio_

    plt.figure(figsize=(7, 6))
    plt.scatter(Z_real[:, 0], Z_real[:, 1], s=6, c="red", alpha=0.5, label="C-CO12 test")
    plt.scatter(Z_fake[:, 0], Z_fake[:, 1], s=6, c="blue", alpha=0.5, label="Fake")
    plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    plt.title(title + " PCA 2D")
    plt.legend()
    plt.tight_layout()
    out_2d = os.path.join(PLOTS_DIR, filename + "_2D.png")
    plt.savefig(out_2d, dpi=200)
    plt.close()
    print("saved:", out_2d)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Z_real[:, 0], Z_real[:, 1], Z_real[:, 2], s=6, c="red", alpha=0.5, label="C-CO12 test")
    ax.scatter(Z_fake[:, 0], Z_fake[:, 1], Z_fake[:, 2], s=6, c="blue", alpha=0.5, label="Fake")
    ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
    ax.set_zlabel(f"PC3 ({var[2] * 100:.1f}%)")
    ax.set_title(title + " PCA 3D")
    ax.legend()
    plt.tight_layout()
    out_3d = os.path.join(PLOTS_DIR, filename + "_3D.png")
    plt.savefig(out_3d, dpi=200)
    plt.close()
    print("saved:", out_3d)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    clean = load_model("clean")
    acorn = load_model("acorn")

    _, X12 = load_session(TARGET_SESSION)
    print("\nC-CO12 test:", X12.shape)

    if X12.shape[1] != N_NEURONS:
        raise ValueError(f"Expected {N_NEURONS} neurons, got {X12.shape[1]}")

    Xfake = build_fake_test(X12)
    print("Fake test:", Xfake.shape)
    check_matching(Xfake, X12)

    for name, model in [("clean", clean), ("acorn", acorn)]:
        print("\nMODEL:", name)
        emb_real = embed(model, X12)
        emb_fake = embed(model, Xfake)
        print("real embedding:", emb_real.shape)
        print("fake embedding:", emb_fake.shape)
        pca_plot(emb_real, emb_fake, name.upper() + " real vs fake", name + "_fake")

    print("\nDONE")
    print("plots:", os.path.abspath(PLOTS_DIR))

if __name__ == "__main__":
    main()




# import os
# import sys
# import random
# import numpy as np
# import torch
# from utils.constants import CEBRA_DIR

# sys.path.insert(0, str(CEBRA_DIR))
# from cebra import CEBRA
# from sklearn.decomposition import PCA
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
# DATASET_NAME = "C-CO"
# TARGET_DAY = 12
# TARGET_SESSION = "C-CO12"
# N_NEURONS = 86
# N_SESSIONS = 53
# SEED = 42
# MODELS_DIR = "models"
# PLOTS_DIR = "plots_Cross"

# def session_path(session):
#     return os.path.join(PERICH_DATA_DIR, session + ".npz")

# def load_session(session):
#     data = np.load(session_path(session), allow_pickle=True)
#     X_train = data["train_data"].astype(np.float32)
#     X_test = data["valid_data"].astype(np.float32)
#     return X_train, X_test

# def load_model(name):
#     path = os.path.join(MODELS_DIR, name + ".pt")
#     model = CEBRA.load(path)
#     print("loaded:", path)
#     return model

# def embed(model, X):
#     return np.asarray(model.transform(X.astype(np.float32)))

# def get_sessions():
#     return [i for i in range(N_SESSIONS) if i != TARGET_DAY]

# def allocate_neurons():
#     rng = np.random.default_rng(SEED)
#     sessions = get_sessions()
#     allocation = {}
#     remain = N_NEURONS
#     while remain > 0:
#         s = int(rng.choice(sessions))
#         allocation[s] = allocation.get(s, 0) + 1
#         remain -= 1
#     return allocation

# def build_cross_test():
#     allocation = allocate_neurons()
#     print("\nNeuron allocation")
#     print(allocation)
#     blocks = []
#     time_lengths = []
#     for day, n in allocation.items():
#         session = f"{DATASET_NAME}{day}"
#         _, X_test = load_session(session)
#         available = X_test.shape[1]
#         if available < n:
#             print("skip", session, "has only", available)
#             continue
#         rng = np.random.default_rng(SEED + day)
#         idx = rng.choice(available, size=n, replace=False)
#         X_sel = X_test[:, idx]
#         blocks.append(X_sel.astype(np.float32))
#         time_lengths.append(X_sel.shape[0])
#         print(session, X_sel.shape)
#     min_time = min(time_lengths)
#     print("minimum time:", min_time)
#     trimmed = []
#     for X in blocks:
#         trimmed.append(X[:min_time])
#     X_cross = np.concatenate(trimmed, axis=1)
#     print("FINAL CROSS:", X_cross.shape)
#     return X_cross

# def normalize_using_reference(X, ref):
#     mu_x = X.mean(axis=0, keepdims=True)
#     std_x = X.std(axis=0, keepdims=True)
#     mu_ref = ref.mean(axis=0, keepdims=True)
#     std_ref = ref.std(axis=0, keepdims=True)
#     std_x = np.where(std_x < 1e-8, 1.0, std_x)
#     X_matched = (X - mu_x) / std_x
#     X_matched = X_matched * std_ref + mu_ref
#     return X_matched.astype(np.float32)

# def pca_plot(A, B, title, filename):
#     os.makedirs(PLOTS_DIR, exist_ok=True)
#     data = np.concatenate([A, B], axis=0)
#     pca = PCA(n_components=3)
#     pca.fit(data)
#     ZA = pca.transform(A)
#     ZB = pca.transform(B)

#     plt.figure(figsize=(7, 6))
#     plt.scatter(ZA[:, 0], ZA[:, 1], s=5, label="C-CO12")
#     plt.scatter(ZB[:, 0], ZB[:, 1], s=5, label="Cross")
#     plt.legend()
#     plt.title(title + " PCA 2D")
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOTS_DIR, filename + "_2D.png"), dpi=200)
#     plt.close()

#     fig = plt.figure(figsize=(8, 7))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.scatter(ZA[:, 0], ZA[:, 1], ZA[:, 2], s=5, label="C-CO12")
#     ax.scatter(ZB[:, 0], ZB[:, 1], ZB[:, 2], s=5, label="Cross")
#     ax.legend()
#     ax.set_title(title + " PCA 3D")
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOTS_DIR, filename + "_3D.png"), dpi=200)
#     plt.close()

# def main():
#     random.seed(SEED)
#     np.random.seed(SEED)
#     clean = load_model("clean")
#     acorn = load_model("acorn")
#     _, X12 = load_session(TARGET_SESSION)
#     print("CCO12:", X12.shape)
#     Xcross = build_cross_test()
#     print("Cross:", Xcross.shape)
#     for name, model in [("clean", clean), ("acorn", acorn)]:
#         print("\nMODEL:", name)
#         emb12 = embed(model, X12)
#         embcross = embed(model, Xcross)
#         pca_plot(emb12, embcross, name + "_RAW", name + "_raw")
#         X12_std = normalize_using_reference(X12, X12)
#         Xcross_std = normalize_using_reference(Xcross, X12)
#         emb12_std = embed(model, X12_std)
#         embcross_std = embed(model, Xcross_std)
#         pca_plot(emb12_std, embcross_std, name + "_STD", name + "_std")
#     print("\nDONE")

# if __name__ == "__main__":
#     main()
