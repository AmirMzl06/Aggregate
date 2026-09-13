import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from utils.constants import CEBRA_DIR

sys.path.insert(0, str(CEBRA_DIR))

import cebra


# ==========================
# PATHS
# ==========================

MODEL_DIR = "./models"

CLEAN_MODEL = os.path.join(MODEL_DIR, "clean.pt")
ACORN_MODEL = os.path.join(MODEL_DIR, "acorn.pt")

OUT = "cross_embedding_plots"
os.makedirs(OUT, exist_ok=True)


DATA_ROOT = "/data/hossein/mm_project/perich_data_valid_final_raw"

TARGET = "C-CO12"

N_NEURONS = 86
SEED = 0


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================
# Load sessions
# ==========================

def load_session(name):

    path = os.path.join(DATA_ROOT, name + ".npz")

    d = np.load(path)

    X_train = d["train_data"].astype(np.float32)
    X_test  = d["valid_data"].astype(np.float32)

    return X_train, X_test



# ==========================
# Mean std matching
# ==========================

def match_distribution(X, ref):

    """
    Match neuron-wise mean/std
    X: other session
    ref: CCO12
    """

    mu_x = X.mean(axis=0)
    std_x = X.std(axis=0)

    mu_r = ref.mean(axis=0)
    std_r = ref.std(axis=0)


    X = (X - mu_x) / (std_x + 1e-8)

    X = X * std_r + mu_r

    return X



# ==========================
# random 86 neurons
# ==========================

def select_neurons(X, idx):

    return X[:, idx]



# ==========================
# Build model
# ==========================

def load_model(weight):

    model = cebra.models.init(
        name="offset10-model",
        num_neurons=N_NEURONS,
        num_units=256,
        num_output=14
    )

    state = torch.load(weight, map_location="cpu")

    if "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    return model



# ==========================
# Embedding
# ==========================

def embedding(model, X):

    x = torch.tensor(X).float().to(DEVICE)

    with torch.no_grad():

        z = model(x)

        if isinstance(z, tuple):
            z = torch.cat(z, dim=1)

    return z.cpu().numpy()



# ==========================
# PCA plots
# ==========================

def plot_pca(real, cross, name):

    X = np.concatenate(
        [real, cross],
        axis=0
    )

    pca2 = PCA(n_components=2)
    pca3 = PCA(n_components=3)


    p2 = pca2.fit_transform(X)

    p3 = pca3.fit_transform(X)


    n = len(real)


    # ---------- 2D ----------

    plt.figure(figsize=(7,6))

    plt.scatter(
        p2[:n,0],
        p2[:n,1],
        s=5,
        alpha=0.5,
        label="C-CO12"
    )

    plt.scatter(
        p2[n:,0],
        p2[n:,1],
        s=5,
        alpha=0.5,
        label="Other C-CO"
    )


    plt.legend()
    plt.title(name+" PCA 2D")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUT,
            name+"_PCA2D.png"
        ),
        dpi=200
    )

    plt.close()



    # ---------- 3D ----------

    fig = plt.figure(figsize=(8,7))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    ax.scatter(
        p3[:n,0],
        p3[:n,1],
        p3[:n,2],
        s=5,
        label="C-CO12"
    )

    ax.scatter(
        p3[n:,0],
        p3[n:,1],
        p3[n:,2],
        s=5,
        label="Other C-CO"
    )


    ax.legend()
    ax.set_title(name+" PCA 3D")


    plt.savefig(
        os.path.join(
            OUT,
            name+"_PCA3D.png"
        ),
        dpi=200
    )

    plt.close()



# ==========================
# MAIN
# ==========================


print("Loading CCO12")

_, X_test = load_session(TARGET)


print(
    "CCO12:",
    X_test.shape
)



# choose random neurons
rng = np.random.default_rng(SEED)

neuron_idx = rng.choice(
    X_test.shape[1],
    N_NEURONS,
    replace=False
)


X_ref = X_test[:, neuron_idx]


print(
    "Selected neurons:",
    neuron_idx
)



# other sessions

sessions = []

for i in range(53):

    name=f"C-CO{i}"

    if name == TARGET:
        continue

    try:
        _, Xt = load_session(name)

        Xt = Xt[:, neuron_idx]

        sessions.append(Xt)

    except:
        pass



cross = np.concatenate(
    sessions,
    axis=0
)


print(
    "Before matching:",
    cross.shape
)



cross = match_distribution(
    cross,
    X_ref
)


print(
    "After matching mean/std"
)



# ==========================
# Run both models
# ==========================


for name,weight in [
    ("CLEAN", CLEAN_MODEL),
    ("ACORN", ACORN_MODEL)
]:

    print("Running", name)


    model = load_model(weight)


    emb_real = embedding(
        model,
        X_ref
    )


    emb_cross = embedding(
        model,
        cross
    )


    print(
        emb_real.shape,
        emb_cross.shape
    )


    np.save(
        os.path.join(
            OUT,
            name+"_real_embedding.npy"
        ),
        emb_real
    )


    np.save(
        os.path.join(
            OUT,
            name+"_cross_embedding.npy"
        ),
        emb_cross
    )


    plot_pca(
        emb_real,
        emb_cross,
        name
    )


print("DONE")
