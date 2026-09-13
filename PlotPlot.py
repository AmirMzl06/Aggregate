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
from mpl_toolkits.mplot3d import Axes3D


# =====================================================
# CONFIG
# =====================================================

PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"

DATASET_NAME = "C-CO"
TARGET_DAY = 12
TARGET_SESSION = "C-CO12"

N_NEURONS = 86
N_SESSIONS = 53

SEED = 42

MODELS_DIR = "models"
PLOTS_DIR = "plots_Cross"


# =====================================================
# LOAD DATA
# =====================================================

def session_path(name):
    return os.path.join(
        PERICH_DATA_DIR,
        f"{name}.npz"
    )


def load_test(session):

    path = session_path(session)

    data = np.load(path)

    X_test = data["valid_data"].astype(np.float32)

    return X_test



# =====================================================
# LOAD MODELS
# =====================================================

def load_model(name):

    path = os.path.join(
        MODELS_DIR,
        f"{name}.pt"
    )

    model = CEBRA.load(path)

    print("loaded:", path)

    return model



# =====================================================
# RANDOM NEURON PICKING
# =====================================================

def choose_neurons_per_session(
        session_ids,
        total_neurons=86,
        seed=42):

    """
    Allocate 86 neurons randomly across sessions.
    """

    rng = np.random.default_rng(seed)


    allocation = {}

    remaining = total_neurons


    usable = session_ids.copy()


    while remaining > 0:

        s = rng.choice(usable)

        allocation[s] = allocation.get(s,0)+1

        remaining -= 1


    return allocation



def get_cross_session_test():

    """
    Take ONLY test sets.
    Random neurons from each session.
    Total = 86 neurons.
    """

    sessions = [
        i for i in range(N_SESSIONS)
        if i != TARGET_DAY
    ]


    allocation = choose_neurons_per_session(
        sessions,
        N_NEURONS,
        SEED
    )


    print("\nNeuron allocation:")
    print(allocation)



    selected=[]


    for day,n in allocation.items():

        session=f"{DATASET_NAME}{day}"

        X=load_test(session)


        rng=np.random.default_rng(
            SEED+day
        )


        idx=rng.choice(
            X.shape[1],
            size=n,
            replace=False
        )


        X=X[:,idx]


        selected.append(X)


        print(
            session,
            "test:",
            X.shape
        )



    X_cross=np.concatenate(
        selected,
        axis=1
    )


    print(
        "cross session shape:",
        X_cross.shape
    )


    return X_cross



# =====================================================
# NORMALIZATION USING CCO12 TEST
# =====================================================

def normalize_with_cco12(
        X,
        ref):


    mu=ref.mean(axis=0)

    std=ref.std(axis=0)


    std[std==0]=1


    Xn=(X-mu)/std


    return Xn.astype(np.float32)



# =====================================================
# EMBEDDING
# =====================================================


def embed(model,X):

    return np.asarray(
        model.transform(
            X.astype(np.float32)
        )
    )



# =====================================================
# PCA PLOT
# =====================================================


def pca_plot(
        A,
        B,
        title,
        name):


    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )


    pca=PCA(
        n_components=3
    )


    Z=np.concatenate(
        [A,B],
        axis=0
    )


    pca.fit(Z)


    ZA=pca.transform(A)
    ZB=pca.transform(B)



    # -------- 2D --------

    plt.figure(figsize=(7,6))

    plt.scatter(
        ZA[:,0],
        ZA[:,1],
        s=5,
        label="C-CO12"
    )

    plt.scatter(
        ZB[:,0],
        ZB[:,1],
        s=5,
        label="cross sessions"
    )


    plt.legend()

    plt.title(
        title+" PCA 2D"
    )


    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            name+"_2D.png"
        ),
        dpi=200
    )

    plt.close()



    # -------- 3D --------

    fig=plt.figure(
        figsize=(8,7)
    )

    ax=fig.add_subplot(
        111,
        projection="3d"
    )


    ax.scatter(
        ZA[:,0],
        ZA[:,1],
        ZA[:,2],
        s=5,
        label="C-CO12"
    )


    ax.scatter(
        ZB[:,0],
        ZB[:,1],
        ZB[:,2],
        s=5,
        label="cross sessions"
    )


    ax.legend()

    ax.set_title(
        title+" PCA 3D"
    )


    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            name+"_3D.png"
        ),
        dpi=200
    )

    plt.close()



# =====================================================
# MAIN
# =====================================================


def main():


    seed=SEED

    random.seed(seed)
    np.random.seed(seed)


    clean=load_model("clean")

    acorn=load_model("acorn")



    # CCO12 test

    X12=load_test(
        TARGET_SESSION
    )


    # cross session test only

    Xcross=get_cross_session_test()



    # sanity

    print(
        X12.shape,
        Xcross.shape
    )



    for name,model in [
        ("clean",clean),
        ("acorn",acorn)
    ]:


        print("\nMODEL:",name)



        # ==============================
        # CASE 1
        # NO NORMALIZATION
        # ==============================


        e12=embed(
            model,
            X12
        )


        ecross=embed(
            model,
            Xcross
        )


        pca_plot(
            e12,
            ecross,
            name+"_raw",
            name+"_raw"
        )



        # ==============================
        # CASE 2
        # CCO12 TEST STD NORMALIZATION
        # ==============================


        X12_norm=normalize_with_cco12(
            X12,
            X12
        )


        Xcross_norm=normalize_with_cco12(
            Xcross,
            X12
        )


        e12n=embed(
            model,
            X12_norm
        )


        ecrossn=embed(
            model,
            Xcross_norm
        )



        pca_plot(
            e12n,
            ecrossn,
            name+"_CCO12std",
            name+"_CCO12std"
        )



    print("DONE")



if __name__=="__main__":
    main()
