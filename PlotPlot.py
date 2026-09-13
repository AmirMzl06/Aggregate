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

N_CCO_SESSIONS = 53

SEED = 42


MODELS_DIR = "models"

PLOTS_DIR = "plots"





# =====================================================
# DATA
# =====================================================


def session_path(session):

    return os.path.join(
        PERICH_DATA_DIR,
        session+".npz"
    )




def load_session(session):

    data=np.load(
        session_path(session),
        allow_pickle=True
    )


    X_train=data["train_data"].astype(
        np.float32
    )


    X_test=data["valid_data"].astype(
        np.float32
    )


    return X_train,X_test





# =====================================================
# MODEL
# =====================================================


def load_model(name):

    path=os.path.join(
        MODELS_DIR,
        name+".pt"
    )


    model=CEBRA.load(path)

    print(
        "loaded:",
        path
    )

    return model





# =====================================================
# SELECT NEURONS
# =====================================================


def select_neurons(
        X,
        n_neurons,
        seed):


    total=X.shape[1]


    if total < n_neurons:

        return None


    if total == n_neurons:

        return X



    rng=np.random.default_rng(
        seed
    )


    idx=np.sort(
        rng.choice(
            total,
            size=n_neurons,
            replace=False
        )
    )


    return X[:,idx]





# =====================================================
# BUILD CROSS SESSION
# =====================================================


def get_sessions():

    return [
        d for d in range(N_CCO_SESSIONS)
        if d != TARGET_DAY
    ]





def allocate_neurons():

    rng=np.random.default_rng(
        SEED
    )


    sessions=get_sessions()


    allocation={}


    remaining=N_NEURONS


    while remaining>0:

        s=int(
            rng.choice(
                sessions
            )
        )


        allocation[s]=allocation.get(
            s,
            0
        )+1


        remaining-=1


    return allocation





def build_cross_test():

    allocation=allocate_neurons()


    print(
        "\nAllocation:"
    )

    print(
        allocation
    )



    blocks=[]



    for day,n in allocation.items():


        session=f"{DATASET_NAME}{day}"


        _,X_test=load_session(
            session
        )


        rng=np.random.default_rng(
            SEED+day
        )


        idx=rng.choice(
            X_test.shape[1],
            size=n,
            replace=False
        )


        X_sel=X_test[:,idx]


        blocks.append(
            X_sel.astype(
                np.float32
            )
        )


        print(
            session,
            X_sel.shape
        )



    # -------------------------------
    # REMOVE EXTRA TIME BINS
    # -------------------------------


    min_time=min(
        x.shape[0]
        for x in blocks
    )


    print(
        "minimum time bins:",
        min_time
    )


    blocks=[
        x[:min_time]
        for x in blocks
    ]



    # -------------------------------
    # CONCATENATE SESSIONS
    # -------------------------------


    X_cross=np.concatenate(
        blocks,
        axis=0
    )



    print(
        "FINAL CROSS:",
        X_cross.shape
    )


    return X_cross





# =====================================================
# NORMALIZE
# =====================================================


def normalize(
        X,
        ref):


    mu=ref.mean(
        axis=0
    )

    std=ref.std(
        axis=0
    )


    std[std==0]=1


    return (
        (X-mu)/std
    ).astype(
        np.float32
    )





# =====================================================
# EMBEDDING
# =====================================================


def embed(
        model,
        X):

    return np.asarray(
        model.transform(
            X
        )
    )





# =====================================================
# PCA PLOT
# =====================================================


def plot_pca(
        A,
        B,
        title,
        name):


    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )


    all_data=np.concatenate(
        [A,B],
        axis=0
    )


    pca=PCA(
        n_components=3
    )


    pca.fit(
        all_data
    )


    A3=pca.transform(A)

    B3=pca.transform(B)



    # 2D

    plt.figure(
        figsize=(7,6)
    )


    plt.scatter(
        A3[:,0],
        A3[:,1],
        s=5,
        label="C-CO12"
    )


    plt.scatter(
        B3[:,0],
        B3[:,1],
        s=5,
        label="Cross"
    )


    plt.legend()

    plt.title(
        title+" 2D"
    )


    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            name+"_2D.png"
        ),
        dpi=200
    )


    plt.close()



    # 3D

    fig=plt.figure(
        figsize=(8,7)
    )


    ax=fig.add_subplot(
        111,
        projection="3d"
    )


    ax.scatter(
        A3[:,0],
        A3[:,1],
        A3[:,2],
        s=5,
        label="C-CO12"
    )


    ax.scatter(
        B3[:,0],
        B3[:,1],
        B3[:,2],
        s=5,
        label="Cross"
    )


    ax.legend()


    ax.set_title(
        title+" 3D"
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


    random.seed(SEED)

    np.random.seed(SEED)



    clean=load_model(
        "clean"
    )


    acorn=load_model(
        "acorn"
    )



    _,X12=load_session(
        TARGET_SESSION
    )



    Xcross=build_cross_test()



    for name,model in [
        ("clean",clean),
        ("acorn",acorn)
    ]:


        print(
            "\nMODEL",
            name
        )



        # -----------------
        # RAW
        # -----------------

        e12=embed(
            model,
            X12
        )


        ecross=embed(
            model,
            Xcross
        )


        plot_pca(
            e12,
            ecross,
            name+"_raw",
            name+"_raw"
        )



        # -----------------
        # STD using CCO12
        # -----------------


        X12n=normalize(
            X12,
            X12
        )


        Xcrossn=normalize(
            Xcross,
            X12
        )



        e12n=embed(
            model,
            X12n
        )


        ecrossn=embed(
            model,
            Xcrossn
        )



        plot_pca(
            e12n,
            ecrossn,
            name+"_CCO12std",
            name+"_std"
        )



    print("DONE")




if __name__=="__main__":

    main()
