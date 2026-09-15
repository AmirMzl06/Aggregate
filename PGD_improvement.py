
"""
C-CO12 ACORN PGD improvement grid

Pipeline:
- Train CLEAN CEBRA+LABEL once
- Fixed epsilon=0.2
- Grid over attack_target / restarts / best iterate
- Train decoder for CLEAN and every adversarial model
- Plot R2 curves: CLEAN vs each ACORN
- Compute and save Jacobian for CLEAN and every ACORN setting

Fork:
CEBRA-PGDimprovement
"""

from pathlib import Path
import sys, os, gc, random, importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
CEBRA_DIR = ROOT / "CEBRA-PGDimprovement"

for m in list(sys.modules):
    if m == "cebra" or m.startswith("cebra."):
        del sys.modules[m]

sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra import CEBRA
import cebra.attribution

print("Using:", cebra.__file__)


PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
SESSION = "C-CO12"
NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"

OUT = ROOT / "ACORN_PGD_GRID_CCO12"
OUT.mkdir(exist_ok=True, parents=True)

EPSILON = 0.2
EPS_STEPS = 10

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMP = 0.4
ARCH = "offset36-model-more-dropout"

MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4

DEVICE = "cuda_if_available"

ATTACK_GRID = [
    ("reference_r1_last", dict(attack_target="reference", adv_restarts=1, adv_best_iterate=False)),
    ("reference_r1_best", dict(attack_target="reference", adv_restarts=1, adv_best_iterate=True)),
    ("reference_r3_best", dict(attack_target="reference", adv_restarts=3, adv_best_iterate=True)),
    ("positive_r3_best", dict(attack_target="positive", adv_restarts=3, adv_best_iterate=True)),
    ("negative_r3_best", dict(attack_target="negative", adv_restarts=3, adv_best_iterate=True)),
    ("ref_positive_r3_best", dict(attack_target="ref_positive", adv_restarts=3, adv_best_iterate=True)),
    ("ref_negative_r3_best", dict(attack_target="ref_negative", adv_restarts=3, adv_best_iterate=True)),
]


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_data():
    d = np.load(NPZ_PATH, allow_pickle=True)
    return (
        d["train_data"].astype(np.float32),
        d["valid_data"].astype(np.float32),
        d["train_label"].astype(np.float32),
        d["valid_label"].astype(np.float32),
    )


def build_clean():
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMP,
        model_architecture=ARCH,
        time_offsets=1,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="standard",
        device=DEVICE,
        verbose=True,
    )


def build_adv(cfg):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMP,
        model_architecture=ARCH,
        time_offsets=1,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,
        training_mode="adversarial",
        adv_epsilon=EPSILON,
        adv_alpha=EPSILON/5,
        adv_steps=EPS_STEPS,
        attack_norm="linf",
        device=DEVICE,
        verbose=True,
        **cfg
    )


class TwoLayerMLP(nn.Module):
    def __init__(self, dim=64, out=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(MLP_DROP),
            nn.Linear(64,out)
        )
    def forward(self,x):
        return self.net(x)


def train_decoder(Z,Y):
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TwoLayerMLP(Z.shape[1],Y.shape[1]).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss=torch.nn.MSELoss()

    z=torch.tensor(Z,device=dev)
    y=torch.tensor(Y,device=dev)

    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        l=loss(model(z),y)
        l.backward()
        opt.step()
    return model


def eval_decoder(dec,Z,Y):
    dev=next(dec.parameters()).device
    with torch.no_grad():
        p=dec(torch.tensor(Z,device=dev)).cpu().numpy()
    return float(np.mean([r2_score(Y[:,i],p[:,i]) for i in range(Y.shape[1])]))


def compute_jacobian(model,X,name):
    net=model.solver_.model
    dev=next(net.parameters()).device
    inp=torch.tensor(X[:128],dtype=torch.float32,device=dev,requires_grad=True)
    method=cebra.attribution.init(
        name="jacobian-based-batched",
        model=net,
        input_data=inp,
        output_dimension=LATENT_DIM
    )
    with torch.enable_grad():
        res=method.compute_attribution_map(batch_size=16)
    jf=np.abs(np.asarray(res["jf"]))
    np.save(OUT/f"{name}_JF.npy",jf)

    plt.figure(figsize=(10,6))
    # plt.imshow(jf.squeeze(),aspect="auto")
    jf_plot = jf
    while jf_plot.ndim > 2:
        jf_plot = np.mean(jf_plot, axis=0)
    plt.imshow(jf_plot, aspect="auto")
    plt.colorbar()
    plt.title(name)
    plt.savefig(OUT/f"{name}_JF.png",dpi=300,bbox_inches="tight")
    plt.close()


def main():

    seed_all()
    Xtr,Xte,Ytr,Yte=load_data()

    results={}

    # CLEAN ONLY ONCE
    clean=build_clean()
    clean.fit(Xtr,Ytr)

    Ztr=clean.transform(Xtr)
    Zte=clean.transform(Xte)

    dec=train_decoder(Ztr,Ytr)
    results["CLEAN"]=eval_decoder(dec,Zte,Yte)

    compute_jacobian(clean,Xtr,"CLEAN")

    for name,cfg in ATTACK_GRID:

        print("\nTRAIN:",name)

        model=build_adv(cfg)
        model.fit(Xtr,Ytr)

        Ztr=model.transform(Xtr)
        Zte=model.transform(Xte)

        dec=train_decoder(Ztr,Ytr)
        results[name]=eval_decoder(dec,Zte,Yte)

        compute_jacobian(model,Xtr,name)

        plt.figure(figsize=(6,5))
        plt.bar(["CLEAN",name],[results["CLEAN"],results[name]])
        plt.ylabel("Decoder Mean R2")
        plt.title(name)
        plt.savefig(OUT/f"R2_{name}.png",dpi=300,bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot([0,1],[results["CLEAN"],results[name]],marker="o",
                 label="CLEAN vs ACORN")
        plt.legend()
        plt.ylabel("Mean R2")
        plt.title(name)
        plt.savefig(OUT/f"Curve_{name}.png",dpi=300,bbox_inches="tight")
        plt.close()

        del model,dec
        cleanup()

    print("\nFINAL R2")
    for k,v in results.items():
        print(k,v)


if __name__=="__main__":
    main()
