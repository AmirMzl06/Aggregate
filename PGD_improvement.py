"""
Perich ACORN PGD improvement grid -- ALL monkeys / tasks

Pipeline (per selected session):
- Train CLEAN CEBRA+LABEL once
- Fixed epsilon=0.2
- Grid over 4 attack configurations (reference / positive / negative / ref_negative,
  each with 3 restarts + best iterate)
- Train decoder for CLEAN and every adversarial model -> test R2
- Compute and save Jacobian for CLEAN and every ACORN setting
- No embedding plots.

Sessions: 2 random sessions from each monkey+task group.

Fork:
CEBRA-PGDimprovement
"""

from pathlib import Path
import sys, os, gc, random, importlib, csv
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

PERICH_DAYS = {
    "C-CO": range(53),
    "C-RT": range(15),
    "J-CO": range(3),
    "T-CO": range(6),
    "T-RT": range(6),
    "M-CO": range(22),
    "M-RT": range(6),
}

SESSIONS_PER_GROUP = 2
SESSION_PICK_SEED = 42

OUT = ROOT / "ACORN_PGD_GRID_ALL_SESSIONS"
OUT.mkdir(exist_ok=True, parents=True)

RESULTS_CSV = OUT / "all_sessions_r2.csv"

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
    ("Acorn", dict(attack_target="reference", adv_restarts=1, adv_best_iterate=False)),
    ("reference_r3_best", dict(attack_target="reference", adv_restarts=3, adv_best_iterate=True)),
    ("positive_r3_best", dict(attack_target="positive", adv_restarts=3, adv_best_iterate=True)),
    ("negative_r3_best", dict(attack_target="negative", adv_restarts=3, adv_best_iterate=True)),
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


def session_path(name):
    return PERICH_DATA_DIR / f"{name}.npz"


def pick_sessions():
    """2 random existing sessions from each monkey+task group."""
    rng = random.Random(SESSION_PICK_SEED)
    picked = []
    for group, days in PERICH_DAYS.items():
        available = [f"{group}{d}" for d in days if session_path(f"{group}{d}").exists()]
        missing = len(list(days)) - len(available)
        if missing:
            print(f"[{group}] {missing} of {len(list(days))} session files not found on disk")
        if not available:
            print(f"[{group}] SKIP -- no session files found")
            continue
        n = min(SESSIONS_PER_GROUP, len(available))
        if n < SESSIONS_PER_GROUP:
            print(f"[{group}] only {n} session(s) available, wanted {SESSIONS_PER_GROUP}")
        chosen = rng.sample(available, n)
        picked.extend(chosen)
        print(f"[{group}] picked: {chosen}")
    return picked


def load_data(session):
    d = np.load(session_path(session), allow_pickle=True)
    Xtr = d["train_data"].astype(np.float32)
    Xte = d["valid_data"].astype(np.float32)
    Ytr = d["train_label"].astype(np.float32)
    Yte = d["valid_label"].astype(np.float32)
    if Ytr.ndim == 1:
        Ytr = Ytr[:, None]
    if Yte.ndim == 1:
        Yte = Yte[:, None]
    return Xtr, Xte, Ytr, Yte


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
            nn.Linear(dim, MLP_HIDDEN),
            nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROP),
            nn.Linear(MLP_HIDDEN, out)
        )
    def forward(self, x):
        return self.net(x)


def train_decoder(Z, Y):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoLayerMLP(Z.shape[1], Y.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.MSELoss()

    z = torch.tensor(np.asarray(Z, dtype=np.float32), device=dev)
    y = torch.tensor(np.asarray(Y, dtype=np.float32), device=dev)

    for _ in range(MLP_EPOCHS):
        opt.zero_grad()
        l = loss(model(z), y)
        l.backward()
        opt.step()
    return model


def eval_decoder(dec, Z, Y):
    dev = next(dec.parameters()).device
    with torch.no_grad():
        p = dec(torch.tensor(np.asarray(Z, dtype=np.float32), device=dev)).cpu().numpy()
    return float(np.mean([r2_score(Y[:, i], p[:, i]) for i in range(Y.shape[1])]))


def compute_jacobian(model, X, name, session_dir):
    net = model.solver_.model
    dev = next(net.parameters()).device
    inp = torch.tensor(X[:128], dtype=torch.float32, device=dev, requires_grad=True)
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=net,
        input_data=inp,
        output_dimension=LATENT_DIM
    )
    with torch.enable_grad():
        res = method.compute_attribution_map(batch_size=16)
    jf = np.abs(np.asarray(res["jf"]))
    np.save(session_dir / f"{name}_JF.npy", jf)

    plt.figure(figsize=(10, 6))
    jf_plot = jf
    while jf_plot.ndim > 2:
        jf_plot = np.mean(jf_plot, axis=0)
    plt.imshow(jf_plot, aspect="auto")
    plt.colorbar()
    plt.title(name)
    plt.savefig(session_dir / f"{name}_JF.png", dpi=300, bbox_inches="tight")
    plt.close()

    del inp, method, res
    cleanup()


def run_session(session):
    print("\n" + "#" * 100)
    print("SESSION:", session)
    print("#" * 100)

    session_dir = OUT / session
    session_dir.mkdir(exist_ok=True, parents=True)

    seed_all()
    Xtr, Xte, Ytr, Yte = load_data(session)
    print("X_train:", Xtr.shape, "| X_test:", Xte.shape,
          "| Y_train:", Ytr.shape, "| Y_test:", Yte.shape)

    results = {}

    # CLEAN ONLY ONCE
    print("\nTRAIN: CLEAN")
    clean = build_clean()
    clean.fit(Xtr, Ytr)

    Ztr = clean.transform(Xtr)
    Zte = clean.transform(Xte)

    dec = train_decoder(Ztr, Ytr)
    results["CLEAN"] = eval_decoder(dec, Zte, Yte)
    print(f"[{session}] CLEAN R2 = {results['CLEAN']:.6f}")

    compute_jacobian(clean, Xtr, "CLEAN", session_dir)

    del clean, dec
    cleanup()

    for name, cfg in ATTACK_GRID:
        print("\nTRAIN:", name)

        model = build_adv(cfg)
        model.fit(Xtr, Ytr)

        Ztr = model.transform(Xtr)
        Zte = model.transform(Xte)

        dec = train_decoder(Ztr, Ytr)
        results[name] = eval_decoder(dec, Zte, Yte)
        print(f"[{session}] {name} R2 = {results[name]:.6f}")

        compute_jacobian(model, Xtr, name, session_dir)

        del model, dec
        cleanup()

    # per-session summary bar chart across all configs
    labels = list(results.keys())
    values = [results[k] for k in labels]
    plt.figure(figsize=(11, 6))
    colors = ["tab:blue"] + ["tab:red"] * (len(labels) - 1)
    plt.bar(labels, values, color=colors)
    plt.ylabel("Decoder Mean R2")
    plt.title(f"{session} -- CLEAN vs ACORN attack configs")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(session_dir / "R2_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[{session}] RESULTS")
    for k, v in results.items():
        print(f"  {k:<24} {v:.6f}")

    return results


def save_all_results(all_results):
    config_names = ["CLEAN"] + [name for name, _ in ATTACK_GRID]

    with RESULTS_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session"] + config_names)
        for session, res in all_results.items():
            w.writerow([session] + [res.get(c, float("nan")) for c in config_names])
    print("\nsaved:", RESULTS_CSV)

    sessions = list(all_results.keys())
    x = np.arange(len(sessions))
    width = 0.8 / len(config_names)

    plt.figure(figsize=(max(12, 1.2 * len(sessions)), 7))
    for i, cfg_name in enumerate(config_names):
        vals = [all_results[s].get(cfg_name, np.nan) for s in sessions]
        plt.bar(x + i * width, vals, width, label=cfg_name)
    plt.xticks(x + 0.4 - width / 2, sessions, rotation=45, ha="right")
    plt.ylabel("Decoder Mean R2")
    plt.title("All sessions -- CLEAN vs ACORN attack configs")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = OUT / "all_sessions_R2_grouped.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", path)

    plt.figure(figsize=(11, 7))
    for cfg_name in config_names:
        vals = [all_results[s].get(cfg_name, np.nan) for s in sessions]
        plt.plot(sessions, vals, marker="o", label=cfg_name)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Decoder Mean R2")
    plt.title("All sessions -- R2 per attack config")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = OUT / "all_sessions_R2_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("saved:", path)

    print("\n" + "=" * 100)
    print("FINAL MEAN R2 ACROSS ALL SESSIONS")
    print("=" * 100)
    for cfg_name in config_names:
        vals = np.array([all_results[s].get(cfg_name, np.nan) for s in sessions], dtype=float)
        finite = vals[np.isfinite(vals)]
        if len(finite):
            print(f"  {cfg_name:<24} mean={finite.mean():.6f}  std={finite.std():.6f}  n={len(finite)}")
        else:
            print(f"  {cfg_name:<24} no finite results")


def main():
    sessions = pick_sessions()
    print("\n" + "=" * 100)
    print(f"TOTAL SESSIONS SELECTED: {len(sessions)}")
    print(sessions)
    print("=" * 100)

    all_results = {}
    for session in sessions:
        try:
            all_results[session] = run_session(session)
        except Exception as exc:
            print(f"\n[{session}] FAILED: {type(exc).__name__}: {exc}")
            cleanup()
            continue

    if not all_results:
        raise RuntimeError("No session completed successfully.")

    save_all_results(all_results)

    print("\nDONE")
    print("Output:", OUT)


if __name__ == "__main__":
    main()

# """
# C-CO12 ACORN PGD improvement grid

# Pipeline:
# - Train CLEAN CEBRA+LABEL once
# - Fixed epsilon=0.2
# - Grid over attack_target / restarts / best iterate
# - Train decoder for CLEAN and every adversarial model
# - Plot R2 curves: CLEAN vs each ACORN
# - Compute and save Jacobian for CLEAN and every ACORN setting

# Fork:
# CEBRA-PGDimprovement
# """

# from pathlib import Path
# import sys, os, gc, random, importlib
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score

# ROOT = Path(__file__).resolve().parent
# CEBRA_DIR = ROOT / "CEBRA-PGDimprovement"

# for m in list(sys.modules):
#     if m == "cebra" or m.startswith("cebra."):
#         del sys.modules[m]

# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# from cebra import CEBRA
# import cebra.attribution

# print("Using:", cebra.__file__)


# PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
# SESSION = "C-CO12"
# NPZ_PATH = PERICH_DATA_DIR / f"{SESSION}.npz"

# OUT = ROOT / "ACORN_PGD_GRID_CCO12"
# OUT.mkdir(exist_ok=True, parents=True)

# EPSILON = 0.2
# EPS_STEPS = 10

# LATENT_DIM = 64
# HIDDEN = 64
# BATCH_SIZE = 2048
# MAX_ITER = 3000
# TEMP = 0.4
# ARCH = "offset36-model-more-dropout"

# MLP_EPOCHS = 2500
# MLP_HIDDEN = 64
# MLP_DROP = 0.4

# DEVICE = "cuda_if_available"

# ATTACK_GRID = [
#     ("reference_r1_last", dict(attack_target="reference", adv_restarts=1, adv_best_iterate=False)),
#     ("reference_r1_best", dict(attack_target="reference", adv_restarts=1, adv_best_iterate=True)),
#     ("reference_r3_best", dict(attack_target="reference", adv_restarts=3, adv_best_iterate=True)),
#     ("positive_r3_best", dict(attack_target="positive", adv_restarts=3, adv_best_iterate=True)),
#     ("negative_r3_best", dict(attack_target="negative", adv_restarts=3, adv_best_iterate=True)),
#     ("ref_positive_r3_best", dict(attack_target="ref_positive", adv_restarts=3, adv_best_iterate=True)),
#     ("ref_negative_r3_best", dict(attack_target="ref_negative", adv_restarts=3, adv_best_iterate=True)),
# ]


# def seed_all(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)


# def cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# def load_data():
#     d = np.load(NPZ_PATH, allow_pickle=True)
#     return (
#         d["train_data"].astype(np.float32),
#         d["valid_data"].astype(np.float32),
#         d["train_label"].astype(np.float32),
#         d["valid_label"].astype(np.float32),
#     )


# def build_clean():
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMP,
#         model_architecture=ARCH,
#         time_offsets=1,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=HIDDEN,
#         training_mode="standard",
#         device=DEVICE,
#         verbose=True,
#     )


# def build_adv(cfg):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=TEMP,
#         model_architecture=ARCH,
#         time_offsets=1,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         num_hidden_units=HIDDEN,
#         training_mode="adversarial",
#         adv_epsilon=EPSILON,
#         adv_alpha=EPSILON/5,
#         adv_steps=EPS_STEPS,
#         attack_norm="linf",
#         device=DEVICE,
#         verbose=True,
#         **cfg
#     )


# class TwoLayerMLP(nn.Module):
#     def __init__(self, dim=64, out=6):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim,64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Dropout(MLP_DROP),
#             nn.Linear(64,out)
#         )
#     def forward(self,x):
#         return self.net(x)


# def train_decoder(Z,Y):
#     dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model=TwoLayerMLP(Z.shape[1],Y.shape[1]).to(dev)
#     opt=torch.optim.Adam(model.parameters(),lr=1e-3)
#     loss=torch.nn.MSELoss()

#     z=torch.tensor(Z,device=dev)
#     y=torch.tensor(Y,device=dev)

#     for _ in range(MLP_EPOCHS):
#         opt.zero_grad()
#         l=loss(model(z),y)
#         l.backward()
#         opt.step()
#     return model


# def eval_decoder(dec,Z,Y):
#     dev=next(dec.parameters()).device
#     with torch.no_grad():
#         p=dec(torch.tensor(Z,device=dev)).cpu().numpy()
#     return float(np.mean([r2_score(Y[:,i],p[:,i]) for i in range(Y.shape[1])]))


# def compute_jacobian(model,X,name):
#     net=model.solver_.model
#     dev=next(net.parameters()).device
#     inp=torch.tensor(X[:128],dtype=torch.float32,device=dev,requires_grad=True)
#     method=cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=net,
#         input_data=inp,
#         output_dimension=LATENT_DIM
#     )
#     with torch.enable_grad():
#         res=method.compute_attribution_map(batch_size=16)
#     jf=np.abs(np.asarray(res["jf"]))
#     np.save(OUT/f"{name}_JF.npy",jf)

#     plt.figure(figsize=(10,6))
#     # plt.imshow(jf.squeeze(),aspect="auto")
#     jf_plot = jf
#     while jf_plot.ndim > 2:
#         jf_plot = np.mean(jf_plot, axis=0)
#     plt.imshow(jf_plot, aspect="auto")
#     plt.colorbar()
#     plt.title(name)
#     plt.savefig(OUT/f"{name}_JF.png",dpi=300,bbox_inches="tight")
#     plt.close()


# def main():

#     seed_all()
#     Xtr,Xte,Ytr,Yte=load_data()

#     results={}

#     # CLEAN ONLY ONCE
#     clean=build_clean()
#     clean.fit(Xtr,Ytr)

#     Ztr=clean.transform(Xtr)
#     Zte=clean.transform(Xte)

#     dec=train_decoder(Ztr,Ytr)
#     results["CLEAN"]=eval_decoder(dec,Zte,Yte)

#     compute_jacobian(clean,Xtr,"CLEAN")

#     for name,cfg in ATTACK_GRID:

#         print("\nTRAIN:",name)

#         model=build_adv(cfg)
#         model.fit(Xtr,Ytr)

#         Ztr=model.transform(Xtr)
#         Zte=model.transform(Xte)

#         dec=train_decoder(Ztr,Ytr)
#         results[name]=eval_decoder(dec,Zte,Yte)

#         compute_jacobian(model,Xtr,name)

#         plt.figure(figsize=(6,5))
#         plt.bar(["CLEAN",name],[results["CLEAN"],results[name]])
#         plt.ylabel("Decoder Mean R2")
#         plt.title(name)
#         plt.savefig(OUT/f"R2_{name}.png",dpi=300,bbox_inches="tight")
#         plt.close()

#         plt.figure(figsize=(8,5))
#         plt.plot([0,1],[results["CLEAN"],results[name]],marker="o",
#                  label="CLEAN vs ACORN")
#         plt.legend()
#         plt.ylabel("Mean R2")
#         plt.title(name)
#         plt.savefig(OUT/f"Curve_{name}.png",dpi=300,bbox_inches="tight")
#         plt.close()

#         del model,dec
#         cleanup()

#     print("\nFINAL R2")
#     for k,v in results.items():
#         print(k,v)


# if __name__=="__main__":
#     main()
