import os
import sys
import gc
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from utils.constants import CEBRA_DIR

if "cebra" in sys.modules:
    del sys.modules["cebra"]
sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA

ROOT = "data/RecogMemory"
SESSION = "P10HMH_092206"
BLOCK = "NO"
EVENT_DIR = os.path.join(ROOT, "Data/events", SESSION, BLOCK)
SPIKE_DIR = os.path.join(ROOT, "Data/sorted", SESSION, BLOCK)
OUT = "RecogMemory_final_results"
os.makedirs(OUT, exist_ok=True)
SEED = 42
BIN_MS = 50
WINDOW_START = -1000
WINDOW_END = 2000
BATCH_SIZE = 5000
MAX_ITER = 4000
DEVICE = "cuda_if_available"
LATENT_DIM = 64

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_neurons():
    neurons = []
    files = sorted([f for f in os.listdir(SPIKE_DIR) if f.endswith("_cells.mat")])
    for f in files:
        mat = loadmat(os.path.join(SPIKE_DIR, f), squeeze_me=True)
        spikes = mat["spikes"]
        neurons.append(spikes[:, 2].astype(np.float64))
    print("neurons:", len(neurons))
    return neurons

def load_recognition_trials():
    mat = loadmat(os.path.join(EVENT_DIR, "eventsRaw.mat"), squeeze_me=True)
    events = mat["events"]
    rec_events = events[events[:, 2] == 81]
    stim = rec_events[rec_events[:, 1] == 1, 0]
    print("recognition stimuli:", len(stim))
    return stim

def load_labels():
    path = os.path.join(EVENT_DIR, "newold81.txt")
    labels = []
    with open(path) as f:
        for line in f:
            p = line.strip().split(";")
            if len(p) < 2:
                continue
            code = int(p[1])
            if code in [31, 32, 33]:
                labels.append(0)
            elif code in [34, 35, 36]:
                labels.append(1)
    labels = np.array(labels)
    print("labels:", labels.shape, np.unique(labels, return_counts=True))
    return labels

def make_trial_tensor(neurons, stim):
    edges = np.arange(WINDOW_START, WINDOW_END + BIN_MS, BIN_MS)
    trials = []
    for onset in stim:
        trial = []
        for ts in neurons:
            rel = ts - onset
            counts, _ = np.histogram(rel, bins=edges)
            trial.append(counts)
        trials.append(np.array(trial).T)
    return np.array(trials)

def build_model(adv):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        training_mode="adversarial" if adv else "single_session",
        adv_alpha=0.2 if adv else 0,
        adv_epsilon=0.05 if adv else 0,
        adv_steps=10 if adv else 0,
        attack_norm="linf",
        num_hidden_units=64,
        device=DEVICE,
        verbose=True
    )

def get_attribution(model, X):
    net = model.solver_.model.to("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    inp = torch.tensor(X, dtype=torch.float32, device=next(net.parameters()).device)
    inp.requires_grad_(True)
    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=net,
        input_data=inp,
        output_dimension=LATENT_DIM
    )
    result = method.compute_attribution_map(batch_size=128)
    jf = result["jf"]
    if "jf-inv-svd" in result:
        jfinv = result["jf-inv-svd"]
    else:
        jfinv = result["jf-inv"]
    return np.mean(np.abs(jf), axis=0), np.mean(np.abs(jfinv), axis=0)

def plot_compare(clean, adv, name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mat, title in zip(axs, [clean, adv], ["CLEAN", "ADV"]):
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_title(title + " " + name)
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Latent")
    fig.colorbar(im, ax=axs)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"CLEAN_vs_ADV_{name}.png"), dpi=300)
    plt.close()

def main():
    seed_all(SEED)
    neurons = load_neurons()
    stim = load_recognition_trials()
    labels = load_labels()
    X = make_trial_tensor(neurons, stim)
    print("X:", X.shape)
    if len(X) != len(labels):
        raise RuntimeError(f"Trial/label mismatch {len(X)} vs {len(labels)}")
    ntrial, time_bins, nneurons = X.shape
    idx_train, idx_test = train_test_split(
        np.arange(ntrial),
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )
    train_X = X[idx_train].reshape(-1, nneurons)
    test_X = X[idx_test].reshape(-1, nneurons)
    train_y = labels[idx_train]
    test_y = labels[idx_test]
    results = []
    attrs = {}
    for adv in [False, True]:
        name = "ADV" if adv else "CLEAN"
        print("\nTRAINING", name)
        model = build_model(adv)
        model.fit(train_X.astype(np.float32))
        model.save(os.path.join(OUT, name + ".pth"))
        ztrain = model.transform(train_X)
        ztest = model.transform(test_X)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(ztrain, train_y)
        pred = clf.predict(ztest)
        acc = accuracy_score(test_y, pred)
        bacc = balanced_accuracy_score(test_y, pred)
        print(name, acc, bacc)
        np.savez(os.path.join(OUT, name + "_embedding.npz"),
                 ztrain=ztrain, ztest=ztest, train_y=train_y, test_y=test_y)
        jf, jfinv = get_attribution(model, train_X[:10000])
        attrs[name] = (jf, jfinv)
        np.savez(os.path.join(OUT, name + "_attr.npz"), jf=jf, jfinv=jfinv)
        pd.DataFrame(jf).to_csv(os.path.join(OUT, name + "_JF.csv"))
        pd.DataFrame(jfinv).to_csv(os.path.join(OUT, name + "_JFINV.csv"))
        results.append({
            "model": name,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "train_trials": len(idx_train),
            "test_trials": len(idx_test),
            "neurons": nneurons
        })
        del model
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(os.path.join(OUT, "results.csv"), index=False)
    plot_compare(attrs["CLEAN"][0], attrs["ADV"][0], "JF")
    plot_compare(attrs["CLEAN"][1], attrs["ADV"][1], "JFINV")
    print("DONE")

if __name__ == "__main__":
    main()


# import os
# import sys
# import gc
# import random
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from scipy.io import loadmat
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_auc_score

# from utils.constants import CEBRA_DIR
# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# from cebra import CEBRA

# ROOT = "data/RecogMemory"
# SESSION = "P10HMH_092206"
# BLOCK = "NO"
# EVENT_DIR = os.path.join(ROOT, "Data/events", SESSION, BLOCK)
# SPIKE_DIR = os.path.join(ROOT, "Data/sorted", SESSION, BLOCK)
# OUT = "RecogMemory_results"
# os.makedirs(OUT, exist_ok=True)
# DEVICE = "cuda_if_available"
# BIN_MS = 50
# WINDOW_START = -1000
# WINDOW_END = 2000
# TRAIN_RATIO = 0.8
# BATCH_SIZE = 5000
# MAX_ITER = 4000
# SEED = 42

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def load_neurons():
#     neurons = []
#     files = sorted([f for f in os.listdir(SPIKE_DIR) if f.endswith("_cells.mat")])
#     print("neurons:", len(files))
#     for f in files:
#         path = os.path.join(SPIKE_DIR, f)
#         mat = loadmat(path, squeeze_me=True)
#         spikes = mat["spikes"]
#         ts = spikes[:, 2]
#         neurons.append(np.asarray(ts, dtype=np.float64))
#     return neurons

# def load_stimulus_onsets():
#     path = os.path.join(EVENT_DIR, "eventsRaw.mat")
#     mat = loadmat(path, squeeze_me=True)
#     events = mat["events"]
#     t = events[:, 0]
#     idx = np.where(events[:, 1] == 1)[0]
#     stim = t[idx]
#     print("stimulus events:", len(stim))
#     return stim

# def load_labels():
#     txts = [x for x in os.listdir(EVENT_DIR) if x.startswith("newold")]
#     labels = []
#     for t in sorted(txts):
#         path = os.path.join(EVENT_DIR, t)
#         with open(path) as f:
#             for line in f:
#                 p = line.strip().split(";")
#                 if len(p) < 2:
#                     continue
#                 code = p[1]
#                 if code == "1":
#                     labels.append(0)
#     return np.array(labels)

# def make_trial_tensor(neurons, stimulus_times):
#     edges = np.arange(WINDOW_START, WINDOW_END + BIN_MS, BIN_MS)
#     trials = []
#     for onset in stimulus_times:
#         trial = []
#         for ts in neurons:
#             rel = ts - onset
#             counts, _ = np.histogram(rel, bins=edges)
#             trial.append(counts)
#         trial = np.array(trial).T
#         trials.append(trial)
#     X = np.array(trials)
#     return X

# def build_model(adv):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=MAX_ITER,
#         output_dimension=64,
#         training_mode="adversarial" if adv else "single_session",
#         adv_alpha=1/5 if adv else 0,
#         adv_epsilon=0.05 if adv else 0,
#         adv_steps=10 if adv else 0,
#         attack_norm="linf",
#         num_hidden_units=64,
#         device=DEVICE,
#         verbose=True
#     )

# def get_attribution(model, X):
#     net = model.solver_.model.to("cuda" if torch.cuda.is_available() else "cpu")
#     net.eval()
#     inp = torch.tensor(X, dtype=torch.float32, device=next(net.parameters()).device)
#     inp.requires_grad_(True)
#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=net,
#         input_data=inp,
#         output_dimension=6
#     )
#     result = method.compute_attribution_map(batch_size=128)
#     jf = result["jf"]
#     if "jf-inv-svd" in result:
#         jfinv = result["jf-inv-svd"]
#     else:
#         jfinv = result["jf-inv"]
#     return np.mean(abs(jf), axis=0), np.mean(abs(jfinv), axis=0)

# def main():
#     seed_all(SEED)
#     neurons = load_neurons()
#     stim = load_stimulus_onsets()
#     X = make_trial_tensor(neurons, stim)
#     print("trial tensor:", X.shape)
#     ntrial, t, n = X.shape
#     Xflat = X.reshape(ntrial * t, n)
#     train_idx, test_idx = train_test_split(np.arange(ntrial), test_size=0.2, random_state=SEED)
#     train_X = X[train_idx].reshape(-1, n)
#     test_X = X[test_idx].reshape(-1, n)
#     results = []
#     for adv in [False, True]:
#         name = "ADV" if adv else "CLEAN"
#         print("\nTRAINING", name)
#         model = build_model(adv)
#         model.fit(train_X.astype(np.float32))
#         model_path = os.path.join(OUT, name + ".pth")
#         model.save(model_path)
#         jf, jfinv = get_attribution(model, test_X[:10000])
#         np.savez(os.path.join(OUT, name + "_attr.npz"), jf=jf, jfinv=jfinv)
#         pd.DataFrame(jf).to_csv(os.path.join(OUT, name + "_JF.csv"))
#         pd.DataFrame(jfinv).to_csv(os.path.join(OUT, name + "_JFINV.csv"))
#         results.append({"model": name, "neurons": n, "trials": ntrial, "time_bins": t})
#         del model
#         gc.collect()
#         torch.cuda.empty_cache()
#     pd.DataFrame(results).to_csv(os.path.join(OUT, "summary.csv"), index=False)

# if __name__ == "__main__":
#     main()
