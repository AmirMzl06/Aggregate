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
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.constants import CEBRA_DIR
sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra import CEBRA

ROOT = "data/RecogMemory"
SESSION = "P10HMH_092206"
BLOCK = "NO"
EVENT_DIR = os.path.join(ROOT, "Data/events", SESSION, BLOCK)
SPIKE_DIR = os.path.join(ROOT, "Data/sorted", SESSION, BLOCK)
OUT = "RecogMemory_results"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda_if_available"
BIN_MS = 50
WINDOW_START = -1000
WINDOW_END = 2000
TRAIN_RATIO = 0.8
BATCH_SIZE = 5000
MAX_ITER = 4000
SEED = 42

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_neurons():
    neurons = []
    files = sorted([f for f in os.listdir(SPIKE_DIR) if f.endswith("_cells.mat")])
    print("neurons:", len(files))
    for f in files:
        path = os.path.join(SPIKE_DIR, f)
        mat = loadmat(path, squeeze_me=True)
        spikes = mat["spikes"]
        ts = spikes[:, 2]
        neurons.append(np.asarray(ts, dtype=np.float64))
    return neurons

def load_stimulus_onsets():
    path = os.path.join(EVENT_DIR, "eventsRaw.mat")
    mat = loadmat(path, squeeze_me=True)
    events = mat["events"]
    t = events[:, 0]
    idx = np.where(events[:, 1] == 1)[0]
    stim = t[idx]
    print("stimulus events:", len(stim))
    return stim

def load_labels():
    txts = [x for x in os.listdir(EVENT_DIR) if x.startswith("newold")]
    labels = []
    for t in sorted(txts):
        path = os.path.join(EVENT_DIR, t)
        with open(path) as f:
            for line in f:
                p = line.strip().split(";")
                if len(p) < 2:
                    continue
                code = p[1]
                if code == "1":
                    labels.append(0)
    return np.array(labels)

def make_trial_tensor(neurons, stimulus_times):
    edges = np.arange(WINDOW_START, WINDOW_END + BIN_MS, BIN_MS)
    trials = []
    for onset in stimulus_times:
        trial = []
        for ts in neurons:
            rel = ts - onset
            counts, _ = np.histogram(rel, bins=edges)
            trial.append(counts)
        trial = np.array(trial).T
        trials.append(trial)
    X = np.array(trials)
    return X

def build_model(adv):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=MAX_ITER,
        output_dimension=64,
        training_mode="adversarial" if adv else "single_session",
        adv_alpha=1/5 if adv else 0,
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
        output_dimension=6
    )
    result = method.compute_attribution_map(batch_size=128)
    jf = result["jf"]
    if "jf-inv-svd" in result:
        jfinv = result["jf-inv-svd"]
    else:
        jfinv = result["jf-inv"]
    return np.mean(abs(jf), axis=0), np.mean(abs(jfinv), axis=0)

def main():
    seed_all(SEED)
    neurons = load_neurons()
    stim = load_stimulus_onsets()
    X = make_trial_tensor(neurons, stim)
    print("trial tensor:", X.shape)
    ntrial, t, n = X.shape
    Xflat = X.reshape(ntrial * t, n)
    train_idx, test_idx = train_test_split(np.arange(ntrial), test_size=0.2, random_state=SEED)
    train_X = X[train_idx].reshape(-1, n)
    test_X = X[test_idx].reshape(-1, n)
    results = []
    for adv in [False, True]:
        name = "ADV" if adv else "CLEAN"
        print("\nTRAINING", name)
        model = build_model(adv)
        model.fit(train_X.astype(np.float32))
        model_path = os.path.join(OUT, name + ".pth")
        model.save(model_path)
        jf, jfinv = get_attribution(model, test_X[:10000])
        np.savez(os.path.join(OUT, name + "_attr.npz"), jf=jf, jfinv=jfinv)
        pd.DataFrame(jf).to_csv(os.path.join(OUT, name + "_JF.csv"))
        pd.DataFrame(jfinv).to_csv(os.path.join(OUT, name + "_JFINV.csv"))
        results.append({"model": name, "neurons": n, "trials": ntrial, "time_bins": t})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(os.path.join(OUT, "summary.csv"), index=False)

if __name__ == "__main__":
    main()
