### Top K
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from utils.constants import CEBRA_DIR

for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
from cebra import CEBRA
print("Using CEBRA from:", cebra.__file__)

ROOT = "data/RecogMemory"
SESSION = "P10HMH_092206"
BLOCK = "NO"
RECOG_EXPERIMENT_ID = 81
EVENT_DIR = os.path.join(ROOT, "Data/events", SESSION, BLOCK)
SPIKE_DIR = os.path.join(ROOT, "Data/sorted", SESSION, BLOCK)
STIM_ORDER_FILE = os.path.join(ROOT, "Code/dataRelease/stimFiles/NewOldDelay_v3.mat")
OUT = "RecogMemory_topk_results"
os.makedirs(OUT, exist_ok=True)

SEED = 42
WINDOW_START_MS = -1000
WINDOW_END_MS = 2000
BIN_MS = 50
TEST_SIZE = 0.20
BATCH_SIZE = 1024
MAX_ITER = 4000
LATENT_DIM = 8
DEVICE = "cuda_if_available"
ADV_EPSILON = 0.05
ADV_ALPHA = 0.01
ADV_STEPS = 10
ATTR_BATCH_SIZE = 16

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_neurons():
    brain_path = os.path.join(EVENT_DIR, "brainArea.mat")
    brain_mat = loadmat(brain_path, squeeze_me=True)
    brain = np.asarray(brain_mat["brainArea"])
    if brain.ndim == 1:
        brain = brain[None, :]
    if brain.shape[1] != 4 and brain.shape[0] == 4:
        brain = brain.T
    units = []
    for row in brain:
        channel = int(row[0])
        cluster_id = float(row[1])
        units.append((channel, cluster_id))
    units = sorted(set(units), key=lambda x: (x[0], x[1]))
    neurons = []
    neuron_names = []
    for channel, cluster_id in units:
        spike_file = os.path.join(SPIKE_DIR, f"A{channel}_cells.mat")
        mat = loadmat(spike_file, squeeze_me=True)
        spikes = np.asarray(mat["spikes"])
        if spikes.ndim == 1:
            spikes = spikes[None, :]
        mask = np.isclose(spikes[:, 0], cluster_id)
        ts = spikes[mask, 2].astype(np.float64)
        if len(ts) == 0:
            raise RuntimeError(f"No spikes for A{channel}, cluster {cluster_id}")
        neurons.append(ts)
        neuron_names.append(f"A{channel}_C{cluster_id:g}")
    print("\n================ DATA ================")
    print("Number of neurons:", len(neurons))
    print("Neurons:", neuron_names)
    return neurons, neuron_names

def load_recognition_onsets():
    path = os.path.join(EVENT_DIR, "eventsRaw.mat")
    mat = loadmat(path, squeeze_me=True)
    events = np.asarray(mat["events"])
    mask = (events[:, 2] == RECOG_EXPERIMENT_ID) & (events[:, 1] == 1)
    stim = events[mask, 0].astype(np.float64)
    print("Recognition stimuli:", len(stim))
    return stim

def load_ground_truth_labels():
    mat = loadmat(STIM_ORDER_FILE, squeeze_me=True, struct_as_record=False)
    exp = np.asarray(mat["experimentStimuli"], dtype=object).reshape(-1)
    recognition = exp[1]
    labels = np.asarray(recognition.newOldRecog).astype(int).reshape(-1)
    if not np.all(np.isin(labels, [0, 1])):
        raise RuntimeError(f"Unexpected labels: {np.unique(labels)}")
    print("Ground truth:", np.unique(labels, return_counts=True))
    return labels

def make_trial_tensor(neurons, stimulus_times):
    edges_ms = np.arange(WINDOW_START_MS, WINDOW_END_MS + BIN_MS, BIN_MS, dtype=np.float64)
    trials = []
    for onset in stimulus_times:
        trial = []
        for spike_ts in neurons:
            rel_ms = (spike_ts - onset) / 1000.0
            counts, _ = np.histogram(rel_ms, bins=edges_ms)
            trial.append(counts.astype(np.float32))
        trial = np.stack(trial, axis=0).T
        trials.append(trial)
    return np.stack(trials, axis=0).astype(np.float32)

def normalize_using_train(X, idx_train):
    train_flat = X[idx_train].reshape(-1, X.shape[-1])
    mu = train_flat.mean(axis=0, keepdims=True)
    sd = train_flat.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    X_norm = (X - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)
    return X_norm.astype(np.float32)

def build_model(adv, n_neurons):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        training_mode="adversarial" if adv else "clean",
        adv_epsilon=ADV_EPSILON if adv else 0,
        adv_alpha=ADV_ALPHA if adv else 0,
        adv_steps=ADV_STEPS if adv else 0,
        attack_norm="linf",
        num_hidden_units=64,
        device=DEVICE,
        verbose=True
    )

def get_trial_embeddings(model, trials):
    embeddings = []
    for trial in trials:
        z = model.transform(trial.astype(np.float32))
        z = np.asarray(z)
        if z.ndim != 2:
            z = z.reshape(z.shape[0], -1)
        trial_embedding = z.mean(axis=0)
        embeddings.append(trial_embedding)
    return np.stack(embeddings, axis=0)

def evaluate_decoder(model, train_trials, test_trials, train_y, test_y):
    z_train = get_trial_embeddings(model, train_trials)
    z_test = get_trial_embeddings(model, test_trials)
    decoder = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED))
    ])
    decoder.fit(z_train, train_y)
    prediction = decoder.predict(z_test)
    accuracy = accuracy_score(test_y, prediction)
    return accuracy

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_train_attribution(model, train_trials, n_neurons):
    net = model.solver_.model.to("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    jf_maps = []
    jfinv_maps = []
    print("\nAttribution on ALL TRAIN trials:", len(train_trials))
    for i, trial in enumerate(train_trials):
        inp = torch.tensor(trial, dtype=torch.float32, device=next(net.parameters()).device)
        inp.requires_grad_(True)
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=net,
            input_data=inp,
            output_dimension=LATENT_DIM
        )
        result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
        jf_raw = to_numpy(result["jf"])
        if "jf-inv-svd" in result:
            jfinv_raw = to_numpy(result["jf-inv-svd"])
        elif "jf-inv" in result:
            jfinv_raw = to_numpy(result["jf-inv"])
        else:
            raise RuntimeError("Inverse Jacobian missing. Keys=" + str(list(result.keys())))
        if i == 0:
            print("RAW JF:", jf_raw.shape)
            print("RAW JFINV:", jfinv_raw.shape)
        jf_map = np.mean(np.abs(jf_raw), axis=0)
        jf_map = np.squeeze(jf_map)
        if jf_map.shape == (n_neurons, LATENT_DIM):
            jf_map = jf_map.T
        if jf_map.shape != (LATENT_DIM, n_neurons):
            raise RuntimeError(f"JF shape {jf_map.shape}, expected ({LATENT_DIM},{n_neurons})")
        jfinv_map = np.mean(np.abs(jfinv_raw), axis=0)
        jfinv_map = np.squeeze(jfinv_map)
        if jfinv_map.shape == (LATENT_DIM, n_neurons):
            jfinv_map = jfinv_map.T
        if jfinv_map.shape != (n_neurons, LATENT_DIM):
            raise RuntimeError(f"JFINV shape {jfinv_map.shape}, expected ({n_neurons},{LATENT_DIM})")
        jf_maps.append(jf_map)
        jfinv_maps.append(jfinv_map)
        del method, result, inp, jf_raw, jfinv_raw
        if torch.cuda.is_available() and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"Attribution {i+1}/{len(train_trials)}")
    jf = np.mean(np.stack(jf_maps), axis=0)
    jfinv = np.mean(np.stack(jfinv_maps), axis=0)
    print("FINAL JF:", jf.shape, "Latent x Neuron")
    print("FINAL JFINV:", jfinv.shape, "Neuron x Latent")
    return jf, jfinv

def get_jf_scores(jf):
    return np.mean(np.abs(jf), axis=0)

def get_jfinv_scores(jfinv):
    return np.mean(np.abs(jfinv), axis=1)

def select_topk(scores, k):
    return np.argsort(scores)[::-1][:k]

def plot_jf(clean, adv, neuron_names):
    vmax = max(np.max(clean), np.max(adv))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, mat, name in zip(axes, [clean, adv], ["CLEAN", "ADV"]):
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{name} — Jacobian |dz/dx|")
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Latent Dimension")
        ax.set_xticks(np.arange(len(neuron_names)))
        ax.set_xticklabels(neuron_names, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(LATENT_DIM))
    fig.colorbar(im, ax=axes, label="Mean |dz/dx|")
    path = os.path.join(OUT, "CLEAN_vs_ADV_JF.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_jfinv(clean, adv, neuron_names):
    vmax = max(np.max(clean), np.max(adv))
    fig, axes = plt.subplots(1, 2, figsize=(10, 12), constrained_layout=True)
    for ax, mat, name in zip(axes, [clean, adv], ["CLEAN", "ADV"]):
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{name} — Inverse Jacobian |dx/dz|")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Neuron")
        ax.set_xticks(np.arange(LATENT_DIM))
        ax.set_yticks(np.arange(len(neuron_names)))
        ax.set_yticklabels(neuron_names, fontsize=7)
    fig.colorbar(im, ax=axes, label="Mean |dx/dz|")
    path = os.path.join(OUT, "CLEAN_vs_ADV_JFINV.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def train_and_evaluate(train_trials, test_trials, train_y, test_y, adv):
    n_neurons = train_trials.shape[-1]
    train_flat = train_trials.reshape(-1, n_neurons)
    model = build_model(adv=adv, n_neurons=n_neurons)
    model.fit(train_flat.astype(np.float32))
    accuracy = evaluate_decoder(model, train_trials, test_trials, train_y, test_y)
    return model, accuracy

def main():
    seed_all(SEED)
    neurons, neuron_names = load_neurons()
    stim = load_recognition_onsets()
    labels = load_ground_truth_labels()
    if len(stim) != len(labels):
        raise RuntimeError(f"{len(stim)} stimuli vs {len(labels)} labels")
    X = make_trial_tensor(neurons, stim)
    n_trials, time_bins, n_neurons = X.shape
    print("\nTrial tensor:", X.shape)
    indices = np.arange(n_trials)
    idx_train, idx_test = train_test_split(indices, test_size=TEST_SIZE, random_state=SEED, stratify=labels)
    print("Train:", len(idx_train))
    print("Test:", len(idx_test))
    print("Train labels:", np.unique(labels[idx_train], return_counts=True))
    print("Test labels:", np.unique(labels[idx_test], return_counts=True))
    X = normalize_using_train(X, idx_train)
    train_trials = X[idx_train]
    test_trials = X[idx_test]
    train_y = labels[idx_train]
    test_y = labels[idx_test]

    results = []
    full_attrs = {}
    for adv in [False, True]:
        seed_all(SEED)
        model_name = "ADV" if adv else "CLEAN"
        print("\n" + "=" * 70)
        print(f"FULL MODEL: {model_name}")
        print("=" * 70)
        model, accuracy = train_and_evaluate(train_trials, test_trials, train_y, test_y, adv)
        print(f"\nFULL {model_name} ACCURACY = {accuracy:.4f}")
        jf, jfinv = get_train_attribution(model, train_trials, n_neurons)
        full_attrs[model_name] = {"jf": jf, "jfinv": jfinv}
        results.append({
            "setting": "FULL",
            "selector_model": "NONE",
            "selector_attribution": "NONE",
            "trained_model": model_name,
            "accuracy": accuracy,
            "n_neurons": n_neurons,
            "selected_indices": "ALL",
            "selected_neurons": "ALL"
        })
        del model
        cleanup()

    plot_jf(full_attrs["CLEAN"]["jf"], full_attrs["ADV"]["jf"], neuron_names)
    plot_jfinv(full_attrs["CLEAN"]["jfinv"], full_attrs["ADV"]["jfinv"], neuron_names)

    K = int(np.sqrt(n_neurons))
    K = max(1, K)
    print("\n" + "=" * 70)
    print(f"TOP-K SELECTION: N={n_neurons}, K={K}")
    print("=" * 70)

    clean_jf_scores = get_jf_scores(full_attrs["CLEAN"]["jf"])
    clean_jf_topk = select_topk(clean_jf_scores, K)
    clean_jfinv_scores = get_jfinv_scores(full_attrs["CLEAN"]["jfinv"])
    clean_jfinv_topk = select_topk(clean_jfinv_scores, K)
    adv_jf_scores = get_jf_scores(full_attrs["ADV"]["jf"])
    adv_jf_topk = select_topk(adv_jf_scores, K)
    adv_jfinv_scores = get_jfinv_scores(full_attrs["ADV"]["jfinv"])
    adv_jfinv_topk = select_topk(adv_jfinv_scores, K)

    selectors = {
        "CLEAN_JF": {"model": "CLEAN", "attr": "JF", "indices": clean_jf_topk},
        "CLEAN_JFINV": {"model": "CLEAN", "attr": "JFINV", "indices": clean_jfinv_topk},
        "ADV_JF": {"model": "ADV", "attr": "JF", "indices": adv_jf_topk},
        "ADV_JFINV": {"model": "ADV", "attr": "JFINV", "indices": adv_jfinv_topk}
    }

    for selector_name, info in selectors.items():
        idxs = info["indices"]
        names = [neuron_names[i] for i in idxs]
        print(f"\n{selector_name}")
        print("indices:", idxs.tolist())
        print("neurons:", names)

    for selector_name, info in selectors.items():
        selected_indices = info["indices"]
        selected_names = [neuron_names[i] for i in selected_indices]
        reduced_train_trials = train_trials[:, :, selected_indices]
        reduced_test_trials = test_trials[:, :, selected_indices]
        print("\n" + "#" * 70)
        print(f"REDUCED DATASET: {selector_name}")
        print(f"Selected {len(selected_indices)} of {n_neurons} neurons")
        print("Selected neurons:", selected_names)
        print("#" * 70)
        for adv in [False, True]:
            seed_all(SEED)
            trained_model_name = "ADV" if adv else "CLEAN"
            print("\n" + "-" * 60)
            print(f"{selector_name} --> {trained_model_name}")
            print("-" * 60)
            model, accuracy = train_and_evaluate(
                reduced_train_trials,
                reduced_test_trials,
                train_y,
                test_y,
                adv
            )
            print(f"\n{selector_name} --> {trained_model_name} ACCURACY = {accuracy:.4f}")
            results.append({
                "setting": "TOPK",
                "selector_model": info["model"],
                "selector_attribution": info["attr"],
                "trained_model": trained_model_name,
                "accuracy": accuracy,
                "n_neurons": len(selected_indices),
                "selected_indices": ",".join(map(str, selected_indices.tolist())),
                "selected_neurons": ",".join(selected_names)
            })
            del model
            cleanup()

    result_df = pd.DataFrame(results)
    result_df = result_df[["setting", "selector_model", "selector_attribution", "trained_model", "accuracy", "n_neurons", "selected_indices", "selected_neurons"]]
    result_path = os.path.join(OUT, "results.csv")
    result_df.to_csv(result_path, index=False)

    print("\n" + "=" * 90)
    print("FINAL RESULTS")
    print("=" * 90)
    print(result_df[["setting", "selector_model", "selector_attribution", "trained_model", "accuracy", "n_neurons"]].to_string(index=False))
    print("\nSaved:", result_path)
    print("Saved:", os.path.join(OUT, "CLEAN_vs_ADV_JF.png"))
    print("Saved:", os.path.join(OUT, "CLEAN_vs_ADV_JFINV.png"))
    print("\nDONE.")

if __name__ == "__main__":
    main()
### Normal Decoder
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
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# from utils.constants import CEBRA_DIR

# for module_name in list(sys.modules):
#     if module_name == "cebra" or module_name.startswith("cebra."):
#         del sys.modules[module_name]
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# import cebra.attribution
# from cebra import CEBRA
# print("Using CEBRA from:", cebra.__file__)

# ROOT = "data/RecogMemory"
# SESSION = "P10HMH_092206"
# BLOCK = "NO"
# RECOG_EXPERIMENT_ID = 81
# EVENT_DIR = os.path.join(ROOT, "Data/events", SESSION, BLOCK)
# SPIKE_DIR = os.path.join(ROOT, "Data/sorted", SESSION, BLOCK)
# STIM_ORDER_FILE = os.path.join(ROOT, "Code/dataRelease/stimFiles/NewOldDelay_v3.mat")
# OUT = "RecogMemory_final_results"
# os.makedirs(OUT, exist_ok=True)
# SEED = 42
# WINDOW_START_MS = -1000
# WINDOW_END_MS = 2000
# BIN_MS = 50
# TEST_SIZE = 0.20
# BATCH_SIZE = 1024
# MAX_ITER = 4000
# LATENT_DIM = 8
# DEVICE = "cuda_if_available"
# ADV_EPSILON = 0.05
# ADV_ALPHA = 0.01
# ADV_STEPS = 10

# def seed_all(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

# def load_neurons():
#     brain_path = os.path.join(EVENT_DIR, "brainArea.mat")
#     brain_mat = loadmat(brain_path, squeeze_me=True)
#     brain = np.asarray(brain_mat["brainArea"])
#     if brain.ndim == 1:
#         brain = brain[None, :]
#     if brain.shape[1] != 4 and brain.shape[0] == 4:
#         brain = brain.T
#     if brain.shape[1] < 2:
#         raise RuntimeError(f"Unexpected brainArea shape: {brain.shape}")
#     units = []
#     for row in brain:
#         channel = int(row[0])
#         cluster_id = float(row[1])
#         units.append((channel, cluster_id))
#     units = sorted(set(units), key=lambda x: (x[0], x[1]))
#     neurons = []
#     neuron_names = []
#     for channel, cluster_id in units:
#         spike_file = os.path.join(SPIKE_DIR, f"A{channel}_cells.mat")
#         if not os.path.exists(spike_file):
#             raise FileNotFoundError(f"Missing spike file: {spike_file}")
#         mat = loadmat(spike_file, squeeze_me=True)
#         spikes = np.asarray(mat["spikes"])
#         if spikes.ndim == 1:
#             spikes = spikes[None, :]
#         mask = np.isclose(spikes[:, 0], cluster_id)
#         ts = spikes[mask, 2].astype(np.float64)
#         if len(ts) == 0:
#             raise RuntimeError(f"No spikes found for A{channel}, cluster {cluster_id}")
#         neurons.append(ts)
#         neuron_names.append(f"A{channel}_C{cluster_id:g}")
#     print("\n================ DATA ================")
#     print("Number of neurons:", len(neurons))
#     print("Neurons:", neuron_names)
#     return neurons, neuron_names

# def load_recognition_onsets():
#     path = os.path.join(EVENT_DIR, "eventsRaw.mat")
#     mat = loadmat(path, squeeze_me=True)
#     events = np.asarray(mat["events"])
#     mask = (events[:, 2] == RECOG_EXPERIMENT_ID) & (events[:, 1] == 1)
#     stim = events[mask, 0].astype(np.float64)
#     print("Recognition stimulus onsets:", len(stim))
#     return stim

# def load_ground_truth_labels():
#     mat = loadmat(STIM_ORDER_FILE, squeeze_me=True, struct_as_record=False)
#     if "experimentStimuli" not in mat:
#         raise RuntimeError("experimentStimuli not found in " + STIM_ORDER_FILE)
#     exp = np.asarray(mat["experimentStimuli"], dtype=object).reshape(-1)
#     if len(exp) < 2:
#         raise RuntimeError(f"Expected at least 2 experiment blocks, got {len(exp)}")
#     recognition = exp[1]
#     if not hasattr(recognition, "newOldRecog"):
#         raise RuntimeError("newOldRecog field not found in experimentStimuli(2)")
#     labels = np.asarray(recognition.newOldRecog).astype(int).reshape(-1)
#     if not np.all(np.isin(labels, [0, 1])):
#         raise RuntimeError(f"Unexpected labels: {np.unique(labels)}")
#     values, counts = np.unique(labels, return_counts=True)
#     print("Ground-truth OLD/NEW:", dict(zip(values, counts)))
#     return labels

# def make_trial_tensor(neurons, stimulus_times):
#     edges_ms = np.arange(WINDOW_START_MS, WINDOW_END_MS + BIN_MS, BIN_MS, dtype=np.float64)
#     trials = []
#     for onset in stimulus_times:
#         trial = []
#         for spike_ts in neurons:
#             rel_ms = (spike_ts - onset) / 1000.0
#             counts, _ = np.histogram(rel_ms, bins=edges_ms)
#             trial.append(counts.astype(np.float32))
#         trial = np.stack(trial, axis=0).T
#         trials.append(trial)
#     X = np.stack(trials, axis=0).astype(np.float32)
#     return X

# def normalize_using_train(X, idx_train):
#     train_flat = X[idx_train].reshape(-1, X.shape[-1])
#     mu = train_flat.mean(axis=0, keepdims=True)
#     sd = train_flat.std(axis=0, keepdims=True)
#     sd[sd < 1e-6] = 1.0
#     X_norm = (X - mu.reshape(1, 1, -1)) / sd.reshape(1, 1, -1)
#     return X_norm.astype(np.float32)

# def build_model(adv):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=MAX_ITER,
#         output_dimension=LATENT_DIM,
#         training_mode="adversarial" if adv else "clean",
#         adv_epsilon=ADV_EPSILON if adv else 0,
#         adv_alpha=ADV_ALPHA if adv else 0,
#         adv_steps=ADV_STEPS if adv else 0,
#         attack_norm="linf",
#         num_hidden_units=64,
#         device=DEVICE,
#         verbose=True
#     )

# def get_trial_embeddings(model, trials):
#     embeddings = []
#     for i, trial in enumerate(trials):
#         z = model.transform(trial.astype(np.float32))
#         z = np.asarray(z)
#         if z.ndim != 2:
#             z = z.reshape(z.shape[0], -1)
#         trial_embedding = z.mean(axis=0)
#         embeddings.append(trial_embedding)
#     return np.stack(embeddings, axis=0)

# def to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)

# def aggregate_attr_array(arr, n_neurons, latent_dim, name):
#     a = to_numpy(arr)
#     a = np.squeeze(a)
#     candidates = []
#     for neuron_axis in range(a.ndim):
#         if a.shape[neuron_axis] != n_neurons:
#             continue
#         for latent_axis in range(a.ndim):
#             if latent_axis == neuron_axis:
#                 continue
#             if a.shape[latent_axis] == latent_dim:
#                 candidates.append((neuron_axis, latent_axis))
#     if not candidates:
#         raise RuntimeError(f"Cannot orient {name} shape {a.shape} to neuron={n_neurons}, latent={latent_dim}")
#     neuron_axis, latent_axis = max(candidates, key=lambda p: p[0] + p[1])
#     a = np.moveaxis(a, (neuron_axis, latent_axis), (-2, -1))
#     a = np.abs(a)
#     if a.ndim > 2:
#         axes = tuple(range(a.ndim - 2))
#         a = a.mean(axis=axes)
#     if a.shape != (n_neurons, latent_dim):
#         raise RuntimeError(f"{name} final shape unexpected: {a.shape}")
#     return a

# def get_train_attribution(model, train_trials, n_neurons):
#     net = model.solver_.model.to("cuda" if torch.cuda.is_available() else "cpu")
#     net.eval()
#     jf_maps = []
#     jfinv_maps = []
#     print("\nComputing Jacobian on ALL TRAIN trials:", len(train_trials))
#     for i, trial in enumerate(train_trials):
#         inp = torch.tensor(trial, dtype=torch.float32, device=next(net.parameters()).device)
#         inp.requires_grad_(True)
#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=net,
#             input_data=inp,
#             output_dimension=LATENT_DIM
#         )
#         result = method.compute_attribution_map(batch_size=16)
#         jf_raw = result["jf"]
#         if "jf-inv-svd" in result:
#             jfinv_raw = result["jf-inv-svd"]
#         elif "jf-inv" in result:
#             jfinv_raw = result["jf-inv"]
#         else:
#             raise RuntimeError(f"No inverse Jacobian. Keys={list(result.keys())}")
#         jf_raw = to_numpy(jf_raw)
#         jfinv_raw = to_numpy(jfinv_raw)
#         if i == 0:
#             print("RAW JF shape:", jf_raw.shape)
#             print("RAW JFINV shape:", jfinv_raw.shape)
#         # FORWARD JACOBIAN: J_f = dz / dx, expected [samples, latent, neuron], average over samples -> [latent, neuron]
#         jf_map = np.mean(np.abs(jf_raw), axis=0)
#         jf_map = np.squeeze(jf_map)
#         if jf_map.shape == (n_neurons, LATENT_DIM):
#             jf_map = jf_map.T
#         if jf_map.shape != (LATENT_DIM, n_neurons):
#             raise RuntimeError(
#                 f"Unexpected JF shape after averaging: {jf_map.shape}; expected ({LATENT_DIM}, {n_neurons})"
#             )
#         # INVERSE JACOBIAN: J_f^-1 ≈ dx / dz, expected [samples, neuron, latent], average over samples -> [neuron, latent]
#         jfinv_map = np.mean(np.abs(jfinv_raw), axis=0)
#         jfinv_map = np.squeeze(jfinv_map)
#         if jfinv_map.shape == (LATENT_DIM, n_neurons):
#             jfinv_map = jfinv_map.T
#         if jfinv_map.shape != (n_neurons, LATENT_DIM):
#             raise RuntimeError(
#                 f"Unexpected JFINV shape after averaging: {jfinv_map.shape}; expected ({n_neurons}, {LATENT_DIM})"
#             )
#         jf_maps.append(jf_map)
#         jfinv_maps.append(jfinv_map)
#         del method, result, inp, jf_raw, jfinv_raw
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         if (i + 1) % 10 == 0:
#             print(f"Attribution: {i+1}/{len(train_trials)} trials")
#     jf = np.mean(np.stack(jf_maps), axis=0)
#     jfinv = np.mean(np.stack(jfinv_maps), axis=0)
#     print("\nFINAL:")
#     print("JF     dz/dx shape:", jf.shape, " = Latent x Neuron")
#     print("JFINV  dx/dz shape:", jfinv.shape, " = Neuron x Latent")
#     return jf, jfinv

# # def get_train_attribution(model, train_trials, n_neurons):
# #     net = model.solver_.model.to("cuda" if torch.cuda.is_available() else "cpu")
# #     net.eval()
# #     jf_maps = []
# #     jfinv_maps = []
# #     print("\nComputing attribution on ALL TRAIN trials:", len(train_trials))
# #     for i, trial in enumerate(train_trials):
# #         inp = torch.tensor(trial, dtype=torch.float32, device=next(net.parameters()).device)
# #         inp.requires_grad_(True)
# #         method = cebra.attribution.init(
# #             name="jacobian-based-batched",
# #             model=net,
# #             input_data=inp,
# #             output_dimension=LATENT_DIM
# #         )
# #         result = method.compute_attribution_map(batch_size=16)
# #         jf_raw = result["jf"]
# #         if "jf-inv-svd" in result:
# #             jfinv_raw = result["jf-inv-svd"]
# #         elif "jf-inv" in result:
# #             jfinv_raw = result["jf-inv"]
# #         else:
# #             raise RuntimeError("No inverse Jacobian found. Keys=" + str(list(result.keys())))
# #         if i == 0:
# #             print("Raw JF shape:", to_numpy(jf_raw).shape)
# #             print("Raw JFINV shape:", to_numpy(jfinv_raw).shape)
# #         jf_map = aggregate_attr_array(jf_raw, n_neurons, LATENT_DIM, "JF")
# #         jfinv_map = aggregate_attr_array(jfinv_raw, n_neurons, LATENT_DIM, "JFINV")
# #         jf_maps.append(jf_map)
# #         jfinv_maps.append(jfinv_map)
# #         del method, result, inp, jf_raw, jfinv_raw
# #         if torch.cuda.is_available() and (i + 1) % 10 == 0:
# #             torch.cuda.empty_cache()
# #     jf = np.mean(np.stack(jf_maps), axis=0)
# #     jfinv = np.mean(np.stack(jfinv_maps), axis=0)
# #     print("Final JF shape:", jf.shape)
# #     print("Final JFINV shape:", jfinv.shape)
# #     return jf, jfinv


# def plot_jf(clean, adv, neuron_names):
#     vmax = max(np.max(clean), np.max(adv))
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
#     for ax, mat, name in zip(axes, [clean, adv], ["CLEAN", "ADV"]):
#         im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
#         ax.set_title(f"{name} — Jacobian $|dz/dx|$")
#         ax.set_xlabel("Neuron")
#         ax.set_ylabel("Latent Dimension")
#         ax.set_xticks(np.arange(len(neuron_names)))
#         ax.set_xticklabels(neuron_names, rotation=90, fontsize=6)
#         ax.set_yticks(np.arange(LATENT_DIM))
#         ax.set_yticklabels([f"z{i}" for i in range(LATENT_DIM)])
#     fig.colorbar(im, ax=axes, label="Mean |dz / dx|")
#     path = os.path.join(OUT, "CLEAN_vs_ADV_JF.png")
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print("Saved:", path)

# def plot_jfinv(clean, adv, neuron_names):
#     vmax = max(np.max(clean), np.max(adv))
#     fig, axes = plt.subplots(1, 2, figsize=(10, 12), constrained_layout=True)
#     for ax, mat, name in zip(axes, [clean, adv], ["CLEAN", "ADV"]):
#         im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
#         ax.set_title(f"{name} — Inverse Jacobian $|dx/dz|$")
#         ax.set_xlabel("Latent Dimension")
#         ax.set_ylabel("Neuron")
#         ax.set_xticks(np.arange(LATENT_DIM))
#         ax.set_xticklabels([f"z{i}" for i in range(LATENT_DIM)])
#         ax.set_yticks(np.arange(len(neuron_names)))
#         ax.set_yticklabels(neuron_names, fontsize=7)
#     fig.colorbar(im, ax=axes, label="Mean |dx / dz|")
#     path = os.path.join(OUT, "CLEAN_vs_ADV_JFINV.png")
#     fig.savefig(path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print("Saved:", path)



# def plot_comparison(clean_map, adv_map, title, filename, neuron_names):
#     vmax = max(float(np.nanmax(clean_map)), float(np.nanmax(adv_map)))
#     if vmax <= 0:
#         vmax = 1.0
#     fig, axes = plt.subplots(1, 2, figsize=(13, 7), constrained_layout=True)
#     images = []
#     for ax, mat, model_name in zip(axes, [clean_map, adv_map], ["CLEAN", "ADV"]):
#         im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
#         images.append(im)
#         ax.set_title(f"{model_name} — {title}")
#         ax.set_xlabel(f"Latent dimension ({LATENT_DIM})")
#         ax.set_ylabel("Neuron")
#         ax.set_xticks(np.arange(LATENT_DIM))
#         if len(neuron_names) <= 40:
#             ax.set_yticks(np.arange(len(neuron_names)))
#             ax.set_yticklabels(neuron_names, fontsize=7)
#     fig.colorbar(images[0], ax=axes, shrink=0.82, label="Mean |attribution| on training trials")
#     save_path = os.path.join(OUT, filename)
#     fig.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print("Saved:", save_path)

# def main():
#     seed_all(SEED)
#     neurons, neuron_names = load_neurons()
#     stim = load_recognition_onsets()
#     labels = load_ground_truth_labels()
#     if len(stim) != len(labels):
#         raise RuntimeError(f"Recognition onset/label mismatch: {len(stim)} onsets vs {len(labels)} labels")
#     X = make_trial_tensor(neurons, stim)
#     n_trials, n_time_bins, n_neurons = X.shape
#     print("Trial tensor:", X.shape)
#     print("Expected bins:", (WINDOW_END_MS - WINDOW_START_MS) // BIN_MS)
#     all_indices = np.arange(n_trials)
#     idx_train, idx_test = train_test_split(all_indices, test_size=TEST_SIZE, random_state=SEED, stratify=labels)
#     print("\nTrain trials:", len(idx_train))
#     print("Test trials:", len(idx_test))
#     print("Train labels:", np.unique(labels[idx_train], return_counts=True))
#     print("Test labels:", np.unique(labels[idx_test], return_counts=True))
#     X = normalize_using_train(X, idx_train)
#     train_trials = X[idx_train]
#     test_trials = X[idx_test]
#     train_y = labels[idx_train]
#     test_y = labels[idx_test]
#     train_X = train_trials.reshape(-1, n_neurons)
#     print("\nCEBRA train input:", train_X.shape)
#     results = []
#     attrs = {}
#     for adv in [False, True]:
#         seed_all(SEED)
#         name = "ADV" if adv else "CLEAN"
#         print("\n" + "=" * 60)
#         print("TRAINING", name)
#         print("=" * 60)
#         model = build_model(adv)
#         model.fit(train_X.astype(np.float32))
#         z_train = get_trial_embeddings(model, train_trials)
#         z_test = get_trial_embeddings(model, test_trials)
#         print(name, "trial embedding train:", z_train.shape)
#         print(name, "trial embedding test:", z_test.shape)
#         decoder = Pipeline([
#             ("scaler", StandardScaler()),
#             ("logreg", LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED))
#         ])
#         decoder.fit(z_train, train_y)
#         prediction = decoder.predict(z_test)
#         accuracy = accuracy_score(test_y, prediction)
#         print(f"\n{name} TEST ACCURACY: {accuracy:.4f}")
#         jf, jfinv = get_train_attribution(model, train_trials, n_neurons)
#         attrs[name] = {"jf": jf, "jfinv": jfinv}
#         results.append({
#             "model": name,
#             "accuracy": accuracy,
#             "n_train_trials": len(idx_train),
#             "n_test_trials": len(idx_test),
#             "n_neurons": n_neurons,
#             "time_bins_per_trial": n_time_bins,
#             "latent_dim": LATENT_DIM,
#             "cebra_max_iterations": MAX_ITER,
#             "adv_epsilon": ADV_EPSILON if adv else 0.0,
#             "adv_alpha": ADV_ALPHA if adv else 0.0,
#             "adv_steps": ADV_STEPS if adv else 0
#         })
#         del model, decoder, z_train, z_test, prediction
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     result_df = pd.DataFrame(results)
#     result_path = os.path.join(OUT, "results.csv")
#     result_df.to_csv(result_path, index=False)
#     print("\nSaved:", result_path)
#     print("\nRESULTS:")
#     print(result_df.to_string(index=False))
#     # plot_comparison(attrs["CLEAN"]["jf"], attrs["ADV"]["jf"],
#     #                 title="Forward Jacobian (JF)",
#     #                 filename="CLEAN_vs_ADV_JF.png",
#     #                 neuron_names=neuron_names)
#     # plot_comparison(attrs["CLEAN"]["jfinv"], attrs["ADV"]["jfinv"],
#     #                 title="Inverse Jacobian (JFINV)",
#     #                 filename="CLEAN_vs_ADV_JFINV.png",
#     #                 neuron_names=neuron_names)
#     plot_jf(
#         attrs["CLEAN"]["jf"],
#         attrs["ADV"]["jf"],
#         neuron_names
#     )
    
#     plot_jfinv(
#         attrs["CLEAN"]["jfinv"],
#         attrs["ADV"]["jfinv"],
#         neuron_names
#     )
#     print("\nDONE.")
#     print("\nOnly these files were created:")
#     print(os.path.join(OUT, "results.csv"))
#     print(os.path.join(OUT, "CLEAN_vs_ADV_JF.png"))
#     print(os.path.join(OUT, "CLEAN_vs_ADV_JFINV.png"))

# if __name__ == "__main__":
#     main()

# # import os
# # import sys
# # import gc
# # import random
# # import numpy as np
# # import pandas as pd
# # import torch
# # import matplotlib.pyplot as plt
# # from scipy.io import loadmat
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import accuracy_score, roc_auc_score

# # from utils.constants import CEBRA_DIR
# # sys.path.insert(0, str(CEBRA_DIR))

# # import cebra
# # from cebra import CEBRA

# # ROOT = "data/RecogMemory"
# # SESSION = "P10HMH_092206"
# # BLOCK = "NO"
# # EVENT_DIR = os.path.join(ROOT, "Data/events", SESSION, BLOCK)
# # SPIKE_DIR = os.path.join(ROOT, "Data/sorted", SESSION, BLOCK)
# # OUT = "RecogMemory_results"
# # os.makedirs(OUT, exist_ok=True)
# # DEVICE = "cuda_if_available"
# # BIN_MS = 50
# # WINDOW_START = -1000
# # WINDOW_END = 2000
# # TRAIN_RATIO = 0.8
# # BATCH_SIZE = 5000
# # MAX_ITER = 4000
# # SEED = 42

# # def seed_all(seed):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)

# # def load_neurons():
# #     neurons = []
# #     files = sorted([f for f in os.listdir(SPIKE_DIR) if f.endswith("_cells.mat")])
# #     print("neurons:", len(files))
# #     for f in files:
# #         path = os.path.join(SPIKE_DIR, f)
# #         mat = loadmat(path, squeeze_me=True)
# #         spikes = mat["spikes"]
# #         ts = spikes[:, 2]
# #         neurons.append(np.asarray(ts, dtype=np.float64))
# #     return neurons

# # def load_stimulus_onsets():
# #     path = os.path.join(EVENT_DIR, "eventsRaw.mat")
# #     mat = loadmat(path, squeeze_me=True)
# #     events = mat["events"]
# #     t = events[:, 0]
# #     idx = np.where(events[:, 1] == 1)[0]
# #     stim = t[idx]
# #     print("stimulus events:", len(stim))
# #     return stim

# # def load_labels():
# #     txts = [x for x in os.listdir(EVENT_DIR) if x.startswith("newold")]
# #     labels = []
# #     for t in sorted(txts):
# #         path = os.path.join(EVENT_DIR, t)
# #         with open(path) as f:
# #             for line in f:
# #                 p = line.strip().split(";")
# #                 if len(p) < 2:
# #                     continue
# #                 code = p[1]
# #                 if code == "1":
# #                     labels.append(0)
# #     return np.array(labels)

# # def make_trial_tensor(neurons, stimulus_times):
# #     edges = np.arange(WINDOW_START, WINDOW_END + BIN_MS, BIN_MS)
# #     trials = []
# #     for onset in stimulus_times:
# #         trial = []
# #         for ts in neurons:
# #             rel = ts - onset
# #             counts, _ = np.histogram(rel, bins=edges)
# #             trial.append(counts)
# #         trial = np.array(trial).T
# #         trials.append(trial)
# #     X = np.array(trials)
# #     return X

# # def build_model(adv):
# #     return CEBRA(
# #         batch_size=BATCH_SIZE,
# #         temperature=0.4,
# #         model_architecture="offset36-model-more-dropout",
# #         time_offsets=4,
# #         max_iterations=MAX_ITER,
# #         output_dimension=64,
# #         training_mode="adversarial" if adv else "single_session",
# #         adv_alpha=1/5 if adv else 0,
# #         adv_epsilon=0.05 if adv else 0,
# #         adv_steps=10 if adv else 0,
# #         attack_norm="linf",
# #         num_hidden_units=64,
# #         device=DEVICE,
# #         verbose=True
# #     )

# # def get_attribution(model, X):
# #     net = model.solver_.model.to("cuda" if torch.cuda.is_available() else "cpu")
# #     net.eval()
# #     inp = torch.tensor(X, dtype=torch.float32, device=next(net.parameters()).device)
# #     inp.requires_grad_(True)
# #     method = cebra.attribution.init(
# #         name="jacobian-based-batched",
# #         model=net,
# #         input_data=inp,
# #         output_dimension=6
# #     )
# #     result = method.compute_attribution_map(batch_size=128)
# #     jf = result["jf"]
# #     if "jf-inv-svd" in result:
# #         jfinv = result["jf-inv-svd"]
# #     else:
# #         jfinv = result["jf-inv"]
# #     return np.mean(abs(jf), axis=0), np.mean(abs(jfinv), axis=0)

# # def main():
# #     seed_all(SEED)
# #     neurons = load_neurons()
# #     stim = load_stimulus_onsets()
# #     X = make_trial_tensor(neurons, stim)
# #     print("trial tensor:", X.shape)
# #     ntrial, t, n = X.shape
# #     Xflat = X.reshape(ntrial * t, n)
# #     train_idx, test_idx = train_test_split(np.arange(ntrial), test_size=0.2, random_state=SEED)
# #     train_X = X[train_idx].reshape(-1, n)
# #     test_X = X[test_idx].reshape(-1, n)
# #     results = []
# #     for adv in [False, True]:
# #         name = "ADV" if adv else "CLEAN"
# #         print("\nTRAINING", name)
# #         model = build_model(adv)
# #         model.fit(train_X.astype(np.float32))
# #         model_path = os.path.join(OUT, name + ".pth")
# #         model.save(model_path)
# #         jf, jfinv = get_attribution(model, test_X[:10000])
# #         np.savez(os.path.join(OUT, name + "_attr.npz"), jf=jf, jfinv=jfinv)
# #         pd.DataFrame(jf).to_csv(os.path.join(OUT, name + "_JF.csv"))
# #         pd.DataFrame(jfinv).to_csv(os.path.join(OUT, name + "_JFINV.csv"))
# #         results.append({"model": name, "neurons": n, "trials": ntrial, "time_bins": t})
# #         del model
# #         gc.collect()
# #         torch.cuda.empty_cache()
# #     pd.DataFrame(results).to_csv(os.path.join(OUT, "summary.csv"), index=False)

# # if __name__ == "__main__":
# #     main()
