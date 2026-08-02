import os
import sys
import gc
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from utils.min_distance import min_l2_distance
from utils.constants import CEBRA_DIR, DATA_DIR

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA


# ============================================================
# Config
# ============================================================
names = [
    # "achilles",
    # "buddy",
    "cicero",
    "gatsby",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIM = 48
BATCH_SIZE = 2048
MAX_ITER = 2500
ATTR_BATCH_SIZE = 128
RANDOM_SEED = 42

# Hippocampus data in CEBRA is spike counts binned into 25ms windows.
# We smooth the time series after binning.
BIN_WIDTH_MS = 25.0
SMOOTH_SIGMA_MS = 100.0
SMOOTH_TRUNCATE_SIGMA = 4.0

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)


# ============================================================
# Decoder
# ============================================================
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=2, dropout_rate=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Helpers
# ============================================================
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mean_r2_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(r2_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(scores)), scores


def get_embeddings(cebra_model, x_np):
    x_t = torch.from_numpy(x_np).float()
    emb = cebra_model.transform(x_t)
    return to_numpy(emb)


def cleanup_cuda(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gaussian_kernel1d(sigma_bins: float, truncate: float = 4.0):
    if sigma_bins <= 0:
        return np.array([1.0], dtype=np.float32)

    radius = int(np.ceil(truncate * sigma_bins))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2.0 * sigma_bins ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def smooth_time_series_gaussian(x: np.ndarray, sigma_ms=100.0, bin_width_ms=25.0, truncate=4.0):
    """
    Smooth each neuron's binned spike-count time series with a Gaussian kernel.
    x: shape (T, N)
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (T, N), got shape={x.shape}")

    sigma_bins = sigma_ms / bin_width_ms
    kernel = gaussian_kernel1d(sigma_bins, truncate=truncate)
    pad = len(kernel) // 2

    x_smooth = np.zeros_like(x, dtype=np.float32)

    for n in range(x.shape[1]):
        series = x[:, n].astype(np.float32)
        padded = np.pad(series, (pad, pad), mode="reflect")
        smoothed = np.convolve(padded, kernel, mode="valid")
        x_smooth[:, n] = smoothed.astype(np.float32)

    return x_smooth


def reduce_attr_to_matrix(attr_tensor, total_neurons):
    """
    Convert attribution tensor to [latent_dim, neurons] whenever possible.
    """
    attr = torch.abs(attr_tensor)

    if attr.ndim == 3:
        attr_2d = attr.mean(dim=0)
    elif attr.ndim == 2:
        attr_2d = attr
    else:
        raise ValueError(f"Unsupported attribution shape: {tuple(attr.shape)}")

    if attr_2d.shape[1] == total_neurons:
        mat = attr_2d
    elif attr_2d.shape[0] == total_neurons:
        mat = attr_2d.T
    else:
        raise ValueError(
            f"Cannot identify neuron axis. Reduced attribution shape={tuple(attr_2d.shape)}, "
            f"total_neurons={total_neurons}"
        )

    return mat.cpu().numpy()


def get_per_neuron_score(attr_tensor, total_neurons):
    mat = reduce_attr_to_matrix(attr_tensor, total_neurons)
    return mat.mean(axis=0)


def save_attr_plot(attr_tensor, total_neurons, save_path, title):
    mat = reduce_attr_to_matrix(attr_tensor, total_neurons)
    mat = np.asarray(mat, dtype=np.float32)

    s = float(mat.sum())
    if s > 0:
        mat = mat / s

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Latent dimension")
    fig.colorbar(im, ax=ax)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_cebra_model(adv, adv_epsilon, output_dim=OUTPUT_DIM):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=MAX_ITER,
        output_dimension=output_dim,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=adv_epsilon / 5 if adv else 0.0,
        adv_epsilon=adv_epsilon if adv else 0.0,
        adv_steps=10 if adv else 0,
        attack_norm="linf",
        num_hidden_units=32,
    )


def train_decoder_with_same_arch(
    cebra_model,
    train_x_np,
    train_y_np,
    test_x_np,
    test_y_np,
    input_dim,
    hidden_dim=64,
    dropout_rate=0.4,
    decoder_iters=10000,
):
    neural_train, neural_val, label_train, label_val = train_test_split(
        train_x_np,
        train_y_np,
        test_size=0.125,
        random_state=42,
        shuffle=False,
    )

    z_train = torch.from_numpy(get_embeddings(cebra_model, neural_train)).float().to(device)
    z_val = torch.from_numpy(get_embeddings(cebra_model, neural_val)).float().to(device)
    z_test = torch.from_numpy(get_embeddings(cebra_model, test_x_np)).float().to(device)

    y_train = torch.from_numpy(label_train).float().to(device)
    y_val = torch.from_numpy(label_val).float().to(device)
    y_test = torch.from_numpy(test_y_np).float().to(device)

    decoder = TwoLayerMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=y_train.shape[1],
        dropout_rate=dropout_rate,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=2e-4)

    initial_state = copy.deepcopy(decoder.state_dict())
    best_r2 = -1e18
    best_epoch = 1
    best_decoder_state = copy.deepcopy(decoder.state_dict())

    patience = 1000
    bad_epochs = 0
    min_epochs = 4000

    for epoch in range(decoder_iters):
        decoder.train()
        optimizer.zero_grad()
        outputs = decoder(z_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        decoder.eval()
        with torch.no_grad():
            val_preds = decoder(z_val).cpu().numpy()
            val_true = y_val.cpu().numpy()

        current_r2, _ = mean_r2_score(val_true, val_preds)

        if current_r2 > best_r2:
            best_r2 = current_r2
            best_epoch = epoch + 1
            bad_epochs = 0
            best_decoder_state = copy.deepcopy(decoder.state_dict())
        else:
            if epoch > min_epochs - patience:
                bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping decoder at epoch {epoch + 1}")
            break

        if (epoch + 1) % 2000 == 0:
            print(
                f"Decoder Epoch [{epoch + 1}/{decoder_iters}] | "
                f"Loss: {loss.item():.4f} | Val R2: {current_r2:.4f}"
            )

    decoder.load_state_dict(best_decoder_state)

    z_full = torch.cat([z_train, z_val], dim=0)
    y_full = torch.cat([y_train, y_val], dim=0)

    decoder.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=2e-4)

    for _ in range(best_epoch):
        decoder.train()
        optimizer.zero_grad()
        outputs = decoder(z_full)
        loss = criterion(outputs, y_full)
        loss.backward()
        optimizer.step()

    decoder.eval()
    with torch.no_grad():
        test_preds = decoder(z_test).cpu().numpy()
        test_true = y_test.cpu().numpy()

    mean_test_r2, per_dim_r2 = mean_r2_score(test_true, test_preds)

    cleanup_cuda(
        z_train, z_val, z_test,
        y_train, y_val, y_test,
        decoder, optimizer,
        z_full, y_full,
        neural_train, neural_val, label_train, label_val
    )

    return mean_test_r2, per_dim_r2


def train_full_model_and_get_attr(
    train_x_np,
    train_y_np,
    test_x_np,
    test_y_np,
    adv,
    total_neurons,
    output_dim=OUTPUT_DIM,
):
    train_tensor = torch.from_numpy(train_x_np).float()
    adv_epsilon = float(min_l2_distance(train_tensor)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)

    model = build_cebra_model(adv=adv, adv_epsilon=adv_epsilon, output_dim=output_dim)
    model.fit(train_x_np, train_y_np)

    save_name = "ACORN" if adv else "CEBRA"
    save_path = os.path.join(out_dir, f"{save_name}.pth")
    model.save(save_path)
    print("Saved model to:", save_path)

    trained_model = model.solver_.model.to(device)
    if hasattr(trained_model, "split_outputs"):
        trained_model.split_outputs = False
    trained_model.eval()

    input_tensor = torch.from_numpy(train_x_np).float().to(device).requires_grad_(True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=trained_model,
        input_data=input_tensor,
        output_dimension=int(getattr(trained_model, "num_output", output_dim)),
    )

    result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(train_x_np)))
    print("Attribution keys:", list(result.keys()))

    jf_tensor = torch.as_tensor(result["jf"]).detach().cpu()
    jfinv_tensor = torch.as_tensor(result["jf-inv-svd"]).detach().cpu()

    jf_scores = get_per_neuron_score(jf_tensor, total_neurons)
    jfinv_scores = get_per_neuron_score(jfinv_tensor, total_neurons)

    k_neurons = int(np.sqrt(total_neurons))
    topk_jf_indices = np.argsort(jf_scores)[::-1][:k_neurons]
    topk_jfinv_indices = np.argsort(jfinv_scores)[::-1][:k_neurons]

    print(f"[{save_name}] Top K = {k_neurons} out of {total_neurons}")
    print(f"[{save_name}] Top K (Jf):    {topk_jf_indices.tolist()}")
    print(f"[{save_name}] Top K (Jf-inv): {topk_jfinv_indices.tolist()}")

    base_r2, per_dim_r2 = train_decoder_with_same_arch(
        cebra_model=model,
        train_x_np=train_x_np,
        train_y_np=train_y_np,
        test_x_np=test_x_np,
        test_y_np=test_y_np,
        input_dim=output_dim,
        hidden_dim=64,
        dropout_rate=0.4,
        decoder_iters=10000,
    )

    cleanup_cuda(model, trained_model, input_tensor, method, result, jf_tensor, jfinv_tensor)

    return {
        "base_r2": base_r2,
        "per_dim_r2": per_dim_r2,
        "jf_tensor": jf_tensor,
        "jfinv_tensor": jfinv_tensor,
        "jf_scores": jf_scores,
        "jfinv_scores": jfinv_scores,
        "topk_jf": topk_jf_indices,
        "topk_jfinv": topk_jfinv_indices,
        "adv_epsilon": adv_epsilon,
        "save_name": save_name,
    }


def train_reduced_model_from_scratch(
    reduced_train_x_np,
    reduced_train_y_np,
    reduced_test_x_np,
    reduced_test_y_np,
    adv,
    output_dim=OUTPUT_DIM,
):
    train_tensor = torch.from_numpy(reduced_train_x_np).float()
    adv_epsilon = float(min_l2_distance(train_tensor)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)

    model = build_cebra_model(adv=adv, adv_epsilon=adv_epsilon, output_dim=output_dim)
    model.fit(reduced_train_x_np, reduced_train_y_np)

    base_r2, per_dim_r2 = train_decoder_with_same_arch(
        cebra_model=model,
        train_x_np=reduced_train_x_np,
        train_y_np=reduced_train_y_np,
        test_x_np=reduced_test_x_np,
        test_y_np=reduced_test_y_np,
        input_dim=output_dim,
        hidden_dim=64,
        dropout_rate=0.4,
        decoder_iters=10000,
    )

    cleanup_cuda(model)
    return base_r2, per_dim_r2


# ============================================================
# Main
# ============================================================
all_rows = []

for name in names:
    print(f"\n{'='*70}")
    print(f" Processing Dataset (Rat): {name} ".center(70, "="))
    print(f"{'='*70}")

    dataset = cebra.datasets.init(f"rat-hippocampus-single-{name}")

    neural_data = (
        dataset.neural.clone() if torch.is_tensor(dataset.neural) else torch.tensor(dataset.neural)
    ).float()

    continuous_index = (
        dataset.continuous_index.clone()
        if torch.is_tensor(dataset.continuous_index)
        else torch.tensor(dataset.continuous_index)
    ).float()

    y_np = continuous_index[:, :2].numpy().astype(np.float32)

    print(f"Raw neural shape: {neural_data.shape} | Labels shape: {y_np.shape}")

    neural_np = neural_data.detach().cpu().numpy().astype(np.float32)
    smoothed_neural_np = smooth_time_series_gaussian(
        neural_np,
        sigma_ms=SMOOTH_SIGMA_MS,
        bin_width_ms=BIN_WIDTH_MS,
        truncate=SMOOTH_TRUNCATE_SIGMA,
    )
    print("Smoothed neural shape:", smoothed_neural_np.shape)

    split_idx = int(0.8 * len(smoothed_neural_np))
    train_data_np = smoothed_neural_np[:split_idx].astype(np.float32)
    valid_data_np = smoothed_neural_np[split_idx:].astype(np.float32)

    train_continuous_label = y_np[:split_idx]
    valid_continuous_label = y_np[split_idx:]

    save_dir = os.path.join(img_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    full_results = {}

    # ========================================================
    # 1) Full models + attribution + plots
    # ========================================================
    for adv in [False, True]:
        cleanup_cuda()

        model_name = "ACORN" if adv else "CEBRA"
        print(f"\n==================== Training FULL {model_name} ====================")

        res = train_full_model_and_get_attr(
            train_x_np=train_data_np,
            train_y_np=train_continuous_label,
            test_x_np=valid_data_np,
            test_y_np=valid_continuous_label,
            adv=adv,
            total_neurons=train_data_np.shape[1],
            output_dim=OUTPUT_DIM,
        )
        full_results[model_name] = res

        # save attribution tensors
        jf_path = os.path.join(out_dir, f"{name}_{model_name}_jf.npz")
        jfinv_path = os.path.join(out_dir, f"{name}_{model_name}_jfinv.npz")

        np.savez_compressed(
            jf_path,
            jf_tensor=res["jf_tensor"].numpy(),
            jf_scores=res["jf_scores"].astype(np.float32),
            topk_jf=res["topk_jf"].astype(np.int32),
            adv_epsilon=np.array([res["adv_epsilon"]], dtype=np.float32),
        )
        np.savez_compressed(
            jfinv_path,
            jfinv_tensor=res["jfinv_tensor"].numpy(),
            jfinv_scores=res["jfinv_scores"].astype(np.float32),
            topk_jfinv=res["topk_jfinv"].astype(np.int32),
            adv_epsilon=np.array([res["adv_epsilon"]], dtype=np.float32),
        )

        print(f"Saved jf tensor to: {jf_path}")
        print(f"Saved jf-inv tensor to: {jfinv_path}")

        # save plots
        jf_plot_path = os.path.join(save_dir, f"{name}_{model_name}_jf.png")
        jfinv_plot_path = os.path.join(save_dir, f"{name}_{model_name}_jfinv.png")

        save_attr_plot(
            res["jf_tensor"],
            total_neurons=train_data_np.shape[1],
            save_path=jf_plot_path,
            title=f"{name} | {model_name} | Jf",
        )
        save_attr_plot(
            res["jfinv_tensor"],
            total_neurons=train_data_np.shape[1],
            save_path=jfinv_plot_path,
            title=f"{name} | {model_name} | Jf-inv",
        )

        print(f"Saved plot to: {jf_plot_path}")
        print(f"Saved plot to: {jfinv_plot_path}")

        all_rows.append({
            "dataset": name,
            "setting": "full",
            "source_model": model_name,
            "retrained_model": model_name,
            "neurons": train_data_np.shape[1],
            "base_r2": res["base_r2"],
            "adv_epsilon": res["adv_epsilon"],
            "jf_npy": jf_path,
            "jfinv_npy": jfinv_path,
            "jf_plot": jf_plot_path,
            "jfinv_plot": jfinv_plot_path,
        })

    # ========================================================
    # 2) Reduced experiments
    #    4 source configs x 2 retrained models = 8 rows
    # ========================================================
    reduced_results = []

    for source_model_name in ["CEBRA", "ACORN"]:
        for metric_key in ["jf", "jfinv"]:
            metric_label = "Jf" if metric_key == "jf" else "Jf-inv"
            idxs = full_results[source_model_name][f"topk_{metric_key}"]

            print(f"\n==================== Reduced dataset: {source_model_name}_TopK_{metric_label} ====================")
            print(f"Keeping {len(idxs)} neurons out of {train_data_np.shape[1]}")

            reduced_train = train_data_np[:, idxs].astype(np.float32)
            reduced_valid = valid_data_np[:, idxs].astype(np.float32)

            for retrained_name, adv in [("CEBRA", False), ("ACORN", True)]:
                print(f"\n--- Retraining {retrained_name} on {source_model_name} Top-K ({metric_label}) ---")

                base_r2, per_dim_r2 = train_reduced_model_from_scratch(
                    reduced_train_x_np=reduced_train,
                    reduced_train_y_np=train_continuous_label,
                    reduced_test_x_np=reduced_valid,
                    reduced_test_y_np=valid_continuous_label,
                    adv=adv,
                    output_dim=OUTPUT_DIM,
                )

                reduced_results.append((f"{source_model_name}_TopK_{metric_label}", retrained_name, len(idxs), base_r2))

                all_rows.append({
                    "dataset": name,
                    "setting": f"{source_model_name}_TopK_{metric_label}",
                    "source_model": source_model_name,
                    "retrained_model": retrained_name,
                    "neurons": len(idxs),
                    "base_r2": base_r2,
                    "adv_epsilon": np.nan,
                    "jf_npy": "",
                    "jfinv_npy": "",
                    "jf_plot": "",
                    "jfinv_plot": "",
                })

                print(f"** {source_model_name} Top-K ({metric_label}) -> {retrained_name} base R2: {base_r2:.4f} **")

    # ========================================================
    # 3) Summary
    # ========================================================
    print("\n" + "#" * 80)
    print(f" SUMMARY FOR {name} ".center(80, "#"))
    print("#" * 80)

    for model_name in ["CEBRA", "ACORN"]:
        print(
            f"{model_name:>6} | full base R2 = {full_results[model_name]['base_r2']:.4f} | "
            f"topJf K = {len(full_results[model_name]['topk_jf'])} | "
            f"topJfinv K = {len(full_results[model_name]['topk_jfinv'])}"
        )

    print("\nReduced dataset results:")
    for source_name, retrained_name, n_neurons, base_r2 in reduced_results:
        print(
            f"{source_name}__{retrained_name:<7} | "
            f"neurons={n_neurons:<3d} | base R2={base_r2:.4f}"
        )

    csv_path = os.path.join(out_dir, f"{name}_smoothed_full_and_reduced_results.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

print("\nPipeline Execution Completed for All Rat Datasets.")

########## RAW #############
# import os
# import sys
# import gc
# import copy
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split

# from utils.min_distance import min_l2_distance
# from utils.constants import CEBRA_DIR, DATA_DIR

# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA


# # ============================================================
# # Config
# # ============================================================
# names = [
#     "achilles",
#     "buddy",
#     "cicero",
#     "gatsby",
# ]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OUTPUT_DIM = 48
# BATCH_SIZE = 2048
# MAX_ITER = 2500
# ATTR_BATCH_SIZE = 128
# RANDOM_SEED = 42

# # Hippocampus input is already binned; we smooth the binned spike counts over time.
# BIN_WIDTH_MS = 25.0
# SMOOTH_SIGMA_MS = 100.0
# SMOOTH_TRUNCATE_SIGMA = 4.0

# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)

# out_dir = "outputs"
# img_dir = "images"
# os.makedirs(out_dir, exist_ok=True)
# os.makedirs(img_dir, exist_ok=True)

# os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)


# # ============================================================
# # Local decoder
# # ============================================================
# class TwoLayerMLP(nn.Module):
#     def __init__(self, input_dim=32, hidden_dim=64, output_dim=2, dropout_rate=0.4):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim, output_dim),
#         )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for layer in self.net:
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)

#     def forward(self, x):
#         return self.net(x)


# # ============================================================
# # Helpers
# # ============================================================
# def to_numpy(x):
#     if isinstance(x, torch.Tensor):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)


# def mean_r2_score(y_true, y_pred):
#     scores = []
#     for i in range(y_true.shape[1]):
#         scores.append(r2_score(y_true[:, i], y_pred[:, i]))
#     return float(np.mean(scores)), scores


# def get_embeddings(cebra_model, x_np):
#     x_t = torch.from_numpy(x_np).float()
#     emb = cebra_model.transform(x_t)
#     return to_numpy(emb)


# def cleanup_cuda(*objs):
#     for obj in objs:
#         try:
#             del obj
#         except Exception:
#             pass
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.ipc_collect()


# def gaussian_kernel1d(sigma_bins: float, truncate: float = 4.0):
#     if sigma_bins <= 0:
#         return np.array([1.0], dtype=np.float32)

#     radius = int(np.ceil(truncate * sigma_bins))
#     x = np.arange(-radius, radius + 1, dtype=np.float32)
#     kernel = np.exp(-(x ** 2) / (2.0 * sigma_bins ** 2))
#     kernel /= kernel.sum()
#     return kernel.astype(np.float32)


# def smooth_time_series_gaussian(x: np.ndarray, sigma_ms=100.0, bin_width_ms=25.0, truncate=4.0):
#     """
#     Smooth each neuron's binned spike-count time series with a Gaussian kernel.
#     x: shape (T, N)
#     """
#     x = np.asarray(x, dtype=np.float32)
#     if x.ndim != 2:
#         raise ValueError(f"Expected 2D array (T, N), got shape={x.shape}")

#     sigma_bins = sigma_ms / bin_width_ms
#     kernel = gaussian_kernel1d(sigma_bins, truncate=truncate)
#     pad = len(kernel) // 2

#     x_smooth = np.zeros_like(x, dtype=np.float32)

#     for n in range(x.shape[1]):
#         series = x[:, n].astype(np.float32)
#         padded = np.pad(series, (pad, pad), mode="reflect")
#         smoothed = np.convolve(padded, kernel, mode="valid")
#         x_smooth[:, n] = smoothed.astype(np.float32)

#     return x_smooth


# def reduce_attr_to_matrix(attr_tensor, total_neurons):
#     """
#     Converts attribution tensor to 2D matrix with shape [latent_dim, neurons]
#     whenever possible.
#     """
#     attr = torch.abs(attr_tensor)

#     if attr.ndim == 3:
#         attr_2d = attr.mean(dim=0)
#     elif attr.ndim == 2:
#         attr_2d = attr
#     else:
#         raise ValueError(f"Unsupported attribution shape: {tuple(attr.shape)}")

#     if attr_2d.shape[1] == total_neurons:
#         matrix = attr_2d
#     elif attr_2d.shape[0] == total_neurons:
#         matrix = attr_2d.T
#     else:
#         raise ValueError(
#             f"Cannot identify neuron axis. Reduced attribution shape={tuple(attr_2d.shape)}, "
#             f"total_neurons={total_neurons}"
#         )

#     return matrix.cpu().numpy()


# def get_per_neuron_score(attr_tensor, total_neurons):
#     matrix = reduce_attr_to_matrix(attr_tensor, total_neurons)
#     return matrix.mean(axis=0)


# def save_attr_plot(attr_tensor, total_neurons, save_path, title):
#     matrix = reduce_attr_to_matrix(attr_tensor, total_neurons)
#     matrix = np.asarray(matrix, dtype=np.float32)

#     s = float(matrix.sum())
#     if s > 0:
#         matrix = matrix / s

#     fig, ax = plt.subplots(figsize=(14, 7))
#     im = ax.imshow(matrix, aspect="auto")
#     ax.set_title(title)
#     ax.set_xlabel("Neuron")
#     ax.set_ylabel("Latent dimension")
#     fig.colorbar(im, ax=ax)
#     fig.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)


# def build_cebra_model(adv, adv_epsilon, output_dim=OUTPUT_DIM):
#     return CEBRA(
#         batch_size=BATCH_SIZE,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=MAX_ITER,
#         output_dimension=output_dim,
#         verbose=True,
#         training_mode="adversarial" if adv else "clean",
#         adv_alpha=adv_epsilon / 5 if adv else 0.0,
#         adv_epsilon=adv_epsilon if adv else 0.0,
#         adv_steps=10 if adv else 0,
#         attack_norm="linf",
#         num_hidden_units=32,
#     )


# def train_decoder_with_same_arch(
#     cebra_model,
#     train_x_np,
#     train_y_np,
#     test_x_np,
#     test_y_np,
#     input_dim,
#     hidden_dim=64,
#     dropout_rate=0.4,
#     decoder_iters=10000,
# ):
#     neural_train, neural_val, label_train, label_val = train_test_split(
#         train_x_np,
#         train_y_np,
#         test_size=0.125,
#         random_state=42,
#         shuffle=False,
#     )

#     z_train = torch.from_numpy(get_embeddings(cebra_model, neural_train)).float().to(device)
#     z_val = torch.from_numpy(get_embeddings(cebra_model, neural_val)).float().to(device)
#     z_test = torch.from_numpy(get_embeddings(cebra_model, test_x_np)).float().to(device)

#     y_train = torch.from_numpy(label_train).float().to(device)
#     y_val = torch.from_numpy(label_val).float().to(device)
#     y_test = torch.from_numpy(test_y_np).float().to(device)

#     decoder = TwoLayerMLP(
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=y_train.shape[1],
#         dropout_rate=dropout_rate,
#     ).to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=2e-4)

#     initial_state = copy.deepcopy(decoder.state_dict())
#     best_r2 = -1e18
#     best_epoch = 1
#     best_decoder_state = copy.deepcopy(decoder.state_dict())

#     patience = 1000
#     bad_epochs = 0
#     min_epochs = 4000

#     for epoch in range(decoder_iters):
#         decoder.train()
#         optimizer.zero_grad()
#         outputs = decoder(z_train)
#         loss = criterion(outputs, y_train)
#         loss.backward()
#         optimizer.step()

#         decoder.eval()
#         with torch.no_grad():
#             val_preds = decoder(z_val).cpu().numpy()
#             val_true = y_val.cpu().numpy()

#         current_r2, _ = mean_r2_score(val_true, val_preds)

#         if current_r2 > best_r2:
#             best_r2 = current_r2
#             best_epoch = epoch + 1
#             bad_epochs = 0
#             best_decoder_state = copy.deepcopy(decoder.state_dict())
#         else:
#             if epoch > min_epochs - patience:
#                 bad_epochs += 1

#         if bad_epochs >= patience:
#             print(f"Early stopping decoder at epoch {epoch + 1}")
#             break

#         if (epoch + 1) % 2000 == 0:
#             print(
#                 f"Decoder Epoch [{epoch + 1}/{decoder_iters}] | "
#                 f"Loss: {loss.item():.4f} | Val R2: {current_r2:.4f}"
#             )

#     decoder.load_state_dict(best_decoder_state)

#     z_full = torch.cat([z_train, z_val], dim=0)
#     y_full = torch.cat([y_train, y_val], dim=0)

#     decoder.load_state_dict(initial_state)
#     optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=2e-4)

#     for _ in range(best_epoch):
#         decoder.train()
#         optimizer.zero_grad()
#         outputs = decoder(z_full)
#         loss = criterion(outputs, y_full)
#         loss.backward()
#         optimizer.step()

#     decoder.eval()
#     with torch.no_grad():
#         test_preds = decoder(z_test).cpu().numpy()
#         test_true = y_test.cpu().numpy()

#     mean_test_r2, per_dim_r2 = mean_r2_score(test_true, test_preds)

#     cleanup_cuda(
#         z_train, z_val, z_test,
#         y_train, y_val, y_test,
#         decoder, optimizer,
#         z_full, y_full,
#         neural_train, neural_val, label_train, label_val
#     )

#     return mean_test_r2, per_dim_r2


# def train_full_model_and_get_attr(
#     train_x_np,
#     train_y_np,
#     test_x_np,
#     test_y_np,
#     adv,
#     total_neurons,
#     output_dim=OUTPUT_DIM,
# ):
#     train_tensor = torch.from_numpy(train_x_np).float()
#     adv_epsilon = float(min_l2_distance(train_tensor)) / 2.0
#     adv_epsilon = max(adv_epsilon, 1e-6)

#     model = build_cebra_model(adv=adv, adv_epsilon=adv_epsilon, output_dim=output_dim)
#     model.fit(train_x_np, train_y_np)

#     save_name = "ACORN" if adv else "CEBRA"
#     save_path = os.path.join(out_dir, f"{save_name}.pth")
#     model.save(save_path)
#     print("Saved model to:", save_path)

#     trained_model = model.solver_.model.to(device)
#     if hasattr(trained_model, "split_outputs"):
#         trained_model.split_outputs = False
#     trained_model.eval()

#     input_tensor = torch.from_numpy(train_x_np).float().to(device).requires_grad_(True)

#     method = cebra.attribution.init(
#         name="jacobian-based-batched",
#         model=trained_model,
#         input_data=input_tensor,
#         output_dimension=int(getattr(trained_model, "num_output", output_dim)),
#     )

#     result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(train_x_np)))
#     print("Attribution keys:", list(result.keys()))

#     jf_tensor = torch.as_tensor(result["jf"]).detach().cpu()
#     jfinv_tensor = torch.as_tensor(result["jf-inv-svd"]).detach().cpu()

#     jf_scores = get_per_neuron_score(jf_tensor, total_neurons)
#     jfinv_scores = get_per_neuron_score(jfinv_tensor, total_neurons)

#     k_neurons = int(np.sqrt(total_neurons))
#     topk_jf_indices = np.argsort(jf_scores)[::-1][:k_neurons]
#     topk_jfinv_indices = np.argsort(jfinv_scores)[::-1][:k_neurons]

#     print(f"[{save_name}] Top K = {k_neurons} out of {total_neurons}")
#     print(f"[{save_name}] Top K (Jf):    {topk_jf_indices.tolist()}")
#     print(f"[{save_name}] Top K (Jf-inv): {topk_jfinv_indices.tolist()}")

#     base_r2, per_dim_r2 = train_decoder_with_same_arch(
#         cebra_model=model,
#         train_x_np=train_x_np,
#         train_y_np=train_y_np,
#         test_x_np=test_x_np,
#         test_y_np=test_y_np,
#         input_dim=output_dim,
#         hidden_dim=64,
#         dropout_rate=0.4,
#         decoder_iters=10000,
#     )

#     cleanup_cuda(model, trained_model, input_tensor, method, result, jf_tensor, jfinv_tensor)

#     return {
#         "base_r2": base_r2,
#         "per_dim_r2": per_dim_r2,
#         "jf_tensor": jf_tensor,
#         "jfinv_tensor": jfinv_tensor,
#         "jf_scores": jf_scores,
#         "jfinv_scores": jfinv_scores,
#         "topk_jf": topk_jf_indices,
#         "topk_jfinv": topk_jfinv_indices,
#         "adv_epsilon": adv_epsilon,
#         "save_name": save_name,
#     }


# # ============================================================
# # Main loop
# # ============================================================
# rows = []

# for name in names:
#     print(f"\n{'='*70}")
#     print(f" Processing Dataset (Rat): {name} ".center(70, "="))
#     print(f"{'='*70}")

#     dataset = cebra.datasets.init(f"rat-hippocampus-single-{name}")

#     neural_data = (
#         dataset.neural.clone() if torch.is_tensor(dataset.neural) else torch.tensor(dataset.neural)
#     ).float()

#     continuous_index = (
#         dataset.continuous_index.clone()
#         if torch.is_tensor(dataset.continuous_index)
#         else torch.tensor(dataset.continuous_index)
#     ).float()

#     # position + direction
#     y_np = continuous_index[:, :2].numpy().astype(np.float32)

#     print(f"Raw neural shape: {neural_data.shape} | Labels shape: {y_np.shape}")

#     # Smooth across time
#     neural_np = neural_data.detach().cpu().numpy().astype(np.float32)
#     smoothed_neural_np = smooth_time_series_gaussian(
#         neural_np,
#         sigma_ms=SMOOTH_SIGMA_MS,
#         bin_width_ms=BIN_WIDTH_MS,
#         truncate=SMOOTH_TRUNCATE_SIGMA,
#     )

#     print("Smoothed neural shape:", smoothed_neural_np.shape)

#     split_idx = int(0.8 * len(smoothed_neural_np))
#     train_data_np = smoothed_neural_np[:split_idx].astype(np.float32)
#     valid_data_np = smoothed_neural_np[split_idx:].astype(np.float32)

#     train_continuous_label = y_np[:split_idx]
#     valid_continuous_label = y_np[split_idx:]

#     save_dir = os.path.join(img_dir, name)
#     os.makedirs(save_dir, exist_ok=True)

#     full_results = {}

#     for adv in [False, True]:
#         cleanup_cuda()

#         model_name = "ACORN" if adv else "CEBRA"
#         print(f"\n==================== Training FULL {model_name} ====================")

#         res = train_full_model_and_get_attr(
#             train_x_np=train_data_np,
#             train_y_np=train_continuous_label,
#             test_x_np=valid_data_np,
#             test_y_np=valid_continuous_label,
#             adv=adv,
#             total_neurons=train_data_np.shape[1],
#             output_dim=OUTPUT_DIM,
#         )
#         full_results[model_name] = res

#         # Save tensors/scores
#         jf_path = os.path.join(out_dir, f"{name}_{model_name}_jf.npz")
#         jfinv_path = os.path.join(out_dir, f"{name}_{model_name}_jfinv.npz")

#         np.savez_compressed(
#             jf_path,
#             jf_tensor=res["jf_tensor"].numpy(),
#             jf_scores=res["jf_scores"].astype(np.float32),
#             topk_jf=res["topk_jf"].astype(np.int32),
#             adv_epsilon=np.array([res["adv_epsilon"]], dtype=np.float32),
#         )

#         np.savez_compressed(
#             jfinv_path,
#             jfinv_tensor=res["jfinv_tensor"].numpy(),
#             jfinv_scores=res["jfinv_scores"].astype(np.float32),
#             topk_jfinv=res["topk_jfinv"].astype(np.int32),
#             adv_epsilon=np.array([res["adv_epsilon"]], dtype=np.float32),
#         )

#         print(f"Saved jf tensor to: {jf_path}")
#         print(f"Saved jf-inv tensor to: {jfinv_path}")

#         # Save plots: 2 per model = 4 per rat
#         jf_plot_path = os.path.join(save_dir, f"{name}_{model_name}_jf.png")
#         jfinv_plot_path = os.path.join(save_dir, f"{name}_{model_name}_jfinv.png")

#         save_attr_plot(
#             res["jf_tensor"],
#             total_neurons=train_data_np.shape[1],
#             save_path=jf_plot_path,
#             title=f"{name} | {model_name} | Jf",
#         )
#         save_attr_plot(
#             res["jfinv_tensor"],
#             total_neurons=train_data_np.shape[1],
#             save_path=jfinv_plot_path,
#             title=f"{name} | {model_name} | Jf-inv",
#         )

#         print(f"Saved plot to: {jf_plot_path}")
#         print(f"Saved plot to: {jfinv_plot_path}")

#         rows.append({
#             "dataset": name,
#             "model": model_name,
#             "setting": "full",
#             "neurons": train_data_np.shape[1],
#             "base_r2": res["base_r2"],
#             "adv_epsilon": res["adv_epsilon"],
#             "jf_npy": jf_path,
#             "jfinv_npy": jfinv_path,
#             "jf_plot": jf_plot_path,
#             "jfinv_plot": jfinv_plot_path,
#         })

#     print("\n" + "#" * 80)
#     print(f" SUMMARY FOR {name} ".center(80, "#"))
#     print("#" * 80)

#     for model_name in ["CEBRA", "ACORN"]:
#         print(
#             f"{model_name:>6} | full base R2 = {full_results[model_name]['base_r2']:.4f} | "
#             f"topJf K = {len(full_results[model_name]['topk_jf'])} | "
#             f"topJfinv K = {len(full_results[model_name]['topk_jfinv'])}"
#         )

#     print("\nSaved outputs:")
#     print(f"- {name}_CEBRA_jf.npz")
#     print(f"- {name}_CEBRA_jfinv.npz")
#     print(f"- {name}_ACORN_jf.npz")
#     print(f"- {name}_ACORN_jfinv.npz")
#     print(f"- plots under images/{name}/")

#     csv_path = os.path.join(out_dir, f"{name}_smoothed_full_results.csv")
#     pd.DataFrame(rows).to_csv(csv_path, index=False)
#     print(f"Saved CSV to: {csv_path}")

# print("\nPipeline Execution Completed for All Rat Datasets.")
