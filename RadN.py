import os
import sys
import gc
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    "achilles",
    "gatsby",
    "buddy",
    "cicero",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIM = 48
BATCH_SIZE = 2048
MAX_ITER = 2500
ATTR_BATCH_SIZE = 64
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)


# ============================================================
# Local decoder
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


def get_per_neuron_score(attr_tensor, total_neurons):
    """
    Returns one score per neuron regardless of attribution shape:
    - (samples, latent, neurons)
    - (samples, neurons, latent)
    - (latent, neurons)
    - (neurons, latent)
    """
    attr = torch.abs(attr_tensor)

    if attr.ndim == 3:
        attr_2d = attr.mean(dim=0)
    elif attr.ndim == 2:
        attr_2d = attr
    else:
        raise ValueError(f"Unsupported attribution shape: {tuple(attr.shape)}")

    if attr_2d.shape[0] == total_neurons:
        # [neurons, latent]
        scores = attr_2d.mean(dim=1)
    elif attr_2d.shape[1] == total_neurons:
        # [latent, neurons]
        scores = attr_2d.mean(dim=0)
    else:
        raise ValueError(
            f"Cannot identify neuron axis. Reduced attribution shape={tuple(attr_2d.shape)}, "
            f"total_neurons={total_neurons}"
        )

    return scores.cpu().numpy()


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

    # retrain on full train+val set using best_epoch
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


def train_full_model_and_get_topk(
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
    adv_epsilon = 2

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
        "topk_jf": topk_jf_indices,
        "topk_jfinv": topk_jfinv_indices,
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
# Main loop
# ============================================================
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

    # behavior labels: position + direction
    y_np = continuous_index[:, :2].numpy().astype(np.float32)

    print(f"Neural shape: {neural_data.shape} | Labels shape: {y_np.shape}")

    split_idx = int(0.8 * len(neural_data))
    train_data_np = neural_data[:split_idx].detach().cpu().numpy().astype(np.float32)
    valid_data_np = neural_data[split_idx:].detach().cpu().numpy().astype(np.float32)

    train_continuous_label = y_np[:split_idx]
    valid_continuous_label = y_np[split_idx:]

    save_dir = os.path.join(img_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    full_results = {}
    reduced_results = []
    rows = []

    # --------------------------------------------------------
    # 1) Train full models and extract top-k
    # --------------------------------------------------------
    for adv in [False, True]:
        cleanup_cuda()

        model_name = "ACORN" if adv else "CEBRA"
        print(f"\n==================== Training FULL {model_name} ====================")

        res = train_full_model_and_get_topk(
            train_x_np=train_data_np,
            train_y_np=train_continuous_label,
            test_x_np=valid_data_np,
            test_y_np=valid_continuous_label,
            adv=adv,
            total_neurons=train_data_np.shape[1],
            output_dim=OUTPUT_DIM,
        )
        full_results[model_name] = res

        rows.append({
            "dataset": name,
            "setting": "full",
            "source_model": model_name,
            "retrained_model": model_name,
            "neurons": train_data_np.shape[1],
            "base_r2": res["base_r2"],
            "topk_metric": "jf/jfinv",
        })

    # --------------------------------------------------------
    # 2) Reduced datasets (8 rows total: 2 source models × 2 metrics × 2 retrained models)
    # --------------------------------------------------------
    retrained_models = [
        ("CEBRA", False),
        ("ACORN", True),
    ]

    for source_model_name in ["CEBRA", "ACORN"]:
        for metric_key in ["jf", "jfinv"]:
            metric_label = "Jf" if metric_key == "jf" else "Jf-inv"
            idxs = full_results[source_model_name][f"topk_{metric_key}"]

            print(f"\n==================== Reduced dataset: {source_model_name}_TopK_{metric_label} ====================")
            print(f"Keeping {len(idxs)} neurons out of {train_data_np.shape[1]}")

            reduced_train = train_data_np[:, idxs].astype(np.float32)
            reduced_valid = valid_data_np[:, idxs].astype(np.float32)

            for retrained_name, adv in retrained_models:
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

                rows.append({
                    "dataset": name,
                    "setting": f"{source_model_name}_TopK_{metric_label}",
                    "source_model": source_model_name,
                    "retrained_model": retrained_name,
                    "neurons": len(idxs),
                    "base_r2": base_r2,
                    "topk_metric": metric_key,
                })

                print(f"** {source_model_name} Top-K ({metric_label}) -> {retrained_name} base R2: {base_r2:.4f} **")

    # --------------------------------------------------------
    # 3) Summary
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # 4) Save CSV
    # --------------------------------------------------------
    csv_path = os.path.join(out_dir, f"{name}_reduced_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

print("\nPipeline Execution Completed for All Rat Datasets.")


# import os
# import sys
# import pandas as pd
# import copy
# import gc
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from utils.min_distance import min_l2_distance

# from utils.constants import CEBRA_DIR, DATA_DIR
# from utils.dataset_loader import DatasetLoader

# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA


# # -----------------------------
# # Config
# # -----------------------------
# datasets = [
#     # ("Chewie_CO_2016_npz", "Chewie_20160927_001.mat.npz"),
#     # ("Mihili_RT_2013_2014_npz", "Mihili_20131207_001_RT.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150730_001.mat.npz"),
#     ("Mihili_CO_2014_npz", "Mihili_20140203_001.mat.npz"),
# ]

# out_dir = "outputs"
# img_dir = "images"
# os.makedirs(out_dir, exist_ok=True)
# os.makedirs(img_dir, exist_ok=True)

# os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")

# NUM_FAKE_NEURONS = 0
# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)


# # -----------------------------
# # Local TwoLayerMLP
# # -----------------------------
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


# # -----------------------------
# # Helpers
# # -----------------------------
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


# def add_fake_neurons_end(neural_data: torch.Tensor, num_fake_neurons: int):
#     neural_data = neural_data.detach().cpu().float()
#     num_samples, num_real_neurons = neural_data.shape

#     if num_fake_neurons <= 0:
#         return neural_data, np.array([], dtype=int)

#     fake_data = torch.tensor(
#         np.random.binomial(n=1, p=0.5, size=(num_samples, num_fake_neurons)),
#         dtype=neural_data.dtype,
#     )

#     combined_neural = torch.cat([neural_data, fake_data], dim=1)
#     fake_indices = np.arange(num_real_neurons, num_real_neurons + num_fake_neurons)

#     return combined_neural, fake_indices


# def reduce_neural_data(x_np, neuron_indices):
#     neuron_indices = np.asarray(neuron_indices, dtype=int)
#     return x_np[:, neuron_indices].astype(np.float32)


# def get_per_neuron_score(attr_tensor, total_neurons):
#     """
#     Returns one score per neuron regardless of whether attr shape is:
#       - (samples, latent, neurons)
#       - (samples, neurons, latent)
#       - (latent, neurons)
#       - (neurons, latent)
#     """
#     attr = torch.abs(attr_tensor)

#     if attr.ndim == 3:
#         attr_2d = attr.mean(dim=0)
#     elif attr.ndim == 2:
#         attr_2d = attr
#     else:
#         raise ValueError(f"Unsupported attribution shape: {tuple(attr.shape)}")

#     if attr_2d.shape[0] == total_neurons:
#         scores = attr_2d.mean(dim=1)
#     elif attr_2d.shape[1] == total_neurons:
#         scores = attr_2d.mean(dim=0)
#     else:
#         raise ValueError(
#             f"Cannot identify neuron axis. Reduced attribution shape={tuple(attr_2d.shape)}, "
#             f"total_neurons={total_neurons}"
#         )

#     return scores.cpu().numpy()


# def train_cebra_and_decoder(
#     train_x_np,
#     train_y_np,
#     test_x_np,
#     test_y_np,
#     output_dim,
#     adv=False,
#     decoder_hidden_dim=64,
#     decoder_iters=10000,
# ):
#     train_x_tensor = torch.from_numpy(train_x_np).float()
#     adv_epsilon = float(min_l2_distance(train_x_tensor)) / 2.0
#     adv_epsilon = max(adv_epsilon, 1e-6)

#     model = CEBRA(
#         batch_size=2048,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=2500,
#         output_dimension=output_dim,
#         verbose=True,
#         training_mode="adversarial" if adv else "clean",
#         adv_alpha=adv_epsilon / 5 if adv else 0.0,
#         adv_epsilon=adv_epsilon if adv else 0.0,
#         adv_steps=10 if adv else 0,
#         attack_norm="linf",
#         num_hidden_units=32,
#     )

#     model.fit(train_x_np, train_y_np)

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

#     result = method.compute_attribution_map(batch_size=min(128, len(train_x_np)))
#     jf_tensor = torch.as_tensor(result["jf"]).detach().cpu()
#     jfinv_tensor = torch.as_tensor(result["jf-inv-svd"]).detach().cpu()

#     cleanup_cuda(method, trained_model, input_tensor, result)

#     decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
#         cebra_model=model,
#         train_x_np=train_x_np,
#         train_y_np=train_y_np,
#         test_x_np=test_x_np,
#         test_y_np=test_y_np,
#         input_dim=output_dim,
#         hidden_dim=decoder_hidden_dim,
#         dropout_rate=0.4,
#         decoder_iters=decoder_iters,
#     )

#     return {
#         "cebra_model": model,
#         "decoder": decoder,
#         "jf_tensor": jf_tensor,
#         "jfinv_tensor": jfinv_tensor,
#         "base_r2": mean_r2,
#         "per_dim_r2": per_dim_r2,
#     }


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

#     decoder.load_state_dict(best_decoder_state)

#     z_full = torch.cat([z_train, z_val], dim=0)
#     y_full = torch.cat([y_train, y_val], dim=0)

#     decoder.load_state_dict(copy.deepcopy(decoder.state_dict()))
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

#     return decoder, mean_test_r2, per_dim_r2


# # -----------------------------
# # Main Loop
# # -----------------------------
# for dataset_name, target_file in datasets:
#     print(f"\n{'#'*60}")
#     print(f"Processing Dataset: {dataset_name} | File: {target_file}")
#     print(f"{'#'*60}")

#     dataset_dir = os.path.join(DATA_DIR, dataset_name)
#     files = sorted(os.listdir(dataset_dir))
#     day_idx = files.index(target_file)
#     print("Selected day index:", day_idx)

#     x_np, y_np = loader.load_dataset_day(day_idx, dataset_name, cache=True)

#     print("x shape:", x_np.shape)
#     print("y shape:", y_np.shape)

#     neural_data = torch.from_numpy(x_np).float() if isinstance(x_np, np.ndarray) else x_np.clone().detach().float()
#     combined_neural, fake_indices = add_fake_neurons_end(neural_data, NUM_FAKE_NEURONS)

#     num_samples, total_neurons = combined_neural.shape
#     print(f"Added {NUM_FAKE_NEURONS} fake neurons. Total neurons now: {total_neurons}")

#     if y_np.ndim > 1 and y_np.shape[1] >= 2:
#         y_cebra = y_np[:, :2]
#     else:
#         y_cebra = y_np.reshape(-1, 1)

#     split_idx = int(0.8 * len(combined_neural))
#     train_data = combined_neural[:split_idx].contiguous()
#     valid_data = combined_neural[split_idx:].contiguous()

#     train_data_np = train_data.detach().cpu().numpy().astype(np.float32)
#     valid_data_np = valid_data.detach().cpu().numpy().astype(np.float32)

#     train_continuous_label = y_cebra[:split_idx].astype(np.float32)
#     valid_continuous_label = y_cebra[split_idx:].astype(np.float32)

#     save_dir = os.path.join(img_dir, target_file.replace(".mat.npz", "").replace(".", "_"))
#     os.makedirs(save_dir, exist_ok=True)

#     # -----------------------------
#     # Step 1: Train full models
#     # -----------------------------
#     full_results = {}
#     for adv in [False, True]:
#         cleanup_cuda()

#         model_name = "ACORN" if adv else "CEBRA"
#         print(f"\n==================== Training FULL {model_name} ====================")

#         res = train_cebra_and_decoder(
#             train_x_np=train_data_np,
#             train_y_np=train_continuous_label,
#             test_x_np=valid_data_np,
#             test_y_np=valid_continuous_label,
#             output_dim=48,
#             adv=adv,
#             decoder_hidden_dim=64,
#             decoder_iters=10000,
#         )
#         full_results[model_name] = res

#         jf_tensor = res["jf_tensor"]
#         jfinv_tensor = res["jfinv_tensor"]

#         jf_scores = get_per_neuron_score(jf_tensor, total_neurons)
#         jfinv_scores = get_per_neuron_score(jfinv_tensor, total_neurons)

#         k_neurons = int(np.sqrt(total_neurons))

#         topk_jf_indices = np.argsort(jf_scores)[::-1][:k_neurons]
#         topk_jfinv_indices = np.argsort(jfinv_scores)[::-1][:k_neurons]

#         print(f"[{model_name}] Top-K Jf indices:", topk_jf_indices.tolist())
#         print(f"[{model_name}] Top-K Jf-inv indices:", topk_jfinv_indices.tolist())

#         full_results[model_name]["topk_jf"] = topk_jf_indices
#         full_results[model_name]["topk_jfinv"] = topk_jfinv_indices

#     # -----------------------------
#     # Step 2: Build reduced datasets from top-K neurons
#     # -----------------------------
#     reduced_sets = {
#         "CEBRA_topJf": full_results["CEBRA"]["topk_jf"],
#         "ACORN_topJf": full_results["ACORN"]["topk_jf"],
#         "CEBRA_topJfinv": full_results["CEBRA"]["topk_jfinv"],
#         "ACORN_topJfinv": full_results["ACORN"]["topk_jfinv"],
#     }

#     reduced_results = {}

#     for reduced_name, idxs in reduced_sets.items():
#         print(f"\n==================== Reduced dataset: {reduced_name} ====================")
#         print(f"Keeping {len(idxs)} neurons out of {total_neurons}")

#         train_reduced = reduce_neural_data(train_data_np, idxs)
#         valid_reduced = reduce_neural_data(valid_data_np, idxs)

#         # retrain both models on this reduced dataset
#         for adv in [False, True]:
#             model_name = "ACORN" if adv else "CEBRA"
#             tag = f"{reduced_name}__{model_name}"

#             cleanup_cuda()
#             print(f"\n--- Retraining {model_name} on {reduced_name} ---")

#             res = train_cebra_and_decoder(
#                 train_x_np=train_reduced,
#                 train_y_np=train_continuous_label,
#                 test_x_np=valid_reduced,
#                 test_y_np=valid_continuous_label,
#                 output_dim=48,
#                 adv=adv,
#                 decoder_hidden_dim=64,
#                 decoder_iters=10000,
#             )

#             reduced_results[tag] = {
#                 "base_r2": res["base_r2"],
#                 "per_dim_r2": res["per_dim_r2"],
#                 "num_neurons": len(idxs),
#             }

#             print(f"** {tag} base R2: {res['base_r2']:.4f} **")

#     # -----------------------------
#     # Summary
#     # -----------------------------
#     print("\n" + "#" * 80)
#     print(f" SUMMARY FOR {dataset_name} ".center(80, "#"))
#     print("#" * 80)

#     for model_name in ["CEBRA", "ACORN"]:
#         print(
#             f"{model_name:>6} | full base R2 = {full_results[model_name]['base_r2']:.4f} | "
#             f"topJf K = {len(full_results[model_name]['topk_jf'])} | "
#             f"topJfinv K = {len(full_results[model_name]['topk_jfinv'])}"
#         )

#     print("\nReduced dataset results:")
#     for tag, info in reduced_results.items():
#         print(
#             f"{tag:<28} | neurons={info['num_neurons']:<3d} | "
#             f"base R2={info['base_r2']:.4f}"
#         )

#     # -----------------------------
#     # Save a compact CSV
#     # -----------------------------
#     rows = []
#     for model_name in ["CEBRA", "ACORN"]:
#         rows.append({
#             "dataset": dataset_name,
#             "setting": "full",
#             "model": model_name,
#             "neurons": total_neurons,
#             "base_r2": full_results[model_name]["base_r2"],
#         })

#     for tag, info in reduced_results.items():
#         setting, model_name = tag.split("__")
#         rows.append({
#             "dataset": dataset_name,
#             "setting": setting,
#             "model": model_name,
#             "neurons": info["num_neurons"],
#             "base_r2": info["base_r2"],
#         })

#     csv_path = os.path.join(out_dir, f"{target_file.replace('.mat.npz','')}_reduced_results.csv")
#     pd.DataFrame(rows).to_csv(csv_path, index=False)
#     print(f"Saved CSV to: {csv_path}")


# import sys
# import os
# import gc
# import copy
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split

# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# from utils.constants import CEBRA_DIR
# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# from cebra import CEBRA
# from utils.min_distance import min_l2_distance

# # ============================================================
# # Config
# # ============================================================
# names = [
#     "achilles",
#     "gatsby",
#     "buddy",
#     "cicero",
# ]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_FAKE_NEURONS = 0
# OUTPUT_DIM = 48
# BATCH_SIZE = 2048
# MAX_ITER = 2500
# ATTR_BATCH_SIZE = 64
# RANDOM_SEED = 42

# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)

# out_dir = "outputs"
# img_dir = "images"
# os.makedirs(out_dir, exist_ok=True)
# os.makedirs(img_dir, exist_ok=True)

# # ============================================================
# # Local TwoLayerMLP
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

# def add_fake_neurons(neural_data: torch.Tensor, num_fake_neurons: int):
#     neural_data = neural_data.detach().cpu().float()
#     num_samples, num_real_neurons = neural_data.shape

#     if num_fake_neurons <= 0:
#         return neural_data, np.array([], dtype=int)

#     fake_data = torch.tensor(
#         np.random.binomial(n=1, p=0.5, size=(num_samples, num_fake_neurons)),
#         dtype=neural_data.dtype,
#     )

#     total_neurons = num_real_neurons + num_fake_neurons
#     fake_indices = np.sort(np.random.choice(total_neurons, num_fake_neurons, replace=False))
#     real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)

#     combined_neural = torch.zeros((num_samples, total_neurons), dtype=neural_data.dtype)
#     combined_neural[:, torch.as_tensor(real_indices, dtype=torch.long)] = neural_data
#     combined_neural[:, torch.as_tensor(fake_indices, dtype=torch.long)] = fake_data

#     return combined_neural, fake_indices

# def train_decoder_with_same_arch(
#     cebra_model, train_x_np, train_y_np, test_x_np, test_y_np,
#     input_dim, hidden_dim=64, dropout_rate=0.4, decoder_iters=10000
# ):
#     neural_train, neural_val, label_train, label_val = train_test_split(
#         train_x_np, train_y_np, test_size=0.125, random_state=42, shuffle=False
#     )

#     z_train = torch.from_numpy(get_embeddings(cebra_model, neural_train)).float().to(device)
#     z_val = torch.from_numpy(get_embeddings(cebra_model, neural_val)).float().to(device)
#     z_test = torch.from_numpy(get_embeddings(cebra_model, test_x_np)).float().to(device)

#     y_train = torch.from_numpy(label_train).float().to(device)
#     y_val = torch.from_numpy(label_val).float().to(device)
#     y_test = torch.from_numpy(test_y_np).float().to(device)

#     decoder = TwoLayerMLP(
#         input_dim=input_dim, hidden_dim=hidden_dim,
#         output_dim=y_train.shape[1], dropout_rate=dropout_rate
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

#     cleanup_cuda(z_train, z_val, z_test, y_train, y_val, y_test, optimizer, z_full, y_full)
#     return decoder, mean_test_r2, per_dim_r2

# # ============================================================
# # Main loop
# # ============================================================
# for name in names:
#     print(f"\n{'='*70}")
#     print(f" Processing Dataset (Rat): {name} ".center(70, "="))
#     print(f"{'='*70}")

#     dataset = cebra.datasets.init(f"rat-hippocampus-single-{name}")
#     neural_data = (dataset.neural.clone() if torch.is_tensor(dataset.neural) else torch.tensor(dataset.neural)).float()
    
#     # Extract only the first two columns (e.g., X, Y position) for decoder prediction
#     continuous_index = (dataset.continuous_index.clone() if torch.is_tensor(dataset.continuous_index) else torch.tensor(dataset.continuous_index)).float()
#     y_np = continuous_index[:, :2].numpy()
    
#     print(f"Neural shape: {neural_data.shape} | Labels shape: {y_np.shape}")

#     combined_neural, fake_indices = add_fake_neurons(neural_data, NUM_FAKE_NEURONS)
#     num_samples, total_neurons = combined_neural.shape
    
#     if NUM_FAKE_NEURONS > 0:
#         print(f"Added {NUM_FAKE_NEURONS} fake neurons at indices: {fake_indices.tolist()}")

#     split_idx = int(0.8 * len(combined_neural))
#     train_data_np = combined_neural[:split_idx].detach().cpu().numpy().astype(np.float32)
#     valid_data_np = combined_neural[split_idx:].detach().cpu().numpy().astype(np.float32)

#     train_continuous_label = y_np[:split_idx].astype(np.float32)
#     valid_continuous_label = y_np[split_idx:].astype(np.float32)

#     save_dir = os.path.join(img_dir, name)
#     os.makedirs(save_dir, exist_ok=True)

#     models_store = {}
#     topk_store = {}
#     results = {}
    
#     # Top K Definition (Square Root of N)
#     k_neurons = int(np.sqrt(total_neurons))
#     print(f"Calculated K (sqrt of N) = {k_neurons} neurons.")

#     train_data_tensor = torch.from_numpy(train_data_np).float()
#     adv_epsilon = float(min_l2_distance(train_data_tensor)) / 2.0
#     adv_epsilon = max(adv_epsilon, 1e-6)
#     adv_apsilon = 5
#     print(f"adv_epsilon = {adv_epsilon:.6f} (min_l2_distance / 2)")

#     # 1. Train CEBRA and ACORN
#     for adv in [False, True]:
#         cleanup_cuda()
#         model_name = "ACORN" if adv else "CEBRA"
#         training_mode = "adversarial" if adv else "clean"
        
#         print(f"\n==================== Training {model_name} ====================")

#         model = CEBRA(
#             batch_size=BATCH_SIZE,
#             temperature=0.4,
#             model_architecture="offset36-model-more-dropout",
#             time_offsets=4,
#             max_iterations=MAX_ITER,
#             output_dimension=OUTPUT_DIM,
#             verbose=True,
#             training_mode=training_mode,
#             adv_alpha=adv_epsilon / 5 if adv else 0.0,
#             adv_epsilon=adv_epsilon if adv else 0.0,
#             adv_steps=10 if adv else 0,
#             attack_norm="linf",
#             num_hidden_units=32,
#         )

#         model.fit(train_data_np, train_continuous_label)

#         save_path = os.path.join(out_dir, f"{model_name}_{name}.pth")
#         model.save(save_path)
#         print(f"Saved model to: {save_path}")

#         trained_model = model.solver_.model.to(device)
#         if hasattr(trained_model, "split_outputs"):
#             trained_model.split_outputs = False
#         trained_model.eval()

#         input_tensor = torch.from_numpy(train_data_np).float().to(device).requires_grad_(True)
        
#         # 2. Extract Attributions
#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=trained_model,
#             input_data=input_tensor,
#             output_dimension=int(getattr(trained_model, "num_output", OUTPUT_DIM)),
#         )

#         result = method.compute_attribution_map(batch_size=ATTR_BATCH_SIZE)
        
#         jf_key = "jf"
#         jfinv_key = "jf-inv-svd" if "jf-inv-svd" in result else "jf-inv"
        
#         jf_tensor = torch.as_tensor(result[jf_key]).detach().cpu()
#         jfinv_tensor = torch.as_tensor(result[jfinv_key]).detach().cpu()
#         results[model_name] = {"jf": jf_tensor, "jf-inv": jfinv_tensor}

#         # Reduce maps to per-neuron scores
#         jf_mean = torch.abs(jf_tensor).mean(dim=0)
#         jf_scores = jf_mean.sum(dim=0) if jf_mean.shape[1] == total_neurons else jf_mean.sum(dim=1)

#         jfinv_mean = torch.abs(jfinv_tensor).mean(dim=0)
#         jfinv_scores = jfinv_mean.sum(dim=0) if jfinv_mean.shape[1] == total_neurons else jfinv_mean.sum(dim=1)

#         topk_jf_indices = torch.topk(jf_scores, k_neurons).indices.numpy()
#         topk_jfinv_indices = torch.topk(jfinv_scores, k_neurons).indices.numpy()
        
#         print(f"[{model_name}] Top K Neuron Indices (Jf): {topk_jf_indices.tolist()}")
#         print(f"[{model_name}] Top K Neuron Indices (Jf-inv): {topk_jfinv_indices.tolist()}")
        
#         topk_store[model_name] = {
#             "jf": topk_jf_indices,
#             "jfinv": topk_jfinv_indices
#         }

#         cleanup_cuda(method, trained_model, input_tensor, result)

#         # 3. Train Decoder
#         print(f"\n--- Training Decoder for {model_name} ---")
#         decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
#             cebra_model=model,
#             train_x_np=train_data_np,
#             train_y_np=y_np[:split_idx].astype(np.float32),
#             test_x_np=valid_data_np,
#             test_y_np=y_np[split_idx:].astype(np.float32),
#             input_dim=OUTPUT_DIM,
#             hidden_dim=64,
#             dropout_rate=0.4,
#             decoder_iters=10000,
#         )

#         print(f"** Base Test R2 Score for {model_name}: {mean_r2:.4f} **")

#         decoder_cpu = copy.deepcopy(decoder).cpu()
#         models_store[model_name] = {
#             "cebra_model": model,
#             "decoder": decoder_cpu,
#             "base_r2": mean_r2
#         }

#         decoder_save_path = os.path.join(out_dir, f"decoder_{model_name}_{name}.pth")
#         torch.save(decoder.state_dict(), decoder_save_path)
#         cleanup_cuda(decoder)

#     # ============================================================
#     # Cross-Lesion Evaluation
#     # ============================================================
#     print("\n" + "="*70)
#     print(" RUNNING CROSS-MODEL LESION EVALUATION (RAD-N TESTING) ".center(70, "="))
#     print("="*70)

#     test_true_np = y_np[split_idx:].astype(np.float32)
#     cross_results = {}

#     for eval_model_name in ["CEBRA", "ACORN"]:
#         cebra_model = models_store[eval_model_name]["cebra_model"]
#         decoder = models_store[eval_model_name]["decoder"].to(device)
#         decoder.eval()

#         cross_results[eval_model_name] = {
#             "base": models_store[eval_model_name]["base_r2"]
#         }

#         with torch.no_grad():
#             for source_model_name in ["ACORN", "CEBRA"]:
#                 # --- Evaluate using Top-K Jf ---
#                 idx_jf = topk_store[source_model_name]["jf"]
#                 valid_masked_jf = valid_data_np.copy()
#                 mask_jf = np.ones(total_neurons, dtype=bool)
#                 mask_jf[idx_jf] = False # True items will be zeroed out
#                 valid_masked_jf[:, mask_jf] = 0.0

#                 emb_jf = get_embeddings(cebra_model, valid_masked_jf)
#                 preds_jf = decoder(torch.from_numpy(emb_jf).float().to(device)).cpu().numpy()
#                 r2_jf, _ = mean_r2_score(test_true_np, preds_jf)

#                 # --- Evaluate using Top-K Jf-inv ---
#                 idx_jfinv = topk_store[source_model_name]["jfinv"]
#                 valid_masked_jfinv = valid_data_np.copy()
#                 mask_jfinv = np.ones(total_neurons, dtype=bool)
#                 mask_jfinv[idx_jfinv] = False
#                 valid_masked_jfinv[:, mask_jfinv] = 0.0

#                 emb_jfinv = get_embeddings(cebra_model, valid_masked_jfinv)
#                 preds_jfinv = decoder(torch.from_numpy(emb_jfinv).float().to(device)).cpu().numpy()
#                 r2_jfinv, _ = mean_r2_score(test_true_np, preds_jfinv)

#                 cross_results[eval_model_name][f"with_radN_{source_model_name.lower()}_jf"] = r2_jf
#                 cross_results[eval_model_name][f"with_radN_{source_model_name.lower()}_jfinv"] = r2_jfinv

#         cleanup_cuda(decoder)

#     # ============================================================
#     # Summary Report & Visualization
#     # ============================================================
#     print("\n" + "#" * 80)
#     print(f" FINAL CROSS-LESION SUMMARY FOR RAT: {name} (K = {k_neurons}) ".center(80, "#"))
#     print("#" * 80)
    
#     print("\n--- [Top K Neuron Indices] ---")
#     print(f"CEBRA Top K (Jf)    : {topk_store['CEBRA']['jf'].tolist()}")
#     print(f"CEBRA Top K (Jf-inv): {topk_store['CEBRA']['jfinv'].tolist()}")
#     print(f"ACORN Top K (Jf)    : {topk_store['ACORN']['jf'].tolist()}")
#     print(f"ACORN Top K (Jf-inv): {topk_store['ACORN']['jfinv'].tolist()}")

#     print("\n--- [Jacobian / Jf Results] ---")
#     adv_res = cross_results["ACORN"]
#     clean_res = cross_results["CEBRA"]
    
#     print(f"adv base:   {adv_res['base']:>7.4f} | adv with radN adv:   {adv_res['with_radN_acorn_jf']:>7.4f} | adv with radN clean:   {adv_res['with_radN_cebra_jf']:>7.4f}")
#     print(f"clean base: {clean_res['base']:>7.4f} | clean with radN adv: {clean_res['with_radN_acorn_jf']:>7.4f} | clean with radN clean: {clean_res['with_radN_cebra_jf']:>7.4f}")

#     print("\n--- [Inverse Jacobian / Jf-inv Results] ---")
#     print(f"adv base:   {adv_res['base']:>7.4f} | adv with radN adv:   {adv_res['with_radN_acorn_jfinv']:>7.4f} | adv with radN clean:   {adv_res['with_radN_cebra_jfinv']:>7.4f}")
#     print(f"clean base: {clean_res['base']:>7.4f} | clean with radN adv: {clean_res['with_radN_acorn_jfinv']:>7.4f} | clean with radN clean: {clean_res['with_radN_cebra_jfinv']:>7.4f}")
#     print("#" * 80 + "\n")

#     # Plot Side-by-Side Jf-inv
#     fig, axes = plt.subplots(1, 2, figsize=(15, 8))
#     model_names = ["CEBRA", "ACORN"]
#     ims = []

#     for ax, mod_name in zip(axes, model_names):
#         jfinv = torch.abs(results[mod_name]["jf-inv"]).mean(0)
#         jfinv = jfinv / jfinv.sum()
#         jfinv_np = jfinv.numpy()

#         n_rows, n_cols = jfinv_np.shape
#         im = ax.matshow(jfinv_np, aspect="auto", cmap="cividis")
#         ims.append(im)

#         self_masked_r2 = cross_results[mod_name][f"with_radN_{mod_name.lower()}_jfinv"]
#         ax.set_title(f"{mod_name}\nBase R2={cross_results[mod_name]['base']:.3f} | Self-Masked Jf-inv R2={self_masked_r2:.3f}", pad=20)
#         ax.set_xlabel(f"Neuron ({n_cols})")
#         ax.set_ylabel(f"Latent Dimension ({n_rows})")

#         if NUM_FAKE_NEURONS > 0:
#             for global_idx in fake_indices:
#                 ax.axvline(x=global_idx, color="red", linestyle="--", alpha=0.8, linewidth=1)

#     fig.subplots_adjust(right=0.85, top=0.85)
#     cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#     fig.colorbar(ims[0], cax=cbar_ax)

#     plot_path = os.path.join(save_dir, f"{name}_CEBRA_vs_ACORN.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Saved summary figure to: {plot_path}")
#     cleanup_cuda(models_store, topk_store, results, cross_results)

# print("\nPipeline Execution Completed for All Rat Datasets.")

###################### Monkeys #####################
# import os
# import sys
# import copy
# import gc
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt

# from sklearn.metrics import r2_score
# from sklearn.model_selection import train_test_split
# from utils.min_distance import min_l2_distance

# from utils.constants import CEBRA_DIR, DATA_DIR
# from utils.dataset_loader import DatasetLoader

# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA

# # -----------------------------
# # Config
# # -----------------------------
# datasets = [
#     ("Chewie_CO_2016_npz", "Chewie_20160927_001.mat.npz"),
#     ("Chewie_CO_2016_npz", "Chewie_20160928_001.mat.npz"),
#     ("Chewie_CO_2016_npz", "Chewie_20160929_001.mat.npz"),
    
#     ("Mihili_RT_2013_2014_npz", "Mihili_20131207_001_RT.mat.npz"),
#     ("Mihili_RT_2013_2014_npz", "Mihili_20131208_001_RT.mat.npz"),
#     ("Mihili_RT_2013_2014_npz", "Mihili_20140114_001_RT.mat.npz"),

#     ("Jango_ISO_2015_npz", "Jango_20150730_001.mat.npz"),
#     ("Jango_ISO_2015_npz", "Jango_20150731_001.mat.npz"),
#     ("Jango_ISO_2015_npz", "Jango_20150801_001.mat.npz"),
    
#     ("Mihili_CO_2014_npz", "Mihili_20140203_001.mat.npz"),
#     ("Mihili_CO_2014_npz", "Mihili_20140211_001.mat.npz"),
#     ("Mihili_CO_2014_npz", "Mihili_20140217_001.mat.npz"),
    
#     # ("Chewie_CO_2016_npz", "Chewie_20160930_001.mat.npz"),
#     # ("Chewie_CO_2016_npz", "Chewie_20161006_001.mat.npz"),
#     # ("Chewie_CO_2016_npz", "Chewie_20161007_001.mat.npz"),
# ]

# out_dir = "outputs"
# img_dir = "images"
# os.makedirs(out_dir, exist_ok=True)
# os.makedirs(img_dir, exist_ok=True)

# os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")

# NUM_FAKE_NEURONS = 0
# RANDOM_SEED = 29
# np.random.seed(RANDOM_SEED)
# torch.manual_seed(RANDOM_SEED)


# # -----------------------------
# # Local TwoLayerMLP
# # -----------------------------
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


# # -----------------------------
# # Helpers
# # -----------------------------
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


# def add_fake_neurons(neural_data: torch.Tensor, num_fake_neurons: int):
#     neural_data = neural_data.detach().cpu().float()
#     num_samples, num_real_neurons = neural_data.shape

#     if num_fake_neurons <= 0:
#         return neural_data, np.array([], dtype=int)

#     fake_data = torch.tensor(
#         np.random.binomial(n=1, p=0.5, size=(num_samples, num_fake_neurons)),
#         dtype=neural_data.dtype,
#     )

#     total_neurons = num_real_neurons + num_fake_neurons
#     fake_indices = np.sort(np.random.choice(total_neurons, num_fake_neurons, replace=False))
#     real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)

#     combined_neural = torch.zeros((num_samples, total_neurons), dtype=neural_data.dtype)

#     real_idx_t = torch.as_tensor(real_indices, dtype=torch.long)
#     fake_idx_t = torch.as_tensor(fake_indices, dtype=torch.long)

#     combined_neural[:, real_idx_t] = neural_data
#     combined_neural[:, fake_idx_t] = fake_data

#     return combined_neural, fake_indices


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
#         optimizer,
#         z_full, y_full,
#         neural_train, neural_val, label_train, label_val
#     )

#     return decoder, mean_test_r2, per_dim_r2


# # -----------------------------
# # Main Loop for all datasets
# # -----------------------------
# for dataset_name, target_file in datasets:
#     print(f"\n{'#'*60}")
#     print(f"Processing Dataset: {dataset_name} | File: {target_file}")
#     print(f"{'#'*60}")

#     dataset_dir = os.path.join(DATA_DIR, dataset_name)
#     files = sorted(os.listdir(dataset_dir))
#     day_idx = files.index(target_file)
#     print("Selected day index:", day_idx)

#     x_np, y_np = loader.load_dataset_day(day_idx, dataset_name, cache=True)

#     print("x shape:", x_np.shape)
#     print("y shape:", y_np.shape)

#     neural_data = torch.from_numpy(x_np).float() if isinstance(x_np, np.ndarray) else x_np.clone().detach().float()
#     combined_neural, fake_indices = add_fake_neurons(neural_data, NUM_FAKE_NEURONS)

#     num_samples, total_neurons = combined_neural.shape
#     print(f"Added {NUM_FAKE_NEURONS} fake neurons at indices: {fake_indices.tolist()}")

#     if y_np.ndim > 1 and y_np.shape[1] >= 2:
#         y_cebra = y_np[:, :2]
#     else:
#         y_cebra = y_np.reshape(-1, 1)

#     split_idx = int(0.8 * len(combined_neural))
#     train_data = combined_neural[:split_idx].contiguous()
#     valid_data = combined_neural[split_idx:].contiguous()

#     train_data_np = train_data.detach().cpu().numpy().astype(np.float32)
#     valid_data_np = valid_data.detach().cpu().numpy().astype(np.float32)

#     train_continuous_label = y_cebra[:split_idx].astype(np.float32)
#     valid_continuous_label = y_cebra[split_idx:].astype(np.float32)

#     save_dir = os.path.join(img_dir, target_file.replace(".mat.npz", "").replace(".", "_"))
#     os.makedirs(save_dir, exist_ok=True)

#     models_store = {}
#     topk_store = {}
#     results = {}
    
#     k_neurons = int(np.sqrt(total_neurons))

#     for adv in [False, True]:
#         cleanup_cuda()

#         model_name = "ACORN" if adv else "CEBRA"
#         print(f"\n==================== Training {model_name} ====================")

#         adv_epsilon = float(min_l2_distance(train_data)) / 2.0
#         adv_epsilon = max(adv_epsilon, 1e-6)

#         model = CEBRA(
#             batch_size=2048,
#             temperature=0.4,
#             model_architecture="offset36-model-more-dropout",
#             time_offsets=4,
#             max_iterations=2500,
#             output_dimension=48,
#             verbose=True,
#             training_mode="adversarial" if adv else "clean",
#             adv_alpha=adv_epsilon / 5,
#             adv_epsilon=adv_epsilon,
#             adv_steps=10,
#             attack_norm="linf",
#             num_hidden_units=32
#         )

#         model.fit(train_data_np, train_continuous_label)

#         save_path = os.path.join(out_dir, f"{model_name}_{target_file}.pth")
#         model.save(save_path)
#         print("Saved model to:", save_path)

#         trained_model = model.solver_.model.to(device)

#         input_tensor = torch.from_numpy(train_data_np).float().to(device).requires_grad_(True)
#         attr_batch_size = min(128, len(train_data_np))

#         output_dim = int(getattr(trained_model, "num_output", 48))
#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=trained_model,
#             input_data=input_tensor,
#             output_dimension=output_dim,
#         )

#         result = method.compute_attribution_map(batch_size=attr_batch_size)
#         print("Attribution keys:", list(result.keys()))

#         jf_key = "jf" 
#         jfinv_key = "jf-inv-svd" if "jf-inv-svd" in result else "jf-inv"
        
#         jf_tensor = torch.as_tensor(result[jf_key]).detach().cpu()
#         jfinv_tensor = torch.as_tensor(result[jfinv_key]).detach().cpu()
#         results[model_name] = {"jf": jf_tensor, "jf-inv": jfinv_tensor}

#         jf_mean = torch.abs(jf_tensor).mean(dim=0)
#         jf_scores = jf_mean.sum(dim=0) if jf_mean.shape[1] == total_neurons else jf_mean.sum(dim=1)

#         jfinv_mean = torch.abs(jfinv_tensor).mean(dim=0)
#         jfinv_scores = jfinv_mean.sum(dim=0) if jfinv_mean.shape[1] == total_neurons else jfinv_mean.sum(dim=1)

#         topk_jf_indices = torch.topk(jf_scores, k_neurons).indices.numpy()
#         topk_jfinv_indices = torch.topk(jfinv_scores, k_neurons).indices.numpy()
        
#         print(f"[{model_name}] Selected Top K={k_neurons} out of {total_neurons} neurons.")
#         print(f"[{model_name}] Top K Neuron Indices (Jf): {topk_jf_indices.tolist()}")
#         print(f"[{model_name}] Top K Neuron Indices (Jf-inv): {topk_jfinv_indices.tolist()}")
        
#         topk_store[model_name] = {
#             "jf": topk_jf_indices,
#             "jfinv": topk_jfinv_indices
#         }

#         cleanup_cuda(method, trained_model, input_tensor, result)

#         print(f"\n--- Training Decoder for {model_name} ---")
#         test_true_np = y_np[split_idx:].astype(np.float32)
#         decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
#             cebra_model=model,
#             train_x_np=train_data_np,
#             train_y_np=y_np[:split_idx].astype(np.float32),
#             test_x_np=valid_data_np,
#             test_y_np=test_true_np,
#             input_dim=48,
#             hidden_dim=64,
#             dropout_rate=0.4,
#             decoder_iters=10000,
#         )

#         print(f"** Base Test R2 Score for {model_name}: {mean_r2:.4f} **")

#         decoder_cpu = copy.deepcopy(decoder).cpu()
#         models_store[model_name] = {
#             "cebra_model": model,
#             "decoder": decoder_cpu,
#             "base_r2": mean_r2
#         }

#         decoder_save_path = os.path.join(out_dir, f"decoder_{model_name}_{target_file}.pth")
#         torch.save(decoder.state_dict(), decoder_save_path)
#         cleanup_cuda(decoder)

#     print("\n" + "="*70)
#     print(" RUNNING CROSS-MODEL LESION EVALUATION (RAD-N TESTING) ".center(70, "="))
#     print("="*70)

#     test_true_np = y_np[split_idx:].astype(np.float32)
#     cross_results = {}

#     for eval_model_name in ["CEBRA", "ACORN"]:
#         cebra_model = models_store[eval_model_name]["cebra_model"]
#         decoder = models_store[eval_model_name]["decoder"].to(device)
#         decoder.eval()

#         cross_results[eval_model_name] = {
#             "base": models_store[eval_model_name]["base_r2"]
#         }

#         with torch.no_grad():
#             for source_model_name in ["ACORN", "CEBRA"]:
#                 idx_jf = topk_store[source_model_name]["jf"]
#                 valid_masked_jf = valid_data_np.copy()
#                 mask_jf = np.ones(total_neurons, dtype=bool)
#                 mask_jf[idx_jf] = False
#                 valid_masked_jf[:, mask_jf] = 0.0

#                 emb_jf = get_embeddings(cebra_model, valid_masked_jf)
#                 preds_jf = decoder(torch.from_numpy(emb_jf).float().to(device)).cpu().numpy()
#                 r2_jf, _ = mean_r2_score(test_true_np, preds_jf)

#                 idx_jfinv = topk_store[source_model_name]["jfinv"]
#                 valid_masked_jfinv = valid_data_np.copy()
#                 mask_jfinv = np.ones(total_neurons, dtype=bool)
#                 mask_jfinv[idx_jfinv] = False
#                 valid_masked_jfinv[:, mask_jfinv] = 0.0

#                 emb_jfinv = get_embeddings(cebra_model, valid_masked_jfinv)
#                 preds_jfinv = decoder(torch.from_numpy(emb_jfinv).float().to(device)).cpu().numpy()
#                 r2_jfinv, _ = mean_r2_score(test_true_np, preds_jfinv)

#                 cross_results[eval_model_name][f"with_radN_{source_model_name.lower()}_jf"] = r2_jf
#                 cross_results[eval_model_name][f"with_radN_{source_model_name.lower()}_jfinv"] = r2_jfinv

#         cleanup_cuda(decoder)

#     print("\n" + "#" * 80)
#     print(f" FINAL CROSS-LESION SUMMARY FOR: {dataset_name} (K = {k_neurons}) ".center(80, "#"))
#     print("#" * 80)
    
#     print("\n--- [Top K Neuron Indices] ---")
#     print(f"CEBRA Top K (Jf)    : {topk_store['CEBRA']['jf'].tolist()}")
#     print(f"CEBRA Top K (Jf-inv): {topk_store['CEBRA']['jfinv'].tolist()}")
#     print(f"ACORN Top K (Jf)    : {topk_store['ACORN']['jf'].tolist()}")
#     print(f"ACORN Top K (Jf-inv): {topk_store['ACORN']['jfinv'].tolist()}")

#     print("\n--- [Jacobian / Jf Results] ---")
#     adv_res = cross_results["ACORN"]
#     clean_res = cross_results["CEBRA"]
    
#     print(f"adv base:   {adv_res['base']:>7.4f} | adv using adv top-K:   {adv_res['with_radN_acorn_jf']:>7.4f} | adv using clean top-K:   {adv_res['with_radN_cebra_jf']:>7.4f}")
#     print(f"clean base: {clean_res['base']:>7.4f} | clean using adv top-K: {clean_res['with_radN_acorn_jf']:>7.4f} | clean using clean top-K: {clean_res['with_radN_cebra_jf']:>7.4f}")

#     print("\n--- [Inverse Jacobian / Jf-inv Results] ---")
#     print(f"adv base:   {adv_res['base']:>7.4f} | adv using adv top-K:   {adv_res['with_radN_acorn_jfinv']:>7.4f} | adv using clean top-K:   {adv_res['with_radN_cebra_jfinv']:>7.4f}")
#     print(f"clean base: {clean_res['base']:>7.4f} | clean using adv top-K: {clean_res['with_radN_acorn_jfinv']:>7.4f} | clean using clean top-K: {clean_res['with_radN_cebra_jfinv']:>7.4f}")
#     print("#" * 80 + "\n")

#     # -----------------------------
#     # Plot and save Jacobians
#     # -----------------------------
#     fig, axes = plt.subplots(1, 2, figsize=(15, 8))
#     model_names = ["CEBRA", "ACORN"]
#     ims = []

#     for ax, name in zip(axes, model_names):
#         result = results[name]

#         jfinv = torch.abs(result["jf-inv"]).mean(0)
#         jfinv = jfinv / jfinv.sum()
#         jfinv_np = jfinv.numpy()

#         n_rows, n_cols = jfinv_np.shape

#         im = ax.matshow(jfinv_np, aspect="auto")
#         ims.append(im)

#         self_masked_r2 = cross_results[name][f"with_radN_{name.lower()}_jfinv"]
#         ax.set_title(f"{name}\nBase R2={cross_results[name]['base']:.3f} | Self-Masked Jf-inv R2={self_masked_r2:.3f}", pad=20)
#         ax.set_xlabel(f"Neuron ({n_cols})")
#         ax.set_ylabel(f"Latent Dimension ({n_rows})")

#         if NUM_FAKE_NEURONS > 0:
#             for global_idx in fake_indices:
#                 ax.axvline(x=global_idx, color="red", linestyle="--", alpha=0.8, linewidth=1)

#     fig.subplots_adjust(right=0.85, top=0.85)

#     cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#     fig.colorbar(ims[0], cax=cbar_ax)

#     plot_path = os.path.join(
#         save_dir,
#         f"{target_file.replace('.mat.npz', '').replace('.', '_')}_CEBRA_vs_ACORN.png",
#     )

#     plt.savefig(plot_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)

#     print("Saved figure to:", plot_path)
    
#     cleanup_cuda(models_store, topk_store, results, cross_results)
