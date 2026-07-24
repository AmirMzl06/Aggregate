###monkey
import os
import sys
import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils.min_distance import min_l2_distance

from utils.constants import CEBRA_DIR, DATA_DIR
from utils.dataset_loader import DatasetLoader

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA


# -----------------------------
# Config
# -----------------------------
datasets = [
    # ("Chewie_CO_2016_npz", "Chewie_20160927_001.mat.npz"),
    # ("Chewie_CO_2016_npz", "Chewie_20160928_001.mat.npz"),
    # ("Chewie_CO_2016_npz", "Chewie_20160929_001.mat.npz"),
    # ("Chewie_CO_2016_npz", "Chewie_20160930_001.mat.npz"),
    # ("Chewie_CO_2016_npz", "Chewie_20161006_001.mat.npz"),
    # ("Chewie_CO_2016_npz", "Chewie_20161007_001.mat.npz"),
    ("Mihili_RT_2013_2014_npz", "Mihili_20131207_001_RT.mat.npz"),
    # ("Mihili_RT_2013_2014_npz", "Mihili_20131208_001_RT.mat.npz"),
    # ("Mihili_RT_2013_2014_npz", "Mihili_20140114_001_RT.mat.npz"),
    # ("Mihili_RT_2013_2014_npz", "Mihili_20140115_001_RT.mat.npz"),
    # ("Mihili_RT_2013_2014_npz", "Mihili_20140116_001_RT.mat.npz"),
    # ("Mihili_RT_2013_2014_npz", "Mihili_20140128_001_RT.mat.npz"),
    ("Jango_ISO_2015_npz", "Jango_20150730_001.mat.npz"),
    # ("Jango_ISO_2015_npz", "Jango_20150731_001.mat.npz"),
    # ("Jango_ISO_2015_npz", "Jango_20150801_001.mat.npz"),
    # ("Jango_ISO_2015_npz", "Jango_20150805_001.mat.npz"),
    # ("Jango_ISO_2015_npz", "Jango_20150806_001.mat.npz"),
    # ("Jango_ISO_2015_npz", "Jango_20150807_001.mat.npz"),
    # ("Jango_ISO_2015_npz", "Jango_20150808_001.mat.npz"),
    ("Mihili_CO_2014_npz", "Mihili_20140203_001.mat.npz"),
    # ("Mihili_CO_2014_npz", "Mihili_20140211_001.mat.npz"),
    # ("Mihili_CO_2014_npz", "Mihili_20140217_001.mat.npz"),
    # ("Mihili_CO_2014_npz", "Mihili_20140218_001.mat.npz"),
    # ("Mihili_CO_2014_npz", "Mihili_20140225_001.mat.npz"),
    # ("Mihili_CO_2014_npz", "Mihili_20140227_001.mat.npz"),
]

out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")
adv_ep = 5

NUM_FAKE_NEURONS = 0
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -----------------------------
# Local TwoLayerMLP
# -----------------------------
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


# -----------------------------
# Helpers
# -----------------------------
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


def add_fake_neurons(neural_data: torch.Tensor, num_fake_neurons: int):
    neural_data = neural_data.detach().cpu().float()
    num_samples, num_real_neurons = neural_data.shape

    if num_fake_neurons <= 0:
        return neural_data, np.array([], dtype=int)

    fake_data = torch.tensor(
        np.random.binomial(
            n=1,
            p=0.5,
            size=(num_samples, num_fake_neurons)
        ),
        dtype=neural_data.dtype,
    )

    total_neurons = num_real_neurons + num_fake_neurons
    fake_indices = np.sort(
        np.random.choice(total_neurons, num_fake_neurons, replace=False)
    )
    real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)

    combined_neural = torch.zeros(
        (num_samples, total_neurons),
        dtype=neural_data.dtype
    )

    real_idx_t = torch.as_tensor(real_indices, dtype=torch.long)
    fake_idx_t = torch.as_tensor(fake_indices, dtype=torch.long)

    combined_neural[:, real_idx_t] = neural_data
    combined_neural[:, fake_idx_t] = fake_data

    return combined_neural, fake_indices


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

    return decoder, mean_test_r2, per_dim_r2


# -----------------------------
# Main Loop for all datasets
# -----------------------------
for dataset_name, target_file in datasets:
    print(f"\n{'#'*60}")
    print(f"Processing Dataset: {dataset_name} | File: {target_file}")
    print(f"{'#'*60}")

    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    files = sorted(os.listdir(dataset_dir))
    day_idx = files.index(target_file)
    print("Selected day index:", day_idx)

    x_np, y_np = loader.load_dataset_day(day_idx, dataset_name, cache=True)

    print("x shape:", x_np.shape)
    print("y shape:", y_np.shape)

    neural_data = torch.from_numpy(x_np).float() if isinstance(x_np, np.ndarray) else x_np.clone().detach().float()
    combined_neural, fake_indices = add_fake_neurons(neural_data, NUM_FAKE_NEURONS)

    num_samples, total_neurons = combined_neural.shape
    print(f"Added {NUM_FAKE_NEURONS} fake neurons at indices: {fake_indices.tolist()}")

    if y_np.ndim > 1 and y_np.shape[1] >= 2:
        y_cebra = y_np[:, :2]
    else:
        y_cebra = y_np.reshape(-1, 1)

    split_idx = int(0.8 * len(combined_neural))
    train_data = combined_neural[:split_idx].contiguous()
    valid_data = combined_neural[split_idx:].contiguous()

    train_data_np = train_data.detach().cpu().numpy().astype(np.float32)
    valid_data_np = valid_data.detach().cpu().numpy().astype(np.float32)

    train_continuous_label = y_cebra[:split_idx].astype(np.float32)
    valid_continuous_label = y_cebra[split_idx:].astype(np.float32)

    results = {}
    r2_results = {}

    save_dir = os.path.join(img_dir, target_file.replace(".mat.npz", "").replace(".", "_"))
    os.makedirs(save_dir, exist_ok=True)

    for adv in [False, True]:
        cleanup_cuda()

        model_name = "ACORN" if adv else "CEBRA"
        print(f"\n==================== Training {model_name} ====================")

        adv_epsilon = float(min_l2_distance(train_data)) / 2.0
        adv_epsilon = max(adv_epsilon, 1e-6)

        model = CEBRA(
            batch_size=2048,
            temperature=0.4,
            model_architecture="offset36-model-more-dropout",
            time_offsets=4,
            max_iterations=2500,
            output_dimension=48,
            verbose=True,
            training_mode="adversarial" if adv else "clean",
            adv_alpha=adv_epsilon / 5,
            adv_epsilon=adv_epsilon,
            adv_steps=10,
            attack_norm="linf",
            num_hidden_units=32
        )

        model.fit(train_data_np, train_continuous_label)

        save_path = os.path.join(out_dir, f"{model_name}_{target_file}.pth")
        model.save(save_path)
        print("Saved model to:", save_path)

        trained_model = model.solver_.model.to(device)

        input_tensor = torch.from_numpy(train_data_np).float().to(device).requires_grad_(True)
        attr_batch_size = min(128, len(train_data_np))

        output_dim = int(getattr(trained_model, "num_output", 48))
        method = cebra.attribution.init(
            name="jacobian-based-batched",
            model=trained_model,
            input_data=input_tensor,
            output_dimension=output_dim,
        )

        result = method.compute_attribution_map(batch_size=attr_batch_size)
        print("Attribution keys:", list(result.keys()))

        jfinv = result["jf-inv-svd"]
        jfinv_tensor = torch.as_tensor(jfinv).detach().cpu()
        results[model_name] = {"jf-inv": jfinv_tensor}

        if NUM_FAKE_NEURONS > 0:
            jfinv_mean = torch.abs(jfinv_tensor).mean(0)
            jfinv_normalized = jfinv_mean / jfinv_mean.sum()
            jfinv_normalized_cpu = jfinv_normalized.numpy()

            mean_all_neurons = jfinv_normalized_cpu.mean(axis=0)
            sorted_indices = np.argsort(mean_all_neurons)[::-1]
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(1, len(mean_all_neurons) + 1)

            fake_ranks = ranks[fake_indices]
            mean_fake_latents = mean_all_neurons[fake_indices]

            print(f"\n>>> [{model_name}] Average Inverse Attribution for Fake Neurons:")
            for idx_order, global_idx in enumerate(fake_indices):
                print(
                    f"    Fake Neuron #{idx_order+1} (Index: {global_idx}): "
                    f"{mean_fake_latents[idx_order]:.6e} | Rank: {fake_ranks[idx_order]} / {total_neurons}"
                )

        cleanup_cuda(method, trained_model, input_tensor, result, jfinv_tensor)

        print(f"\n--- Training Decoder for {model_name} ---")
        decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
            cebra_model=model,
            train_x_np=train_data_np,
            train_y_np=y_np[:split_idx].astype(np.float32),
            test_x_np=valid_data_np,
            test_y_np=y_np[split_idx:].astype(np.float32),
            input_dim=48,
            hidden_dim=64,
            dropout_rate=0.4,
            decoder_iters=10000,
        )

        r2_results[model_name] = {
            "mean_r2": mean_r2,
            "per_dim_r2": per_dim_r2,
        }

        decoder_save_path = os.path.join(out_dir, f"decoder_{model_name}_{target_file}.pth")
        torch.save(decoder.state_dict(), decoder_save_path)

        print(f"** Final mean R2 Score for {model_name}: {mean_r2:.4f} **")
        print(f"** Per-dimension R2 for {model_name}: {[round(v, 4) for v in per_dim_r2]} **\n")

        cleanup_cuda(model, decoder)

    print("\n" + "=" * 40)
    print(f" SUMMARY OF R2 SCORES FOR {dataset_name} ".center(40, "="))
    print("=" * 40)
    for name, scores in r2_results.items():
        print(f" Model: {name:<6} | Mean R2: {scores['mean_r2']:.4f}")
    print("=" * 40)

    # -----------------------------
    # Plot and save Jacobians
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    model_names = ["CEBRA", "ACORN"]
    ims = []

    for ax, name in zip(axes, model_names):
        result = results[name]

        jfinv = torch.abs(result["jf-inv"]).mean(0)
        jfinv = jfinv / jfinv.sum()
        jfinv_np = jfinv.numpy()

        n_rows, n_cols = jfinv_np.shape

        im = ax.matshow(
            jfinv_np,
            aspect="auto",
        )
        ims.append(im)

        ax.set_title(f"{name}\nR2={r2_results[name]['mean_r2']:.3f}", pad=20)
        ax.set_xlabel(f"Neuron ({n_cols})")
        ax.set_ylabel(f"Latent Dimension ({n_rows})")

        if NUM_FAKE_NEURONS > 0:
            for global_idx in fake_indices:
                ax.axvline(x=global_idx, color="red", linestyle="--", alpha=0.8, linewidth=1)

    fig.subplots_adjust(right=0.85, top=0.85)

    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(ims[0], cax=cbar_ax)

    plot_path = os.path.join(
        save_dir,
        f"{target_file.replace('.mat.npz', '').replace('.', '_')}_CEBRA_vs_ACORN.png",
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved figure to:", plot_path)
    print("Decoder scores:", r2_results)
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
#     # ("Chewie_CO_2016_npz", "Chewie_20160928_001.mat.npz"),
#     # ("Chewie_CO_2016_npz", "Chewie_20160929_001.mat.npz"),
#     # ("Chewie_CO_2016_npz", "Chewie_20160930_001.mat.npz"),
#     # ("Chewie_CO_2016_npz", "Chewie_20161006_001.mat.npz"),
#     # ("Chewie_CO_2016_npz", "Chewie_20161007_001.mat.npz"),
#     # ("Mihili_RT_2013_2014_npz", "Mihili_20131207_001_RT.mat.npz"),
#     # ("Mihili_RT_2013_2014_npz", "Mihili_20131208_001_RT.mat.npz"),
#     # ("Mihili_RT_2013_2014_npz", "Mihili_20140114_001_RT.mat.npz"),
#     # ("Mihili_RT_2013_2014_npz", "Mihili_20140115_001_RT.mat.npz"),
#     # ("Mihili_RT_2013_2014_npz", "Mihili_20140116_001_RT.mat.npz"),
#     # ("Mihili_RT_2013_2014_npz", "Mihili_20140128_001_RT.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150730_001.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150731_001.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150801_001.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150805_001.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150806_001.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150807_001.mat.npz"),
#     # ("Jango_ISO_2015_npz", "Jango_20150808_001.mat.npz"),
#     # ("Mihili_CO_2014_npz", "Mihili_20140203_001.mat.npz"),
#     # ("Mihili_CO_2014_npz", "Mihili_20140211_001.mat.npz"),
#     # ("Mihili_CO_2014_npz", "Mihili_20140217_001.mat.npz"),
#     # ("Mihili_CO_2014_npz", "Mihili_20140218_001.mat.npz"),
#     # ("Mihili_CO_2014_npz", "Mihili_20140225_001.mat.npz"),
#     # ("Mihili_CO_2014_npz", "Mihili_20140227_001.mat.npz"),
# ]

# out_dir = "outputs"
# img_dir = "images"
# os.makedirs(out_dir, exist_ok=True)
# os.makedirs(img_dir, exist_ok=True)

# os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")
# adv_ep = 5

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


# def add_fake_neurons(neural_data: torch.Tensor, num_fake_neurons: int):
#     neural_data = neural_data.detach().cpu().float()
#     num_samples, num_real_neurons = neural_data.shape

#     if num_fake_neurons <= 0:
#         return neural_data, np.array([], dtype=int)

#     fake_data = torch.tensor(
#         np.random.binomial(
#             n=1,
#             p=0.5,
#             size=(num_samples, num_fake_neurons)
#         ),
#         dtype=neural_data.dtype,
#     )

#     total_neurons = num_real_neurons + num_fake_neurons
#     fake_indices = np.sort(
#         np.random.choice(total_neurons, num_fake_neurons, replace=False)
#     )
#     real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)

#     combined_neural = torch.zeros(
#         (num_samples, total_neurons),
#         dtype=neural_data.dtype
#     )

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

#     return decoder, mean_test_r2, per_dim_r2


# # -----------------------------
# # Main Loop for all 4 Monkeys
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

#     results = {}
#     r2_results = {}

#     save_dir = os.path.join(img_dir, target_file.replace(".mat.npz", "").replace(".", "_"))
#     os.makedirs(save_dir, exist_ok=True)

#     for adv in [False, True]:
#         cleanup_cuda()

#         model_name = "ACORN" if adv else "CEBRA"
#         print(f"\n==================== Training {model_name} ====================")

#         adv_epsilon = float(min_l2_distance(train_data)) / 2.0

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

#         # Attribution
#         trained_model = model.solver_.model.to(device)

#         N_ATTR = min(512, len(train_data_np))
#         attr_idx = np.random.choice(len(train_data_np), N_ATTR, replace=False)
#         attr_data = train_data_np[attr_idx]

#         input_tensor = torch.from_numpy(attr_data).float().to(device).requires_grad_(True)

#         output_dim = int(getattr(trained_model, "num_output", 48))
#         method = cebra.attribution.init(
#             # name="jacobian-based",
#             name = "jacobian-based-batched",
#             model=trained_model,
#             input_data=input_tensor,
#             output_dimension=output_dim,
#         )
        
#         result = method.compute_attribution_map()

#         # jf_cpu = result["jf"].detach().cpu() if torch.is_tensor(result["jf"]) else torch.tensor(result["jf"])
#         # results[model_name] = {"jf": jf_cpu}
        
#         jfinv = result["jf-inv-svd"]
#         jfinv_tensor = torch.tensor(jfinv) if not torch.is_tensor(jfinv) else jfinv.detach().cpu()
#         results[model_name] = {"jf-inv": jfinv_tensor}

#         if NUM_FAKE_NEURONS > 0:
#             jfinv_mean = torch.abs(jfinv_tensor).mean(0)
#             jfinv_normalized = jfinv_mean / jfinv_mean.sum()
#             jfinv_normalized_cpu = jfinv_normalized.numpy()

#             mean_all_neurons = jfinv_normalized_cpu.mean(axis=1) 
            
#             sorted_indices = np.argsort(mean_all_neurons)[::-1]
#             ranks = np.empty_like(sorted_indices)
#             ranks[sorted_indices] = np.arange(1, len(mean_all_neurons) + 1)
            
#             fake_ranks = ranks[fake_indices]
#             mean_fake_latents = mean_all_neurons[fake_indices]

#             print(f"\n>>> [{model_name}] Average Inverse Attribution for Fake Neurons:")
#             for idx_order, global_idx in enumerate(fake_indices):
#                 print(f"    Fake Neuron #{idx_order+1} (Index: {global_idx}): {mean_fake_latents[idx_order]:.6e} | Rank: {fake_ranks[idx_order]} / {total_neurons}")

#         cleanup_cuda(method, trained_model, input_tensor, jfinv_tensor, attr_data)
        
#         # if NUM_FAKE_NEURONS > 0:
#         #     jf_normalized = torch.abs(jf_cpu).mean(0)
#         #     jf_normalized = jf_normalized / jf_normalized.sum()
#         #     jf_normalized_cpu = jf_normalized.detach().cpu().numpy()

#         #     mean_all_neurons = jf_normalized_cpu.mean(axis=0)
#         #     sorted_indices = np.argsort(mean_all_neurons)[::-1]
#         #     ranks = np.empty_like(sorted_indices)
#         #     ranks[sorted_indices] = np.arange(1, len(mean_all_neurons) + 1)
            
#         #     fake_ranks = ranks[fake_indices]
#         #     mean_fake_latents = mean_all_neurons[fake_indices]

#         #     print(f"\n>>> [{model_name}] Average Latent Attribution for Fake Neurons:")
#         #     for idx_order, global_idx in enumerate(fake_indices):
#         #         print(f"    Fake Neuron #{idx_order+1} (Index: {global_idx}): {mean_fake_latents[idx_order]:.6e} | Rank: {fake_ranks[idx_order]} / {total_neurons}")

#         # cleanup_cuda(method, trained_model, input_tensor, result, jf_cpu, attr_data)

#         # Decoder R2
#         print(f"\n--- Training Decoder for {model_name} ---")
#         decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
#             cebra_model=model,
#             train_x_np=train_data_np,
#             train_y_np=y_np[:split_idx].astype(np.float32),
#             test_x_np=valid_data_np,
#             test_y_np=y_np[split_idx:].astype(np.float32),
#             input_dim=48,
#             hidden_dim=64,
#             dropout_rate=0.4,
#             decoder_iters=10000,
#         )

#         r2_results[model_name] = {
#             "mean_r2": mean_r2,
#             "per_dim_r2": per_dim_r2,
#         }

#         decoder_save_path = os.path.join(out_dir, f"decoder_{model_name}_{target_file}.pth")
#         torch.save(decoder.state_dict(), decoder_save_path)

#         print(f"** Final mean R2 Score for {model_name}: {mean_r2:.4f} **")
#         print(f"** Per-dimension R2 for {model_name}: {[round(v, 4) for v in per_dim_r2]} **\n")

#         cleanup_cuda(model, decoder)

#     print("\n" + "=" * 40)
#     print(f" SUMMARY OF R2 SCORES FOR {dataset_name} ".center(40, "="))
#     print("=" * 40)
#     for name, scores in r2_results.items():
#         print(f" Model: {name:<6} | Mean R2: {scores['mean_r2']:.4f}")
#     print("=" * 40)

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

#         im = ax.matshow(
#             jfinv_np,
#             aspect="auto",
#         )
#         ims.append(im)

#         ax.set_title(f"{name}\nR2={r2_results[name]['mean_r2']:.3f}", pad=20)

#         ax.set_xlabel(f"Latent Dimension ({n_cols})")
#         ax.set_ylabel(f"Neuron ({n_rows})")

#         if NUM_FAKE_NEURONS > 0:
#             for global_idx in fake_indices:
#                 ax.axhline(y=global_idx, color='red', linestyle='--', alpha=0.8, linewidth=1)

    
#     # fig, axes = plt.subplots(1, 2, figsize=(15, 8))
#     # model_names = ["CEBRA", "ACORN"]
#     # ims = []

#     # for ax, name in zip(axes, model_names):
#     #     result = results[name]
#     #     jf = torch.abs(result["jf"]).mean(0)
#     #     jf = jf / jf.sum()
#     #     jf = jf.detach().cpu().numpy()

#     #     n_rows, n_cols = jf.shape

#     #     im = ax.matshow(
#     #         jf,
#     #         aspect="auto",
#     #     )
#     #     ims.append(im)

#     #     ax.set_title(f"{name}\nR2={r2_results[name]['mean_r2']:.3f}", pad=20)

#     #     ax.set_xlabel(f"Latent Dimension ({n_cols})")
#     #     ax.set_ylabel(f"Neuron ({n_rows})")

#     #     if NUM_FAKE_NEURONS > 0:
#     #         for global_idx in fake_indices:
#     #             ax.axvline(x=global_idx, color='red', linestyle='--', alpha=0.8, linewidth=1)

#     fig.subplots_adjust(right=0.85, top=0.85)

#     cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
#     fig.colorbar(ims[0], cax=cbar_ax)

#     plot_path = os.path.join(
#         save_dir,
#         f"{target_file.replace('.mat.npz', '').replace('.', '_')}_CEBRA_vs_ACORN.png",
#     )

#     plt.savefig(
#         plot_path,
#         dpi=300,
#         bbox_inches="tight",
#     )
#     plt.close(fig)

#     print("Saved figure to:", plot_path)
#     print("Decoder scores:", r2_results)

###Hippocampus
# import sys
# import os
# import gc
# import itertools
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# from utils.constants import CEBRA_DIR
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra 
# from cebra import CEBRA
# from utils.dataset_loader import DatasetLoader
# from torch.utils.data import DataLoader, TensorDataset
# from utils.normalization import normalize_to_target
# from utils.min_distance import min_l2_distance

# names = [
#     'achilles',
#     'gatsby',
#     'buddy',
#     'cicero'
# ]

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# NUM_FAKE_NEURONS = 0

# for name in names:
#     print(f"\n========== Processing Rat: {name} ==========")
    
#     dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
    
#     neural_data = dataset.neural.clone() if torch.is_tensor(dataset.neural) else torch.tensor(dataset.neural)
#     num_samples, num_real_neurons = neural_data.shape
    
#     fake_data = torch.tensor(np.random.binomial(n=1, p=0.5, size=(num_samples, NUM_FAKE_NEURONS)), dtype=neural_data.dtype)
    
#     total_neurons = num_real_neurons + NUM_FAKE_NEURONS
#     fake_indices = np.sort(np.random.choice(total_neurons, NUM_FAKE_NEURONS, replace=False))
#     real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)
    
#     combined_neural = torch.zeros((num_samples, total_neurons), dtype=neural_data.dtype)
#     combined_neural[:, real_indices] = neural_data
#     combined_neural[:, fake_indices] = fake_data
    
#     print(f"Added {NUM_FAKE_NEURONS} fake neurons at indices: {fake_indices.tolist()}")
    
#     split_idx = int(0.8 * len(combined_neural))
#     train_data = combined_neural[:split_idx]
#     valid_data = combined_neural[split_idx:]
    
#     continuous_index = dataset.continuous_index.clone() if torch.is_tensor(dataset.continuous_index) else torch.tensor(dataset.continuous_index)
#     train_continuous_label = continuous_index[:split_idx, :2].numpy()
#     valid_continuous_label = continuous_index[split_idx:, :2].numpy()

#     save_dir = os.path.join("images", name)
#     os.makedirs(save_dir, exist_ok=True)

#     # --- 2. Train & Evaluate for both Clean and Adversarial ---
#     for adv in [False, True]:
#         model_name = "ACORN" if adv else "CEBRA"
#         adv_epsilon = 5
#         epochs = 2500

#         print(f"\n--- Training {model_name} (adv = {adv}) ---")
        
#         model = CEBRA(
#             batch_size=1024,
#             temperature=0.4,
#             model_architecture="offset36-model-more-dropout",
#             time_offsets=4,
#             max_iterations=epochs,
#             output_dimension=48,
#             verbose=True,
#             training_mode='adversarial' if adv else 'clean',
#             adv_alpha=adv_epsilon / 5 if adv else 0.0,
#             adv_epsilon=adv_epsilon if adv else 0.0,
#             adv_steps=10 if adv else 0, 
#             attack_norm="l2"
#         )
        
#         path = f"{name}_adv.pth" if adv else f"{name}.pth"
        
#         # Fit & Save
#         model.fit(train_data, train_continuous_label)
#         model.save(path)
        
#         # Load & Attribution
#         model = CEBRA.load(path, weights_only=False)
#         model = model.solver_.model.to(device)
        
#         input_data = train_data.clone().detach().float().to(device)
#         input_data.requires_grad_(True)
        
#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=model,
#             input_data=input_data,
#             output_dimension=model.num_output,
#             num_samples=min(2000, len(input_data))
#         )

#         print(f"Computing Jacobian Map for {model_name}...")
#         result = method.compute_attribution_map(batch_size=32)
        
#         # --- 3. Extracting Mean Latents and Plotting ---
#         jf = abs(result['jf']).mean(0)
#         jf_normalized = jf / jf.sum()
        

#         if hasattr(jf_normalized, 'detach'):
#             jf_normalized_cpu = jf_normalized.detach().cpu().numpy()
#         else:
#             jf_normalized_cpu = np.array(jf_normalized)
        
#         fake_jf = jf_normalized_cpu[:, fake_indices] 
#         mean_fake_latents = fake_jf.mean(axis=0)
        
#         print(f"\n>>> [{model_name}] Average Latent Attribution for Fake Neurons:")
#         for idx_order, global_idx in enumerate(fake_indices):
#             print(f"    Fake Neuron #{idx_order+1} (Index: {global_idx}): {mean_fake_latents[idx_order]:.6e}")

#         fig, ax = plt.subplots(figsize=(15, 8))
#         im = ax.matshow(
#             jf_normalized_cpu,
#             aspect="auto",
#             cmap="cividis",
#         )
#         ax.set_title(f"Rat: {name} | Model: {model_name}", fontsize=14)
        
#         for global_idx in fake_indices:
#             ax.axvline(x=global_idx, color='red', linestyle='--', alpha=0.8, linewidth=1)
            
#         fig.colorbar(im, ax=ax)
        
#         plot_path = os.path.join(save_dir, f"{model_name}_jacobian.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches="tight")
#         print(f"Plot saved successfully at: {plot_path}")
        
#         plt.close(fig)

#         del model
#         del input_data
#         del method
#         del result
#         del jf
#         del jf_normalized
        
#         gc.collect()
#         torch.cuda.empty_cache()



# import sys
# import os
# import itertools
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# if "cebra" in sys.modules:
#     del sys.modules["cebra"]

# from utils.constants import CEBRA_DIR
# sys.path.insert(0, str(CEBRA_DIR))
# import cebra 
# from cebra import CEBRA
# from utils.dataset_loader import DatasetLoader
# from torch.utils.data import DataLoader, TensorDataset
# from utils.normalization import normalize_to_target
# from utils.min_distance import min_l2_distance

# # names = ['buddy']
# names = [
#     'achilles',
#     # 'gatsby',
#     # 'buddy',
#     # 'cicero'
# ]

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# NUM_FAKE_NEURONS = 5

# for name in names:
#     print(f"\n========== Processing Rat: {name} ==========")
    
#     # --- 1. Load Dataset and Inject Fake Neurons ---
#     dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
    
#     neural_data = dataset.neural.clone() if torch.is_tensor(dataset.neural) else torch.tensor(dataset.neural)
#     num_samples, num_real_neurons = neural_data.shape
    
#     fake_data = torch.tensor(np.random.binomial(n=1, p=0.5, size=(num_samples, NUM_FAKE_NEURONS)), dtype=neural_data.dtype)
    
#     total_neurons = num_real_neurons + NUM_FAKE_NEURONS
#     fake_indices = np.sort(np.random.choice(total_neurons, NUM_FAKE_NEURONS, replace=False))
#     real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)
    
#     combined_neural = torch.zeros((num_samples, total_neurons), dtype=neural_data.dtype)
#     combined_neural[:, real_indices] = neural_data
#     combined_neural[:, fake_indices] = fake_data
    
#     print(f"Added {NUM_FAKE_NEURONS} fake neurons at indices: {fake_indices.tolist()}")
    
#     split_idx = int(0.8 * len(combined_neural))
#     train_data = combined_neural[:split_idx]
#     valid_data = combined_neural[split_idx:]
    
#     continuous_index = dataset.continuous_index.clone() if torch.is_tensor(dataset.continuous_index) else torch.tensor(dataset.continuous_index)
#     train_continuous_label = continuous_index[:split_idx, :2].numpy()
#     valid_continuous_label = continuous_index[split_idx:, :2].numpy()

#     save_dir = os.path.join("images", name)
#     os.makedirs(save_dir, exist_ok=True)
    
#     results = {}

#     for adv in [False, True]:
#         model_name = "ACORN" if adv else "CEBRA"
#         adv_epsilon = 0.5
#         epochs = 1500
      
#         print(f"\n--- Training {model_name} (adv = {adv}) ---")
        
#         model = CEBRA(
#             batch_size=1024,
#             temperature=0.4,
#             model_architecture="offset36-model-more-dropout",
#             time_offsets=4,
#             max_iterations=epochs,
#             output_dimension=48,
#             verbose=True,
#             training_mode='adversarial' if adv else 'clean',
#             adv_alpha=adv_epsilon / 5 if adv else 0.0,
#             adv_epsilon=adv_epsilon if adv else 0.0,
#             adv_steps=10 if adv else 0, 
#             attack_norm="l2"
#         )
        
#         path = f"{name}_adv.pth" if adv else f"{name}.pth"
        
#         # Fit & Save
#         model.fit(train_data, train_continuous_label)
#         model.save(path)
        
#         # Load & Attribution
#         model = CEBRA.load(path, weights_only=False)
#         model = model.solver_.model.to(device)
        
#         input_data = train_data.clone().detach().to(device).requires_grad_(True)
#         method = cebra.attribution.init(
#             name="jacobian-based",
#             model=model,
#             input_data=input_data,
#             output_dimension=model.num_output
#         )
        
#         print(f"Computing Jacobian Map for {model_name}...")
#         result = method.compute_attribution_map()
#         results[adv] = result

#         jf = abs(result['jf']).mean(0)
#         jf_normalized = jf / jf.sum()
        
#         fake_jf = jf_normalized[:, fake_indices] 
#         mean_fake_latents = fake_jf.mean(axis=0)
        
#         print(f"\n>>> [{model_name}] Average Latent Attribution for Fake Neurons:")
#         for idx_order, global_idx in enumerate(fake_indices):
#             print(f"    Fake Neuron #{idx_order+1} (Index: {global_idx}): {mean_fake_latents[idx_order]:.6e}")

#         fig, ax = plt.subplots(figsize=(15, 8))
#         im = ax.matshow(
#             jf_normalized,
#             aspect="auto",
#             cmap="cividis",
#         )
#         ax.set_title(f"Rat: {name} | Model: {model_name}", fontsize=14)
        
#         for global_idx in fake_indices:
#             ax.axvline(x=global_idx, color='red', linestyle='--', alpha=0.8, linewidth=1)
            
#         fig.colorbar(im, ax=ax)
        
#         plot_path = os.path.join(save_dir, f"{model_name}_jacobian.png")
#         plt.savefig(plot_path, dpi=300, bbox_inches="tight")
#         print(f"Plot saved successfully at: {plot_path}")
#         plt.show()
