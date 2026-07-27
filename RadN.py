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
    ("Mihili_RT_2013_2014_npz", "Mihili_20131207_001_RT.mat.npz"),
    ("Jango_ISO_2015_npz", "Jango_20150730_001.mat.npz"),
    ("Mihili_CO_2014_npz", "Mihili_20140203_001.mat.npz"),
    ("Chewie_CO_2016_npz", "Chewie_20160927_001.mat.npz")
]

out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")

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
        np.random.binomial(n=1, p=0.5, size=(num_samples, num_fake_neurons)),
        dtype=neural_data.dtype,
    )

    total_neurons = num_real_neurons + num_fake_neurons
    fake_indices = np.sort(np.random.choice(total_neurons, num_fake_neurons, replace=False))
    real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)

    combined_neural = torch.zeros((num_samples, total_neurons), dtype=neural_data.dtype)

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
        optimizer,
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

    save_dir = os.path.join(img_dir, target_file.replace(".mat.npz", "").replace(".", "_"))
    os.makedirs(save_dir, exist_ok=True)

    models_store = {}
    topk_store = {}
    results = {}
    
    k_neurons = int(np.sqrt(total_neurons))

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

        jf_key = "jf" 
        jfinv_key = "jf-inv-svd" if "jf-inv-svd" in result else "jf-inv"
        
        jf_tensor = torch.as_tensor(result[jf_key]).detach().cpu()
        jfinv_tensor = torch.as_tensor(result[jfinv_key]).detach().cpu()
        results[model_name] = {"jf": jf_tensor, "jf-inv": jfinv_tensor}

        jf_mean = torch.abs(jf_tensor).mean(dim=0)
        jf_scores = jf_mean.sum(dim=0) if jf_mean.shape[1] == total_neurons else jf_mean.sum(dim=1)

        jfinv_mean = torch.abs(jfinv_tensor).mean(dim=0)
        jfinv_scores = jfinv_mean.sum(dim=0) if jfinv_mean.shape[1] == total_neurons else jfinv_mean.sum(dim=1)

        topk_jf_indices = torch.topk(jf_scores, k_neurons).indices.numpy()
        topk_jfinv_indices = torch.topk(jfinv_scores, k_neurons).indices.numpy()
        
        print(f"[{model_name}] Selected Top K={k_neurons} out of {total_neurons} neurons.")
        topk_store[model_name] = {
            "jf": topk_jf_indices,
            "jfinv": topk_jfinv_indices
        }

        cleanup_cuda(method, trained_model, input_tensor, result)

        print(f"\n--- Training Decoder for {model_name} ---")
        test_true_np = y_np[split_idx:].astype(np.float32)
        decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
            cebra_model=model,
            train_x_np=train_data_np,
            train_y_np=y_np[:split_idx].astype(np.float32),
            test_x_np=valid_data_np,
            test_y_np=test_true_np,
            input_dim=48,
            hidden_dim=64,
            dropout_rate=0.4,
            decoder_iters=10000,
        )

        print(f"** Base Test R2 Score for {model_name}: {mean_r2:.4f} **")

        decoder_cpu = copy.deepcopy(decoder).cpu()
        models_store[model_name] = {
            "cebra_model": model,
            "decoder": decoder_cpu,
            "base_r2": mean_r2
        }

        decoder_save_path = os.path.join(out_dir, f"decoder_{model_name}_{target_file}.pth")
        torch.save(decoder.state_dict(), decoder_save_path)
        cleanup_cuda(decoder)

    print("\n" + "="*70)
    print(" RUNNING CROSS-MODEL LESION EVALUATION (RAD-N TESTING) ".center(70, "="))
    print("="*70)

    test_true_np = y_np[split_idx:].astype(np.float32)
    cross_results = {}

    for eval_model_name in ["CEBRA", "ACORN"]:
        cebra_model = models_store[eval_model_name]["cebra_model"]
        decoder = models_store[eval_model_name]["decoder"].to(device)
        decoder.eval()

        cross_results[eval_model_name] = {
            "base": models_store[eval_model_name]["base_r2"]
        }

        with torch.no_grad():
            for source_model_name in ["ACORN", "CEBRA"]:
                idx_jf = topk_store[source_model_name]["jf"]
                valid_masked_jf = valid_data_np.copy()
                mask_jf = np.ones(total_neurons, dtype=bool)
                mask_jf[idx_jf] = False
                valid_masked_jf[:, mask_jf] = 0.0

                emb_jf = get_embeddings(cebra_model, valid_masked_jf)
                preds_jf = decoder(torch.from_numpy(emb_jf).float().to(device)).cpu().numpy()
                r2_jf, _ = mean_r2_score(test_true_np, preds_jf)

                idx_jfinv = topk_store[source_model_name]["jfinv"]
                valid_masked_jfinv = valid_data_np.copy()
                mask_jfinv = np.ones(total_neurons, dtype=bool)
                mask_jfinv[idx_jfinv] = False
                valid_masked_jfinv[:, mask_jfinv] = 0.0

                emb_jfinv = get_embeddings(cebra_model, valid_masked_jfinv)
                preds_jfinv = decoder(torch.from_numpy(emb_jfinv).float().to(device)).cpu().numpy()
                r2_jfinv, _ = mean_r2_score(test_true_np, preds_jfinv)

                cross_results[eval_model_name][f"with_radN_{source_model_name.lower()}_jf"] = r2_jf
                cross_results[eval_model_name][f"with_radN_{source_model_name.lower()}_jfinv"] = r2_jfinv

        cleanup_cuda(decoder)

    print("\n" + "#" * 80)
    print(f" FINAL CROSS-LESION SUMMARY FOR: {dataset_name} (K = {k_neurons}) ".center(80, "#"))
    print("#" * 80)
    
    print("\n--- [Jacobian / Jf Results] ---")
    adv_res = cross_results["ACORN"]
    clean_res = cross_results["CEBRA"]
    
    print(f"adv base:   {adv_res['base']:>7.4f} | adv with radN adv:   {adv_res['with_radN_acorn_jf']:>7.4f} | adv with radN clean:   {adv_res['with_radN_cebra_jf']:>7.4f}")
    print(f"clean base: {clean_res['base']:>7.4f} | clean with radN adv: {clean_res['with_radN_acorn_jf']:>7.4f} | clean with radN clean: {clean_res['with_radN_cebra_jf']:>7.4f}")

    print("\n--- [Inverse Jacobian / Jf-inv Results] ---")
    print(f"adv base:   {adv_res['base']:>7.4f} | adv with radN adv:   {adv_res['with_radN_acorn_jfinv']:>7.4f} | adv with radN clean:   {adv_res['with_radN_cebra_jfinv']:>7.4f}")
    print(f"clean base: {clean_res['base']:>7.4f} | clean with radN adv: {clean_res['with_radN_acorn_jfinv']:>7.4f} | clean with radN clean: {clean_res['with_radN_cebra_jfinv']:>7.4f}")
    print("#" * 80 + "\n")

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

        im = ax.matshow(jfinv_np, aspect="auto")
        ims.append(im)

        self_masked_r2 = cross_results[name][f"with_radN_{name.lower()}_jfinv"]
        ax.set_title(f"{name}\nBase R2={cross_results[name]['base']:.3f} | Self-Masked Jf-inv R2={self_masked_r2:.3f}", pad=20)
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
    
    cleanup_cuda(models_store, topk_store, results, cross_results)






###################################
##### Only for radN same model ####
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
    ("Mihili_RT_2013_2014_npz", "Mihili_20131207_001_RT.mat.npz"),
    ("Jango_ISO_2015_npz", "Jango_20150730_001.mat.npz"),
    ("Mihili_CO_2014_npz", "Mihili_20140203_001.mat.npz"),
    ("Chewie_CO_2016_npz", "Chewie_20160927_001.mat.npz")
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
        np.random.binomial(n=1, p=0.5, size=(num_samples, num_fake_neurons)),
        dtype=neural_data.dtype,
    )

    total_neurons = num_real_neurons + num_fake_neurons
    fake_indices = np.sort(np.random.choice(total_neurons, num_fake_neurons, replace=False))
    real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)

    combined_neural = torch.zeros((num_samples, total_neurons), dtype=neural_data.dtype)

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

        # Extract Jacobian and Inverse Jacobian safely
        jf_key = "jf" 
        jfinv_key = "jf-inv-svd" if "jf-inv-svd" in result else "jf-inv"
        
        jf_tensor = torch.as_tensor(result[jf_key]).detach().cpu()
        jfinv_tensor = torch.as_tensor(result[jfinv_key]).detach().cpu()
        results[model_name] = {"jf": jf_tensor, "jf-inv": jfinv_tensor}

        # Calculate Neuron Scores (Mean over samples, Sum over latents)
        # Jf is usually (Samples, Latents, Neurons)
        jf_mean = torch.abs(jf_tensor).mean(dim=0)
        jf_scores = jf_mean.sum(dim=0) if jf_mean.shape[1] == total_neurons else jf_mean.sum(dim=1)

        # Jf-inv is usually (Samples, Neurons, Latents)
        jfinv_mean = torch.abs(jfinv_tensor).mean(dim=0)
        jfinv_scores = jfinv_mean.sum(dim=0) if jfinv_mean.shape[1] == total_neurons else jfinv_mean.sum(dim=1)

        # Determine K = sqrt(N)
        k_neurons = int(np.sqrt(total_neurons))
        topk_jf_indices = torch.topk(jf_scores, k_neurons).indices.numpy()
        topk_jfinv_indices = torch.topk(jfinv_scores, k_neurons).indices.numpy()
        
        print(f"[{model_name}] Selected Top K={k_neurons} out of {total_neurons} neurons.")

        cleanup_cuda(method, trained_model, input_tensor, result)

        print(f"\n--- Training Decoder for {model_name} ---")
        test_true_np = y_np[split_idx:].astype(np.float32)
        decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
            cebra_model=model,
            train_x_np=train_data_np,
            train_y_np=y_np[:split_idx].astype(np.float32),
            test_x_np=valid_data_np,
            test_y_np=test_true_np,
            input_dim=48,
            hidden_dim=64,
            dropout_rate=0.4,
            decoder_iters=10000,
        )

        print(f"** Base Test R2 Score for {model_name}: {mean_r2:.4f} **")

        # -----------------------------
        # Masked Evaluation (Lesion Test)
        # -----------------------------
        decoder.eval()
        with torch.no_grad():
            # 1. Mask based on Jacobian (Jf)
            valid_masked_jf = valid_data_np.copy()
            mask_jf = np.ones(total_neurons, dtype=bool)
            mask_jf[topk_jf_indices] = False  # False means DO NOT zero out
            valid_masked_jf[:, mask_jf] = 0.0 # Zero out non-top K neurons

            emb_jf = get_embeddings(model, valid_masked_jf)
            preds_jf = decoder(torch.from_numpy(emb_jf).float().to(device)).cpu().numpy()
            r2_jf, _ = mean_r2_score(test_true_np, preds_jf)

            # 2. Mask based on Inverse Jacobian (Jf-inv)
            valid_masked_jfinv = valid_data_np.copy()
            mask_jfinv = np.ones(total_neurons, dtype=bool)
            mask_jfinv[topk_jfinv_indices] = False
            valid_masked_jfinv[:, mask_jfinv] = 0.0

            emb_jfinv = get_embeddings(model, valid_masked_jfinv)
            preds_jfinv = decoder(torch.from_numpy(emb_jfinv).float().to(device)).cpu().numpy()
            r2_jfinv, _ = mean_r2_score(test_true_np, preds_jfinv)

        print(f"** Masked Test R2 (Top {k_neurons} from Jf):     {r2_jf:.4f} **")
        print(f"** Masked Test R2 (Top {k_neurons} from Jf-inv): {r2_jfinv:.4f} **\n")

        r2_results[model_name] = {
            "mean_r2": mean_r2,
            "masked_jf_r2": r2_jf,
            "masked_jfinv_r2": r2_jfinv,
            "per_dim_r2": per_dim_r2,
        }

        decoder_save_path = os.path.join(out_dir, f"decoder_{model_name}_{target_file}.pth")
        torch.save(decoder.state_dict(), decoder_save_path)
        cleanup_cuda(model, decoder)

    print("\n" + "=" * 80)
    print(f" SUMMARY OF R2 SCORES FOR {dataset_name} ".center(80, "="))
    print("=" * 80)
    for name, scores in r2_results.items():
        print(f" Model: {name:<6} | Base R2: {scores['mean_r2']:.4f} | Masked Jf R2: {scores['masked_jf_r2']:.4f} | Masked Jf-inv R2: {scores['masked_jfinv_r2']:.4f}")
    print("=" * 80)

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

        im = ax.matshow(jfinv_np, aspect="auto")
        ims.append(im)

        ax.set_title(f"{name}\nBase R2={r2_results[name]['mean_r2']:.3f} | Masked Jf-inv R2={r2_results[name]['masked_jfinv_r2']:.3f}", pad=20)
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
