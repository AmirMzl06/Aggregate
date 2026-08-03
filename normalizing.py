import os
import sys
import gc
import copy
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
    "achilles",
    "buddy",
    "cicero",
    "gatsby",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIM = 48
BATCH_SIZE = 2048
MAX_ITER = 2500
ATTR_BATCH_SIZE = 128
RANDOM_SEED = 42

NUM_FAKE_NEURONS = 0  # raw only
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


def zscore_fit_transform(train_x, test_x, eps=1e-8):
    """
    Per-neuron z-score using train statistics only.
    """
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)

    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma = np.maximum(sigma, eps)

    train_z = (train_x - mu) / sigma
    test_z = (test_x - mu) / sigma
    return train_z.astype(np.float32), test_z.astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def reduce_attr_to_matrix(attr_tensor, total_neurons):
    """
    Convert attribution tensor to [latent_dim, neurons] when possible.
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

    return mat.detach().cpu().numpy().astype(np.float32)


def normalize_map_for_plot(mat):
    mat = np.asarray(mat, dtype=np.float32)
    s = float(mat.sum())
    if s > 0:
        mat = mat / s
    return mat


def save_four_panel_figure(
    save_path,
    dataset_name,
    clean_jf_map,
    acorn_jf_map,
    clean_jfinv_map,
    acorn_jfinv_map,
    clean_r2,
    acorn_r2,
):
    mats = [
        normalize_map_for_plot(clean_jf_map),
        normalize_map_for_plot(acorn_jf_map),
        normalize_map_for_plot(clean_jfinv_map),
        normalize_map_for_plot(acorn_jfinv_map),
    ]

    vmax = max(float(m.max()) for m in mats)
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    titles = [
        f"CEBRA | Jf\nR2={clean_r2:.4f}",
        f"ACORN | Jf\nR2={acorn_r2:.4f}",
        f"CEBRA | Jf-inv\nR2={clean_r2:.4f}",
        f"ACORN | Jf-inv\nR2={acorn_r2:.4f}",
    ]

    last_im = None
    for ax, mat, title in zip(axes.flat, mats, titles):
        last_im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=vmax, cmap="cividis")
        ax.set_title(title)
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Latent dim")

    fig.suptitle(f"{dataset_name} | raw z-score", fontsize=16)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9, label="Normalized |attribution|")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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

    return decoder, mean_test_r2, per_dim_r2


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

    jf_map = reduce_attr_to_matrix(jf_tensor, total_neurons)
    jfinv_map = reduce_attr_to_matrix(jfinv_tensor, total_neurons)

    jf_scores = jf_map.mean(axis=0)
    jfinv_scores = jfinv_map.mean(axis=0)

    print(f"[{save_name}] Attribution matrix shape (Jf): {jf_map.shape}")
    print(f"[{save_name}] Attribution matrix shape (Jf-inv): {jfinv_map.shape}")

    decoder, base_r2, per_dim_r2 = train_decoder_with_same_arch(
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

    cleanup_cuda(model, trained_model, input_tensor, method, result, jf_tensor, jfinv_tensor, decoder)

    return {
        "base_r2": base_r2,
        "per_dim_r2": per_dim_r2,
        "jf_tensor": jf_tensor,
        "jfinv_tensor": jfinv_tensor,
        "jf_map": jf_map,
        "jfinv_map": jfinv_map,
        "jf_scores": jf_scores,
        "jfinv_scores": jfinv_scores,
        "adv_epsilon": adv_epsilon,
        "save_name": save_name,
    }


# ============================================================
# Main
# ============================================================
rows = []

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

    # position + direction
    y_np = continuous_index[:, :2].numpy().astype(np.float32)

    print(f"Raw neural shape: {neural_data.shape} | Labels shape: {y_np.shape}")

    neural_np = neural_data.detach().cpu().numpy().astype(np.float32)

    split_idx = int(0.8 * len(neural_np))
    train_raw = neural_np[:split_idx]
    test_raw = neural_np[split_idx:]
    train_y = y_np[:split_idx]
    test_y = y_np[split_idx:]

    # z-score using train statistics only
    train_data_np, valid_data_np, mu, sigma = zscore_fit_transform(train_raw, test_raw)
    print("Z-scored neural shape:", train_data_np.shape, valid_data_np.shape)

    save_dir = os.path.join(img_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    full_results = {}

    for adv in [False, True]:
        cleanup_cuda()

        model_name = "ACORN" if adv else "CEBRA"
        print(f"\n==================== Training FULL {model_name} ====================")

        res = train_full_model_and_get_attr(
            train_x_np=train_data_np,
            train_y_np=train_y,
            test_x_np=valid_data_np,
            test_y_np=test_y,
            adv=adv,
            total_neurons=train_data_np.shape[1],
            output_dim=OUTPUT_DIM,
        )
        full_results[model_name] = res

        npz_path = os.path.join(out_dir, f"{name}_{model_name}_raw_zscore_attrs.npz")
        np.savez_compressed(
            npz_path,
            jf_map=res["jf_map"].astype(np.float32),
            jfinv_map=res["jfinv_map"].astype(np.float32),
            jf_scores=res["jf_scores"].astype(np.float32),
            jfinv_scores=res["jfinv_scores"].astype(np.float32),
            base_r2=np.array([res["base_r2"]], dtype=np.float32),
            adv_epsilon=np.array([res["adv_epsilon"]], dtype=np.float32),
            zscore_mu=mu,
            zscore_sigma=sigma,
        )
        print(f"Saved attribution npz to: {npz_path}")

        rows.append({
            "dataset": name,
            "setting": "raw_zscore_full",
            "model": model_name,
            "neurons": train_data_np.shape[1],
            "base_r2": res["base_r2"],
            "adv_epsilon": res["adv_epsilon"],
            "artifact": npz_path,
        })

    panel_path = os.path.join(save_dir, f"{name}_raw_zscore_4panel.png")
    save_four_panel_figure(
        save_path=panel_path,
        dataset_name=name,
        clean_jf_map=full_results["CEBRA"]["jf_map"],
        acorn_jf_map=full_results["ACORN"]["jf_map"],
        clean_jfinv_map=full_results["CEBRA"]["jfinv_map"],
        acorn_jfinv_map=full_results["ACORN"]["jfinv_map"],
        clean_r2=full_results["CEBRA"]["base_r2"],
        acorn_r2=full_results["ACORN"]["base_r2"],
    )
    print(f"Saved 4-panel figure to: {panel_path}")

    print("\n" + "#" * 80)
    print(f" SUMMARY FOR {name} (RAW + Z-SCORE) ".center(80, "#"))
    print("#" * 80)

    for model_name in ["CEBRA", "ACORN"]:
        print(
            f"{model_name:>6} | base R2 = {full_results[model_name]['base_r2']:.4f}"
        )

    csv_path = os.path.join(out_dir, f"{name}_raw_zscore_results.csv")
    pd.DataFrame([r for r in rows if r["dataset"] == name]).to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")

summary_csv = os.path.join(out_dir, "rat_raw_zscore_summary.csv")
pd.DataFrame(rows).to_csv(summary_csv, index=False)
print(f"\nSaved global CSV summary to: {summary_csv}")
print("\nPipeline Execution Completed for All Rat Datasets.")
