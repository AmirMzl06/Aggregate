import os
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

import sys
sys.path.insert(0, str(CEBRA_DIR))
import cebra
import cebra.attribution
from cebra import CEBRA


# ============================================================
# Config
# ============================================================
SMD_ROOT = os.path.join(DATA_DIR, "SMD")
TRAIN_DIR = os.path.join(SMD_ROOT, "train")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
MAX_ITER = 2500
ATTR_BATCH_SIZE = 128
RANDOM_SEED = 42

OUTPUT_DIM = 16
DECODER_HIDDEN = 64
DECODER_ITERS = 10000
DECODER_PATIENCE = 1000
DECODER_MIN_EPOCHS = 4000

TRAIN_RATIO = 0.8
TOPK_MODE = "sqrt"  # sqrt(#features)

SAVE_MODELS = True
SAVE_NPZ = True

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
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=2, dropout_rate=0.2):
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
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Helpers
# ============================================================
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


def load_smd_matrix(path: str) -> np.ndarray:
    x = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    return x.astype(np.float32)


def time_labels(n_steps: int) -> np.ndarray:
    if n_steps <= 1:
        return np.zeros((n_steps, 1), dtype=np.float32)
    return np.linspace(0.0, 1.0, n_steps, dtype=np.float32).reshape(-1, 1)


def split_series(x: np.ndarray, ratio: float = TRAIN_RATIO):
    split_idx = int(len(x) * ratio)
    return x[:split_idx], x[split_idx:]


def mean_r2_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(r2_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(scores)), scores


def get_embeddings(cebra_model, x_np):
    x_t = torch.from_numpy(x_np).float()
    emb = cebra_model.transform(x_t)
    if isinstance(emb, torch.Tensor):
        return emb.detach().cpu().numpy()
    return np.asarray(emb)


def reduce_attr_to_matrix(attr_tensor, n_features: int):
    if torch.is_tensor(attr_tensor):
        attr = attr_tensor.detach().cpu()
    else:
        attr = torch.as_tensor(attr_tensor).cpu()

    attr = torch.abs(attr)

    if attr.ndim == 3:
        attr_2d = attr.mean(dim=0)
    elif attr.ndim == 2:
        attr_2d = attr
    elif attr.ndim == 1:
        attr_2d = attr[None, :]
    else:
        raise ValueError(f"Unsupported attribution shape: {tuple(attr.shape)}")

    if attr_2d.shape[1] == n_features:
        mat = attr_2d
    elif attr_2d.shape[0] == n_features:
        mat = attr_2d.T
    else:
        raise ValueError(
            f"Cannot identify feature axis. Reduced attribution shape={tuple(attr_2d.shape)}, "
            f"n_features={n_features}"
        )

    return mat.numpy().astype(np.float32)


def normalize_for_plot(mat):
    mat = np.asarray(mat, dtype=np.float32)
    s = float(mat.sum())
    if s > 0:
        mat = mat / s
    return mat


def save_heatmap(mat, save_path, title, xlabel="Feature / Channel", ylabel="Latent dim"):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(normalize_for_plot(mat), aspect="auto", cmap="cividis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, shrink=0.9, label="Normalized |attribution|")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_model_plots(save_dir, dataset_name, model_tag, jf_map, jfinv_map, r2_value):
    os.makedirs(save_dir, exist_ok=True)

    jf_path = os.path.join(save_dir, f"{dataset_name}_{model_tag}_JC.png")
    jfinv_path = os.path.join(save_dir, f"{dataset_name}_{model_tag}_JC_INV.png")
    pair_path = os.path.join(save_dir, f"{dataset_name}_{model_tag}_JC_PAIR.png")

    save_heatmap(
        jf_map,
        jf_path,
        title=f"{model_tag} | Jc | R2={r2_value:.4f}",
    )
    save_heatmap(
        jfinv_map,
        jfinv_path,
        title=f"{model_tag} | Jc-inv | R2={r2_value:.4f}",
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    mats = [normalize_for_plot(jf_map), normalize_for_plot(jfinv_map)]
    titles = [f"{model_tag} | Jc", f"{model_tag} | Jc-inv"]

    last_im = None
    for ax, mat, title in zip(axes, mats, titles):
        last_im = ax.imshow(mat, aspect="auto", cmap="cividis")
        ax.set_title(title)
        ax.set_xlabel("Feature / Channel")
        ax.set_ylabel("Latent dim")

    fig.suptitle(f"{dataset_name} | {model_tag} | R2={r2_value:.4f}", fontsize=15)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.9, label="Normalized |attribution|")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(pair_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return jf_path, jfinv_path, pair_path


def build_cebra_model(adv: bool, adv_epsilon: float, output_dim: int = OUTPUT_DIM):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=MAX_ITER,
        output_dimension=output_dim,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=adv_epsilon / 5 if adv else 0.0,
        adv_epsilon=adv_epsilon if adv else 0.0,
        adv_steps=10 if adv else 0,
        attack_norm="linf",
        num_hidden_units=32,
        device="cuda_if_available",
    )


def train_decoder_reconstruction(
    cebra_model,
    train_input_np,
    train_target_np,
    test_input_np,
    test_target_np,
    embedding_dim,
    target_dim,
    hidden_dim=DECODER_HIDDEN,
    dropout_rate=0.2,
    decoder_iters=DECODER_ITERS,
):
    train_in, val_in, train_y, val_y = train_test_split(
        train_input_np,
        train_target_np,
        test_size=0.125,
        random_state=42,
        shuffle=False,
    )

    z_train = torch.from_numpy(get_embeddings(cebra_model, train_in)).float().to(device)
    z_val = torch.from_numpy(get_embeddings(cebra_model, val_in)).float().to(device)
    z_test = torch.from_numpy(get_embeddings(cebra_model, test_input_np)).float().to(device)

    y_train = torch.from_numpy(train_y).float().to(device)
    y_val = torch.from_numpy(val_y).float().to(device)
    y_test = torch.from_numpy(test_target_np).float().to(device)

    decoder = TwoLayerMLP(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=target_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=2e-4)

    initial_state = copy.deepcopy(decoder.state_dict())
    best_r2 = -1e18
    best_epoch = 1
    best_state = copy.deepcopy(decoder.state_dict())
    bad_epochs = 0

    for epoch in range(decoder_iters):
        decoder.train()
        optimizer.zero_grad()
        preds = decoder(z_train)
        loss = criterion(preds, y_train)
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
            best_state = copy.deepcopy(decoder.state_dict())
        else:
            if epoch > DECODER_MIN_EPOCHS - DECODER_PATIENCE:
                bad_epochs += 1

        if bad_epochs >= DECODER_PATIENCE:
            print(f"Early stopping decoder at epoch {epoch + 1}")
            break

        if (epoch + 1) % 2000 == 0:
            print(
                f"Decoder Epoch [{epoch + 1}/{decoder_iters}] | "
                f"Loss: {loss.item():.4f} | Val R2: {current_r2:.4f}"
            )

    decoder.load_state_dict(best_state)

    z_full = torch.cat([z_train, z_val], dim=0)
    y_full = torch.cat([y_train, y_val], dim=0)

    decoder.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=2e-4)

    for _ in range(best_epoch):
        decoder.train()
        optimizer.zero_grad()
        preds = decoder(z_full)
        loss = criterion(preds, y_full)
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
        decoder, optimizer, z_full, y_full,
        train_in, val_in, train_y, val_y
    )

    return decoder, mean_test_r2, per_dim_r2


def train_branch(
    dataset_name,
    branch_name,
    train_input_np,
    train_time_np,
    train_target_np,
    test_input_np,
    test_target_np,
    adv: bool,
):
    n_features = train_input_np.shape[1]

    train_tensor = torch.from_numpy(train_input_np).float()
    adv_epsilon = float(min_l2_distance(train_tensor)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)

    model = build_cebra_model(adv=adv, adv_epsilon=adv_epsilon, output_dim=OUTPUT_DIM)
    model.fit(train_input_np, train_time_np)

    model_tag = "ACORN" if adv else "CEBRA"
    save_path = os.path.join(out_dir, f"SMD_{dataset_name}_{model_tag}.pth")
    if SAVE_MODELS:
        model.save(save_path)
        print("Saved model to:", save_path)

    trained_model = model.solver_.model.to(device)
    if hasattr(trained_model, "split_outputs"):
        trained_model.split_outputs = False
    trained_model.eval()

    input_tensor = torch.from_numpy(train_input_np).float().to(device).requires_grad_(True)

    method = cebra.attribution.init(
        name="jacobian-based-batched",
        model=trained_model,
        input_data=input_tensor,
        output_dimension=int(getattr(trained_model, "num_output", OUTPUT_DIM)),
    )

    result = method.compute_attribution_map(batch_size=min(ATTR_BATCH_SIZE, len(train_input_np)))
    print("Attribution keys:", list(result.keys()))

    jf_key = "jf"
    jfinv_key = "jf-inv-svd" if "jf-inv-svd" in result else "jf-inv"

    jf_tensor = torch.as_tensor(result[jf_key]).detach().cpu()
    jfinv_tensor = torch.as_tensor(result[jfinv_key]).detach().cpu()

    jf_map = reduce_attr_to_matrix(jf_tensor, n_features)
    jfinv_map = reduce_attr_to_matrix(jfinv_tensor, n_features)

    jf_scores = np.abs(jf_map).mean(axis=0)
    jfinv_scores = np.abs(jfinv_map).mean(axis=0)

    if TOPK_MODE == "sqrt":
        k = int(np.sqrt(n_features))
    else:
        k = min(10, n_features)

    topk_jf = np.argsort(jf_scores)[::-1][:k]
    topk_jfinv = np.argsort(jfinv_scores)[::-1][:k]

    print(f"[{model_tag}] Top-K = {k} out of {n_features}")
    print(f"[{model_tag}] Top-K (Jc):    {topk_jf.tolist()}")
    print(f"[{model_tag}] Top-K (Jc-inv): {topk_jfinv.tolist()}")

    raw_attr_path = os.path.join(out_dir, f"SMD_{dataset_name}_{model_tag}_raw_jacobians.pt")
    torch.save(
        {
            "jf": jf_tensor,
            "jf_inv": jfinv_tensor,
        },
        raw_attr_path,
    )
    print(f"Saved raw Jacobians to: {raw_attr_path}")

    npz_attr_path = os.path.join(out_dir, f"SMD_{dataset_name}_{model_tag}_attrs.npz")
    if SAVE_NPZ:
        np.savez_compressed(
            npz_attr_path,
            jf_map=jf_map.astype(np.float32),
            jfinv_map=jfinv_map.astype(np.float32),
            jf_scores=jf_scores.astype(np.float32),
            jfinv_scores=jfinv_scores.astype(np.float32),
            topk_jf=topk_jf.astype(np.int32),
            topk_jfinv=topk_jfinv.astype(np.int32),
            adv_epsilon=np.array([adv_epsilon], dtype=np.float32),
        )
        print(f"Saved attribution npz to: {npz_attr_path}")

    # Save plots in images/
    plot_dir = os.path.join(img_dir, "SMD", dataset_name)
    jf_plot_path, jfinv_plot_path, pair_plot_path = save_model_plots(
        save_dir=plot_dir,
        dataset_name=dataset_name,
        model_tag=model_tag,
        jf_map=jf_map,
        jfinv_map=jfinv_map,
        r2_value=np.nan,  # filled later in summary; plots still saved now
    )

    decoder, base_r2, per_dim_r2 = train_decoder_reconstruction(
        cebra_model=model,
        train_input_np=train_input_np,
        train_target_np=train_target_np,
        test_input_np=test_input_np,
        test_target_np=test_target_np,
        embedding_dim=OUTPUT_DIM,
        target_dim=train_target_np.shape[1],
        hidden_dim=DECODER_HIDDEN,
        dropout_rate=0.2,
        decoder_iters=DECODER_ITERS,
    )

    decoder_path = os.path.join(out_dir, f"SMD_{dataset_name}_{model_tag}_decoder.pth")
    torch.save(decoder.state_dict(), decoder_path)
    print("Saved decoder to:", decoder_path)

    cleanup_cuda(method, trained_model, input_tensor, result, jf_tensor, jfinv_tensor, decoder, model)

    return {
        "model_tag": model_tag,
        "adv": adv,
        "adv_epsilon": adv_epsilon,
        "jf_map": jf_map,
        "jfinv_map": jfinv_map,
        "jf_scores": jf_scores,
        "jfinv_scores": jfinv_scores,
        "topk_jf": topk_jf,
        "topk_jfinv": topk_jfinv,
        "base_r2": base_r2,
        "per_dim_r2": per_dim_r2,
        "n_features": n_features,
        "raw_attr_path": raw_attr_path,
        "npz_attr_path": npz_attr_path,
        "decoder_path": decoder_path,
        "jf_plot_path": jf_plot_path,
        "jfinv_plot_path": jfinv_plot_path,
        "pair_plot_path": pair_plot_path,
    }


# ============================================================
# Main
# ============================================================
def main():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"SMD train directory not found: {TRAIN_DIR}")

    train_files = sorted([f for f in os.listdir(TRAIN_DIR) if f.endswith(".txt")])
    if not train_files:
        raise RuntimeError(f"No .txt files found in {TRAIN_DIR}")

    global_rows = []

    for filename in train_files:
        dataset_name = os.path.splitext(filename)[0]
        file_path = os.path.join(TRAIN_DIR, filename)

        print("\n" + "=" * 100)
        print(f"Processing SMD file: {filename}")
        print("=" * 100)

        x_raw = load_smd_matrix(file_path)
        print("Raw shape:", x_raw.shape)

        t_all = time_labels(len(x_raw))
        train_x, test_x = split_series(x_raw, TRAIN_RATIO)
        train_t, test_t = split_series(t_all, TRAIN_RATIO)

        print("Train shape:", train_x.shape, "| Test shape:", test_x.shape)

        full_results = {}

        # Full models: raw data only
        for adv in [False, True]:
            model_name = "ACORN" if adv else "CEBRA"
            print(f"\n==================== Training FULL {model_name} ====================")

            res = train_branch(
                dataset_name=dataset_name,
                branch_name="full",
                train_input_np=train_x.astype(np.float32),
                train_time_np=train_t.astype(np.float32),
                train_target_np=train_x.astype(np.float32),
                test_input_np=test_x.astype(np.float32),
                test_target_np=test_x.astype(np.float32),
                adv=adv,
            )
            full_results[model_name] = res

            # Re-save plot with the final R2 in the title if you want consistency:
            save_model_plots(
                save_dir=os.path.join(img_dir, "SMD", dataset_name),
                dataset_name=dataset_name,
                model_tag=model_name,
                jf_map=res["jf_map"],
                jfinv_map=res["jfinv_map"],
                r2_value=res["base_r2"],
            )

            global_rows.append({
                "dataset": dataset_name,
                "setting": "full",
                "source_model": model_name,
                "retrained_model": model_name,
                "n_features": x_raw.shape[1],
                "base_r2": res["base_r2"],
                "adv_epsilon": res["adv_epsilon"],
                "topk_k": len(res["topk_jf"]),
                "raw_attr_path": res["raw_attr_path"],
                "npz_attr_path": res["npz_attr_path"],
                "decoder_path": res["decoder_path"],
                "jf_plot_path": res["jf_plot_path"],
                "jfinv_plot_path": res["jfinv_plot_path"],
                "pair_plot_path": res["pair_plot_path"],
            })

        print("\n" + "#" * 90)
        print(f" SUMMARY FOR {dataset_name} | raw ".center(90, "#"))
        print("#" * 90)

        for model_name in ["CEBRA", "ACORN"]:
            print(
                f"{model_name:>6} | full base R2 = {full_results[model_name]['base_r2']:.4f} | "
                f"topJc K = {len(full_results[model_name]['topk_jf'])} | "
                f"topJc-inv K = {len(full_results[model_name]['topk_jfinv'])}"
            )

        print(f"\nSaved plots under: {os.path.join(img_dir, 'SMD', dataset_name)}")
        csv_path = os.path.join(out_dir, f"SMD_{dataset_name}_raw_full_results.csv")
        pd.DataFrame([r for r in global_rows if r["dataset"] == dataset_name]).to_csv(csv_path, index=False)
        print(f"Saved CSV to: {csv_path}")

        cleanup_cuda(full_results)

    global_csv = os.path.join(out_dir, "SMD_raw_global_summary.csv")
    pd.DataFrame(global_rows).to_csv(global_csv, index=False)
    print(f"\nSaved global CSV summary to: {global_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
