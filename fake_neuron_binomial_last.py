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
MONKEY_CONFIGS = [
    {
        "display_name": "Chewie",
        "dataset_name": "Chewie_CO_2016_npz",
        "target_file": "Chewie_20160927_001.mat.npz",
    },
    {
        "display_name": "Jango",
        "dataset_name": "Jango_ISO_2015_npz",
        "target_file": "Jango_20150730_001.mat.npz",
    },
    {
        "display_name": "Mihili_RT",
        "dataset_name": "Mihili_RT_2013_2014_npz",
        "target_file": "Mihili_20131207_001_RT.mat.npz",
    },
    {
        "display_name": "Mihili_CO",
        "dataset_name": "Mihili_CO_2014_npz",
        "target_file": "Mihili_20140203_001.mat.npz",
    },
]

out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")
adv_ep = 5

NUM_FAKE_NEURONS = 5
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


def add_fake_neurons_at_end(neural_data: torch.Tensor, num_fake_neurons: int):
    """
    Add fake neurons sampled from Bernoulli(0.5) to the end of the neuron axis.
    Returns:
        combined_neural: (T, N_real + N_fake)
        fake_indices: np.ndarray of fake neuron positions at the end
    """
    neural_data = neural_data.detach().cpu().float()
    num_samples, num_real_neurons = neural_data.shape

    if num_fake_neurons <= 0:
        return neural_data, np.array([], dtype=int)

    fake_data = torch.tensor(
        np.random.binomial(
            n=1,
            p=0.5,
            size=(num_samples, num_fake_neurons),
        ),
        dtype=neural_data.dtype,
    )

    combined_neural = torch.cat([neural_data, fake_data], dim=1)
    fake_indices = np.arange(num_real_neurons, num_real_neurons + num_fake_neurons)

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


def run_one_monkey(display_name, dataset_name, target_file):
    print(f"\n\n==================== {display_name} ====================")

    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    files = sorted(os.listdir(dataset_dir))
    if target_file not in files:
        raise ValueError(f"{target_file} not found inside {dataset_dir}")

    day_idx = files.index(target_file)
    print("Selected day index:", day_idx)
    print("Selected file:", files[day_idx])

    x_np, y_np = loader.load_dataset_day(day_idx, dataset_name, cache=True)

    print("x shape:", x_np.shape)
    print("y shape:", y_np.shape)

    neural_data = torch.from_numpy(x_np).float() if isinstance(x_np, np.ndarray) else x_np.clone().detach().float()
    combined_neural, fake_indices = add_fake_neurons_at_end(neural_data, NUM_FAKE_NEURONS)

    print(f"Added {NUM_FAKE_NEURONS} fake neurons at the END.")
    print(f"Fake neuron indices: {fake_indices.tolist()}")

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

    save_dir = os.path.join(img_dir, display_name)
    os.makedirs(save_dir, exist_ok=True)

    results = {}
    r2_results = {}

    for adv in [False, True]:
        cleanup_cuda()

        model_name = "ACORN" if adv else "CEBRA"
        print(f"\n--- Training {model_name} for {display_name} ---")

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
            num_hidden_units=32,
        )

        model.fit(train_data_np, train_continuous_label)

        save_path = os.path.join(out_dir, f"{display_name}_{model_name}_{target_file}.pth")
        model.save(save_path)
        print("Saved model to:", save_path)

        trained_model = model.solver_.model.to(device)

        N_ATTR = min(512, len(train_data_np))
        attr_idx = np.random.choice(len(train_data_np), N_ATTR, replace=False)
        attr_data = train_data_np[attr_idx]

        input_tensor = torch.from_numpy(attr_data).float().to(device).requires_grad_(True)

        output_dim = int(getattr(trained_model, "num_output", 48))
        method = cebra.attribution.init(
            name="jacobian-based",
            model=trained_model,
            input_data=input_tensor,
            output_dimension=output_dim,
        )
        result = method.compute_attribution_map()

        jf_cpu = result["jf"].detach().cpu() if torch.is_tensor(result["jf"]) else torch.tensor(result["jf"])
        results[model_name] = {"jf": jf_cpu}

        if NUM_FAKE_NEURONS > 0:
            jf_normalized = torch.abs(jf_cpu).mean(0)
            jf_normalized = jf_normalized / jf_normalized.sum()
            jf_normalized_cpu = jf_normalized.detach().cpu().numpy()

            fake_jf = jf_normalized_cpu[:, fake_indices]
            mean_fake_latents = fake_jf.mean(axis=0)

            print(f"\n>>> [{display_name} | {model_name}] Average Latent Attribution for Fake Neurons:")
            for idx_order, global_idx in enumerate(fake_indices):
                print(f"    Fake Neuron #{idx_order+1} (Index: {global_idx}): {mean_fake_latents[idx_order]:.6e}")

        cleanup_cuda(method, trained_model, input_tensor, result, jf_cpu, attr_data)

        print(f"\n--- Training Decoder for {display_name} | {model_name} ---")
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

        decoder_save_path = os.path.join(out_dir, f"decoder_{display_name}_{model_name}_{target_file}.pth")
        torch.save(decoder.state_dict(), decoder_save_path)

        print(f"** Final mean R2 Score for {display_name} | {model_name}: {mean_r2:.4f} **")
        print(f"** Per-dimension R2 for {display_name} | {model_name}: {[round(v, 4) for v in per_dim_r2]} **\n")

        cleanup_cuda(model, decoder)

    print("\n" + "=" * 50)
    print(f" SUMMARY: {display_name} ".center(50, "="))
    print("=" * 50)
    for name, scores in r2_results.items():
        print(f" Model: {name:<6} | Mean R2: {scores['mean_r2']:.4f}")
    print("=" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    model_names = ["CEBRA", "ACORN"]
    ims = []

    for ax, name in zip(axes, model_names):
        result = results[name]
        jf = torch.abs(result["jf"]).mean(0)
        jf = jf / jf.sum()
        jf = jf.detach().cpu().numpy()

        n_rows, n_cols = jf.shape

        im = ax.matshow(
            jf,
            aspect="auto",
        )
        ims.append(im)

        ax.set_title(f"{name}\nR2={r2_results[name]['mean_r2']:.3f}", pad=20)
        ax.set_xlabel(f"Latent Dimension ({n_cols})")
        ax.set_ylabel(f"Neuron ({n_rows})")

        if NUM_FAKE_NEURONS > 0:
            boundary = n_cols - NUM_FAKE_NEURONS - 0.5
            ax.axvline(
                x=boundary,
                color="red",
                linestyle="--",
                linewidth=2,
                alpha=0.9,
            )

            ax.text(
                boundary / 2,
                -3,
                "Real neurons",
                ha="center",
                va="bottom",
                fontsize=11,
            )

            ax.text(
                boundary + NUM_FAKE_NEURONS / 2,
                -3,
                "Fake neurons",
                ha="center",
                va="bottom",
                fontsize=11,
                color="red",
            )

    fig.subplots_adjust(right=0.85, top=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(ims[0], cax=cbar_ax)

    plot_path = os.path.join(
        save_dir,
        f"{target_file.replace('.mat.npz', '').replace('.', '_')}_CEBRA_vs_ACORN.png",
    )

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved figure to:", plot_path)
    print("Decoder scores:", r2_results)

    cleanup_cuda()
    return results, r2_results


# -----------------------------
# Run all four monkeys
# -----------------------------
all_results = {}
all_r2_results = {}

for cfg in MONKEY_CONFIGS:
    monkey_results, monkey_r2 = run_one_monkey(
        cfg["display_name"],
        cfg["dataset_name"],
        cfg["target_file"],
    )
    all_results[cfg["display_name"]] = monkey_results
    all_r2_results[cfg["display_name"]] = monkey_r2

print("\nDONE.")
