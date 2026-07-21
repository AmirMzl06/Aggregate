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

from utils.constants import CEBRA_DIR, DATA_DIR
from utils.dataset_loader import DatasetLoader

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA


# -----------------------------
# Config
# -----------------------------

dataset_name = "Mihili_RT_2013_2014_npz"
target_file = "Mihili_20131207_001_RT.mat.npz"

# dataset_name = "Jango_ISO_2015_npz"
# target_file = "Jango_20150730_001.mat.npz"

# dataset_name = "Mihili_CO_2014_npz"
# target_file = "Mihili_20140203_001.mat.npz"

out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")
adv_ep = 5


# -----------------------------
# Local TwoLayerMLP (no extra import)
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

    # Phase 1: early stopping on validation
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

    # Phase 2: retrain on train+val for best_epoch
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

    # Final test R2
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
# Find the exact day index for the file
# -----------------------------
dataset_dir = os.path.join(DATA_DIR, dataset_name)
files = sorted(os.listdir(dataset_dir))
day_idx = files.index(target_file)
print("Selected day index:", day_idx)
print("Selected file:", files[day_idx])

# -----------------------------
# Load one day
# -----------------------------
x_np, y_np = loader.load_dataset_day(day_idx, dataset_name, cache=True)

print("x shape:", x_np.shape)
print("y shape:", y_np.shape)

# CEBRA labels: first 2 dims
if y_np.ndim > 1 and y_np.shape[1] >= 2:
    y_cebra = y_np[:, :2]
else:
    y_cebra = y_np.reshape(-1, 1)

split_idx = int(0.8 * len(x_np))
train_data = x_np[:split_idx].astype(np.float32)
valid_data = x_np[split_idx:].astype(np.float32)

train_continuous_label = y_cebra[:split_idx].astype(np.float32)
valid_continuous_label = y_cebra[split_idx:].astype(np.float32)

results = {}
r2_results = {}

# -----------------------------
# Train CEBRA / ACORN, decoder, and attribution
# -----------------------------
for adv in [False, True]:
    cleanup_cuda()

    model_name = "ACORN" if adv else "CEBRA"
    print(f"\n==================== Training {model_name} ====================")

    model = CEBRA(
        batch_size=1024,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=2500,
        output_dimension=48,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=adv_ep / 5,
        adv_epsilon=adv_ep,
        adv_steps=10,
        attack_norm="l2",
    )

    model.fit(train_data, train_continuous_label)

    save_path = os.path.join(out_dir, f"{model_name}_{target_file}.pth")
    model.save(save_path)
    print("Saved model to:", save_path)

    # Attribution
    trained_model = model.solver_.model.to(device)

    N_ATTR = min(512, len(train_data))
    attr_idx = np.random.choice(len(train_data), N_ATTR, replace=False)
    attr_data = train_data[attr_idx]

    input_tensor = torch.from_numpy(attr_data).float().to(device).requires_grad_(True)

    output_dim = int(getattr(trained_model, "num_output", 48))
    method = cebra.attribution.init(
        name="jacobian-based",
        model=trained_model,
        input_data=input_tensor,
        output_dimension=output_dim,
    )
    result = method.compute_attribution_map()
    results[model_name] = result

    cleanup_cuda(method, trained_model, input_tensor, attr_data)

    # Decoder R2
    print(f"\n--- Training Decoder for {model_name} ---")
    decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
        cebra_model=model,
        train_x_np=train_data,
        train_y_np=y_np[:split_idx].astype(np.float32),
        test_x_np=valid_data,
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


# -----------------------------
# Print R2 summary
# -----------------------------
print("\n" + "=" * 40)
print(" SUMMARY OF R2 SCORES ".center(40, "="))
print("=" * 40)
for name, scores in r2_results.items():
    print(f" Model: {name:<6} | Mean R2: {scores['mean_r2']:.4f}")
print("=" * 40)

# -----------------------------
# Plot and save Jacobians
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
model_names = ["CEBRA", "ACORN"]
ims = []

for ax, name in zip(axes, model_names):
    result = results[name]
    jf = abs(result["jf"]).mean(0)

    im = ax.matshow(
        jf / jf.sum(),
        aspect="auto",
        cmap="cividis",
    )
    ims.append(im)
    ax.set_title(f"{name}\nR2={r2_results[name]['mean_r2']:.3f}")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(ims[0], cax=cbar_ax)

safe_target = target_file.replace(".mat.npz", "").replace(".", "_")
fig_path = os.path.join(img_dir, f"{safe_target}_CEBRA_vs_ACORN.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved figure to:", fig_path)
print("Decoder scores:", r2_results)

# import os
# import sys
# import numpy as np
# import torch
# import matplotlib.pyplot as plt

# from utils.constants import CEBRA_DIR, DATA_DIR
# from utils.dataset_loader import DatasetLoader

# sys.path.insert(0, str(CEBRA_DIR))
# import cebra
# from cebra import CEBRA


# # -----------------------------
# # Config
# # -----------------------------
# # dataset_name = "Jango_ISO_2015_npz"
# # target_file = "Jango_20150730_001.mat.npz"

# dataset_name = "Mihili_CO_2014_npz"
# target_file = "Mihili_20140203_001.mat.npz"

# out_dir = "outputs"
# img_dir = "images"

# os.makedirs(out_dir, exist_ok=True)
# os.makedirs(img_dir, exist_ok=True)

# os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

# loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")

# adv_ep = 5

# # -----------------------------
# # Find the exact day index for the file
# # -----------------------------
# dataset_dir = os.path.join(DATA_DIR, dataset_name)
# files = sorted(os.listdir(dataset_dir))
# day_idx = files.index(target_file)
# print("Selected day index:", day_idx)
# print("Selected file:", files[day_idx])


# # -----------------------------
# # Load one day
# # -----------------------------
# x_np, y_np = loader.load_dataset_day(day_idx, dataset_name, cache=True)

# print("x shape:", x_np.shape)
# print("y shape:", y_np.shape)

# if y_np.ndim > 1 and y_np.shape[1] >= 2:
#     y_np = y_np[:, :2]
# else:
#     y_np = y_np.reshape(-1, 1)

# split_idx = int(0.8 * len(x_np))
# train_data = x_np[:split_idx]
# valid_data = x_np[split_idx:]

# train_continuous_label = y_np[:split_idx]
# valid_continuous_label = y_np[split_idx:]


# # -----------------------------
# # Train + Jacobian for clean/adv
# # -----------------------------
# results = {}

# for adv in [False, True]:
#     model_name = "ACORN" if adv else "CEBRA"

#     model = CEBRA(
#         batch_size=1024,
#         temperature=0.4,
#         model_architecture="offset36-model-more-dropout",
#         time_offsets=4,
#         max_iterations=2500,
#         output_dimension=48,
#         verbose=True,
#         training_mode="adversarial" if adv else "clean",
#         adv_alpha=adv_ep / 5,
#         adv_epsilon=adv_ep,
#         adv_steps=10,
#         attack_norm="l2",
#     )

#     model.fit(train_data, train_continuous_label)

#     save_path = os.path.join(out_dir, f"{model_name}_{target_file}.pth")
#     model.save(save_path)
#     print("Saved model to:", save_path)

#     # load trained model for attribution
#     loaded = CEBRA.load(save_path, weights_only=False)
#     trained_model = loaded.solver_.model.to("cuda")

#     method = cebra.attribution.init(
#         name="jacobian-based",
#         model=trained_model,
#         input_data=torch.from_numpy(train_data).float().requires_grad_(True),
#         output_dimension=trained_model.num_output,
#     )

#     result = method.compute_attribution_map()
#     results[model_name] = result


# # -----------------------------
# # Plot and save in images/
# # -----------------------------
# fig, axes = plt.subplots(1, 2, figsize=(18,8))

# model_names = ["CEBRA", "ACORN"]
# ims = []

# for ax, name in zip(axes, model_names):
#     result = results[name]
#     jf = abs(result["jf"]).mean(0)

#     im = ax.matshow(
#         jf / jf.sum(),
#         aspect="auto",
#         cmap="cividis"
#     )

#     ims.append(im)
#     ax.set_title(name)

# fig.subplots_adjust(right=0.88)

# cbar_ax = fig.add_axes([0.90,0.15,0.02,0.7])
# fig.colorbar(ims[0], cax=cbar_ax)

# plt.savefig(
#     os.path.join(
#         img_dir,
#         "Chewie_20160927_CEBRA_vs_ACORN.png"
#     ),
#     dpi=300,
#     bbox_inches="tight"
# )

# plt.show()
