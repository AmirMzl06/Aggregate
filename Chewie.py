import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from utils.constants import CEBRA_DIR, DATA_DIR
from utils.dataset_loader import DatasetLoader
from utils.cebra_decoder import TwoLayerMLP

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA


# -----------------------------
# Config
# -----------------------------
dataset_name = "Mihili_CO_2014_npz"
target_file = "Mihili_20140203_001.mat.npz"

out_dir = "outputs"
img_dir = "images"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")
adv_ep = 5


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
    """
    Train TwoLayerMLP on embeddings from a fitted CEBRA model,
    using the same kind of early stopping / retrain logic as the repo.
    """
    # split TRAIN into train/val (same style as the repo)
    neural_train, neural_val, label_train, label_val = train_test_split(
        train_x_np,
        train_y_np,
        test_size=0.125,
        random_state=42,
        shuffle=False,
    )

    # embed
    z_train = torch.from_numpy(to_numpy(cebra_model.transform(torch.from_numpy(neural_train).float()))).float().to(device)
    z_val = torch.from_numpy(to_numpy(cebra_model.transform(torch.from_numpy(neural_val).float()))).float().to(device)
    z_test = torch.from_numpy(to_numpy(cebra_model.transform(torch.from_numpy(test_x_np).float()))).float().to(device)

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
    best_state = copy.deepcopy(decoder.state_dict())
    patience = 1000
    bad = 0
    min_epochs = 4000

    # Phase 1: early stopping on val
    for epoch in range(decoder_iters):
        decoder.train()
        optimizer.zero_grad()
        pred = decoder(z_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        decoder.eval()
        with torch.no_grad():
            val_pred = decoder(z_val).detach().cpu().numpy()
            val_true = y_val.detach().cpu().numpy()

        val_r2, _ = mean_r2_score(val_true, val_pred)

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch + 1
            bad = 0
            best_state = copy.deepcopy(decoder.state_dict())
        else:
            if epoch > min_epochs - patience:
                bad += 1
            if bad >= patience:
                print(f"Decoder early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 1000 == 0:
            print(f"Decoder epoch [{epoch + 1}/{decoder_iters}] | loss={loss.item():.4f} | val_r2={val_r2:.4f}")

    # restore best (optional)
    decoder.load_state_dict(best_state)

    # Phase 2: retrain on train+val for best_epoch
    z_full = torch.cat([z_train, z_val], dim=0)
    y_full = torch.cat([y_train, y_val], dim=0)

    decoder.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=2e-4)

    for _ in range(best_epoch):
        decoder.train()
        optimizer.zero_grad()
        pred = decoder(z_full)
        loss = criterion(pred, y_full)
        loss.backward()
        optimizer.step()

    # Final test R2
    decoder.eval()
    with torch.no_grad():
        test_pred = decoder(z_test).detach().cpu().numpy()
        test_true = y_test.detach().cpu().numpy()

    mean_r2, per_dim_r2 = mean_r2_score(test_true, test_pred)
    return decoder, mean_r2, per_dim_r2


# -----------------------------
# Find exact file day
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

# CEBRA training labels: first 2 dims like before
if y_np.ndim > 1 and y_np.shape[1] >= 2:
    y_cebra = y_np[:, :2]
else:
    y_cebra = y_np.reshape(-1, 1)

split_idx = int(0.8 * len(x_np))
train_data = x_np[:split_idx]
valid_data = x_np[split_idx:]

train_continuous_label = y_cebra[:split_idx]
valid_continuous_label = y_cebra[split_idx:]


results = {}
decoder_scores = {}

# -----------------------------
# Train clean / adv
# -----------------------------
for adv in [False, True]:
    model_name = "ACORN" if adv else "CEBRA"

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

    loaded = CEBRA.load(save_path, weights_only=False)

    # -----------------------------
    # Decoder R2 with same TwoLayerMLP architecture
    # target = full y_np (all dimensions)
    # -----------------------------
    decoder, mean_r2, per_dim_r2 = train_decoder_with_same_arch(
        cebra_model=loaded,
        train_x_np=train_data,
        train_y_np=y_np[:split_idx],
        test_x_np=valid_data,
        test_y_np=y_np[split_idx:],
        input_dim=48,
        hidden_dim=64,
        dropout_rate=0.4,
        decoder_iters=10000,
    )

    decoder_scores[model_name] = {
        "mean_r2": mean_r2,
        "per_dim_r2": per_dim_r2,
    }

    torch.save(
        decoder.state_dict(),
        os.path.join(out_dir, f"decoder_{model_name}_{target_file}.pth")
    )

    print(f"{model_name} decoder mean R2: {mean_r2:.4f}")
    print(f"{model_name} decoder per-dim R2: {[round(v, 4) for v in per_dim_r2]}")

    # -----------------------------
    # Jacobian attribution
    # -----------------------------
    trained_model = loaded.solver_.model.to("cuda")
    method = cebra.attribution.init(
        name="jacobian-based",
        model=trained_model,
        input_data=torch.from_numpy(train_data).float().requires_grad_(True),
        output_dimension=trained_model.num_output,
    )
    result = method.compute_attribution_map()
    results[model_name] = result


# -----------------------------
# Plot Jacobians
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
    ax.set_title(f"{name}\nR2={decoder_scores[name]['mean_r2']:.3f}")

fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
fig.colorbar(ims[0], cax=cbar_ax)

fig_path = os.path.join(img_dir, f"{target_file.replace('.mat.npz', '')}_CEBRA_vs_ACORN.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved figure to:", fig_path)
print("Decoder scores:", decoder_scores)



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
