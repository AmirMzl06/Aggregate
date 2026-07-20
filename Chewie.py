import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.constants import CEBRA_DIR, DATA_DIR
from utils.dataset_loader import DatasetLoader

sys.path.insert(0, str(CEBRA_DIR))
import cebra
from cebra import CEBRA


# -----------------------------
# Config
# -----------------------------
# dataset_name = "Jango_ISO_2015_npz"
# target_file = "Jango_20150730_001.mat.npz"

dataset_name = "Mihili_CO_2014_npz"
target_file = "Mihili_20140203_001.mat.npz"

out_dir = "outputs"
img_dir = "images"

os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

os.environ["CEBRA_DATADIR"] = os.path.abspath(DATA_DIR)

loader = DatasetLoader(data_root_dir=DATA_DIR, cache_dir="./weights_cache/")


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

if y_np.ndim > 1 and y_np.shape[1] >= 2:
    y_np = y_np[:, :2]
else:
    y_np = y_np.reshape(-1, 1)

split_idx = int(0.8 * len(x_np))
train_data = x_np[:split_idx]
valid_data = x_np[split_idx:]

train_continuous_label = y_np[:split_idx]
valid_continuous_label = y_np[split_idx:]


# -----------------------------
# Train + Jacobian for clean/adv
# -----------------------------
results = {}

for adv in [False, True]:
    model_name = "ACORN" if adv else "CEBRA"

    model = CEBRA(
        batch_size=1024,
        temperature=0.4,
        model_architecture="offset36-model-more-dropout",
        time_offsets=4,
        max_iterations=3000,
        output_dimension=48,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=0.5 / 5,
        adv_epsilon=0.5,
        adv_steps=10,
        attack_norm="l2",
    )

    model.fit(train_data, train_continuous_label)

    save_path = os.path.join(out_dir, f"{model_name}_{target_file}.pth")
    model.save(save_path)
    print("Saved model to:", save_path)

    # load trained model for attribution
    loaded = CEBRA.load(save_path, weights_only=False)
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
# Plot and save in images/
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18,8))

model_names = ["CEBRA", "ACORN"]
ims = []

for ax, name in zip(axes, model_names):
    result = results[name]
    jf = abs(result["jf"]).mean(0)

    im = ax.matshow(
        jf / jf.sum(),
        aspect="auto",
        cmap="cividis"
    )

    ims.append(im)
    ax.set_title(name)

fig.subplots_adjust(right=0.88)

cbar_ax = fig.add_axes([0.90,0.15,0.02,0.7])
fig.colorbar(ims[0], cax=cbar_ax)

plt.savefig(
    os.path.join(
        img_dir,
        "Chewie_20160927_CEBRA_vs_ACORN.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
