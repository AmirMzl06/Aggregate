import os
import sys
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from utils.constants import CEBRA_DIR, DATA_DIR
from utils.dataset_loader import DatasetLoader
from utils.cebra_decoder import TwoLayerMLP

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

adv_ep = 5

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
# Train + Jacobian + Decoder for clean/adv
# -----------------------------
results = {}
r2_results = {}  # برای ذخیره نتایج R2 هر مدل

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for adv in [False, True]:
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

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    print(f"\n--- Training Decoder for {model_name} ---")
    
    z_train = model.transform(train_data)
    z_valid = model.transform(valid_data)
    
    X_train_t = torch.from_numpy(z_train).float().to(device)
    Y_train_t = torch.from_numpy(train_continuous_label).float().to(device)
    X_valid_t = torch.from_numpy(z_valid).float().to(device)
    Y_valid_t = torch.from_numpy(valid_continuous_label).float().to(device)
    
    decoder = TwoLayerMLP(
        input_dim=model.output_dimension, 
        hidden_dim=64, 
        output_dim=train_continuous_label.shape[1],
        dropout_rate=0.4
    ).to(device)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001, weight_decay=2e-4)
    
    best_r2 = -1e9
    patience = 1000
    bad_epochs = 0
    min_epochs = 4000
    best_decoder_state = None
    
    for epoch in range(10000):
        decoder.train()
        optimizer.zero_grad()
        outputs = decoder(X_train_t)
        loss = criterion(outputs, Y_train_t)
        loss.backward()
        optimizer.step()
        
        decoder.eval()
        with torch.no_grad():
            val_preds = decoder(X_valid_t).cpu().numpy()
            val_true = Y_valid_t.cpu().numpy()
            
            r2_scores = [r2_score(val_true[:, i], val_preds[:, i]) for i in range(val_true.shape[1])]
            current_r2 = sum(r2_scores) / len(r2_scores)
            
        if current_r2 > best_r2:
            best_r2 = current_r2
            bad_epochs = 0
            best_decoder_state = copy.deepcopy(decoder.state_dict())
        else:
            if epoch > min_epochs - patience:
                bad_epochs += 1
                
        if bad_epochs >= patience:
            print(f"Early stopping decoder at epoch {epoch + 1}")
            break
            
        if (epoch + 1) % 2000 == 0:
            print(f"Epoch [{epoch + 1}/10000], Loss: {loss.item():.4f}, Val R2: {current_r2:.4f}")
            
    if best_decoder_state is not None:
        decoder.load_state_dict(best_decoder_state)
    
    print(f"** Final R2 Score for {model_name}: {best_r2:.4f} **\n")
    r2_results[model_name] = best_r2

# -----------------------------
# Print R2 Comparison Summary
# -----------------------------
print("\n" + "="*40)
print(" SUMMARY OF R2 SCORES ".center(40, "="))
print("="*40)
for name, score in r2_results.items():
    print(f" Model: {name:<6} | R2 Score: {score:.4f}")
print("="*40)

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
