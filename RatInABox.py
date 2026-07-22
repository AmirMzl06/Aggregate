import os
import gc
import pickle
import itertools
from pathlib import Path

import requests
import numpy as np
import torch
import matplotlib.pyplot as plt

from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
import cebra

from utils.min_distance import min_l2_distance


# =========================================================
# Config
# =========================================================
DATA_URL = (
    "https://zenodo.org/records/15267195/files/"
    "cynthi_neurons90_gridbase0.5_gridmodules3_grid_head_direction_place_speed_duration2000_noise0.25_bs100_seed231209234.p?download=1"
)
DATA_FILE = "cynthi_neurons90.p"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

NUM_STEPS = 1000
BATCH_SIZE = 2500
N_LATENTS = 14

# Notebook settings
BEHAVIOR_INDICES = (0, 4)
TIME_INDICES = (0, 14)

# Clean xCEBRA regularization
REG_START = 0.0
REG_END = 0.1
REG_SWITCH_ON = NUM_STEPS // 4
REG_SWITCH_OFF = NUM_STEPS // 2

OUT_DIR = Path("outputs_ratinabox")
IMG_DIR = Path("images_ratinabox")
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)


# =========================================================
# Utilities
# =========================================================
def download_dataset_if_needed():
    if Path(DATA_FILE).exists():
        return
    print(f"Downloading {DATA_FILE} ...")
    response = requests.get(DATA_URL, timeout=300)
    response.raise_for_status()
    Path(DATA_FILE).write_bytes(response.content)
    print("Download finished.")


def load_dataset():
    with open(DATA_FILE, "rb") as f:
        dataset = pickle.load(f)

    neural = torch.tensor(dataset["spikes"]).float()
    position = torch.tensor(dataset["position"]).float()
    return dataset, neural, position


def build_ground_truth_attribution(num_neurons: int):
    cells = np.array(
        list(
            itertools.chain.from_iterable(
                [
                    ["position"] * 100,
                    ["hd"] * 100,
                    ["position"] * 100,
                    ["grid"] * 60,
                ]
            )
        )
    )

    latents = [
        (["position", "grid"], 3),
        (["speed"], 11),
    ]
    latents = [group for group, repeats in latents for _ in range(repeats)]

    ground_truth = np.zeros((len(latents), len(cells)), dtype=bool)
    for i, latent in enumerate(latents):
        for j, cell_type in enumerate(cells):
            ground_truth[i, j] = cell_type in latent

    if num_neurons != len(cells):
        print(
            f"Warning: notebook GT has {len(cells)} neurons but data has {num_neurons}. "
            "Padding/truncating GT to match."
        )
        if num_neurons > len(cells):
            pad = num_neurons - len(cells)
            cells = np.concatenate([cells, np.array(["unknown"] * pad)])
            ground_truth = np.pad(
                ground_truth,
                ((0, 0), (0, pad)),
                mode="constant",
                constant_values=False,
            )
        else:
            cells = cells[:num_neurons]
            ground_truth = ground_truth[:, :num_neurons]

    return cells, latents, ground_truth


def cleanup(*objs):
    for obj in objs:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_dataset(neural: torch.Tensor, position: torch.Tensor):
    return DatasetxCEBRA(neural, position=position)


def train_model(dataset, mode: str):
    loader = ContrastiveMultiObjectiveLoader(
        dataset=dataset,
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
    ).to(DEVICE)

    config = MultiObjectiveConfig(loader)

    config.set_slice(*BEHAVIOR_INDICES)
    config.set_loss("FixedCosineInfoNCE", temperature=1.0)
    config.set_distribution("time_delta", time_delta=1, label_name="position")
    config.push()

    config.set_slice(*TIME_INDICES)
    config.set_loss("FixedCosineInfoNCE", temperature=1.0)
    config.set_distribution("time", time_offset=10)
    config.push()

    config.finalize()

    criterion = config.criterion
    feature_ranges = config.feature_ranges

    neural_model = cebra.models.init(
        name="offset10-model",
        num_neurons=dataset.neural.shape[1],
        num_units=256,
        num_output=N_LATENTS,
    ).to(DEVICE)

    dataset.configure_for(neural_model)

    opt = torch.optim.Adam(
        list(neural_model.parameters()) + list(criterion.parameters()),
        lr=3e-4,
        weight_decay=0.0,
    )

    regularizer = cebra.models.jacobian_regularizer.JacobianReg()

    solver = cebra.solver.init(
        name="multiobjective-solver",
        model=neural_model,
        feature_ranges=feature_ranges,
        regularizer=regularizer,
        renormalize=True,
        use_sam=False,
        criterion=criterion,
        optimizer=opt,
        tqdm_on=True,
    ).to(DEVICE)

    solver.training_mode = mode

    # =========================================================
    # setup like previous runs
    # =========================================================
    adv_epsilon = float(min_l2_distance(dataset.neural)) / 2.0
    adv_epsilon = max(adv_epsilon, 1e-6)

    solver.adv_epsilon = adv_epsilon
    solver.adv_alpha = adv_epsilon / 5
    solver.adv_steps = 10
    solver.attack_norm = "linf"

    weight_scheduler = LinearRampUp(
        n_splits=2,
        step_to_switch_on_reg=REG_SWITCH_ON,
        step_to_switch_off_reg=REG_SWITCH_OFF,
        start_weight=REG_START,
        end_weight=REG_END,
    )

    solver.fit(
        loader=loader,
        valid_loader=None,
        log_frequency=None,
        scheduler_regularizer=weight_scheduler,
        scheduler_loss=None,
    )

    return solver


def compute_attribution_and_auc(model, neural: torch.Tensor, ground_truth: np.ndarray):
    model = model.to(DEVICE)
    if hasattr(model, "split_outputs"):
        model.split_outputs = False

    neural = neural.clone().detach().to(DEVICE)
    neural.requires_grad_(True)

    method = cebra.attribution.init(
        name="jacobian-based",
        model=model,
        input_data=neural,
        output_dimension=model.num_output,
    )

    result = method.compute_attribution_map()

    jf = torch.abs(result["jf"]).mean(0)
    jfinv = torch.abs(result["jf-inv-svd"]).mean(0)
    jfconvabsinv = torch.abs(result["jf-convabs-inv-svd"]).mean(0)

    auc_jf = method.compute_attribution_score(jf.cpu().numpy(), ground_truth)
    auc_jfinv = method.compute_attribution_score(jfinv.cpu().numpy(), ground_truth)
    auc_jfconvabsinv = method.compute_attribution_score(
        jfconvabsinv.cpu().numpy(),
        ground_truth,
    )

    return {
        "result": result,
        "jf": jf.cpu().numpy(),
        "jfinv": jfinv.cpu().numpy(),
        "jfconvabsinv": jfconvabsinv.cpu().numpy(),
        "auc": {
            "jf": auc_jf,
            "jfinv": auc_jfinv,
            "jfconvabsinv": auc_jfconvabsinv,
        },
    }


def save_heatmaps(name: str, maps: dict, aucs: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = [
        f"{name} | jf\nAUC={aucs['jf']:.2f}",
        f"{name} | jf-inv-svd\nAUC={aucs['jfinv']:.2f}",
        f"{name} | jf-convabs-inv-svd\nAUC={aucs['jfconvabsinv']:.2f}",
    ]

    for ax, key, title in zip(axes, ["jf", "jfinv", "jfconvabsinv"], titles):
        im = ax.matshow(np.abs(maps[key]), aspect="auto", cmap="cividis")
        ax.set_title(title)
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Latents")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    path = IMG_DIR / f"{name}_attribution_maps.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {path}")


# =========================================================
# Main
# =========================================================
download_dataset_if_needed()
dataset, neural, position = load_dataset()

print("Dataset keys:", list(dataset.keys()))
print("neural shape:", tuple(neural.shape))
print("position shape:", tuple(position.shape))

_, _, ground_truth_attribution = build_ground_truth_attribution(neural.shape[1])
print("ground_truth_attribution shape:", ground_truth_attribution.shape)

# Train clean model
clean_dataset = make_dataset(neural, position)
clean_solver = train_model(clean_dataset, mode="clean")
clean_model = clean_solver.model.to(DEVICE)

# Train adversarial model
adv_dataset = make_dataset(neural, position)
adv_solver = train_model(adv_dataset, mode="adversarial")
adv_model = adv_solver.model.to(DEVICE)

# Attribution + AUROC
print("\nComputing attribution maps...")

clean_pack = compute_attribution_and_auc(clean_model, neural, ground_truth_attribution)
adv_pack = compute_attribution_and_auc(adv_model, neural, ground_truth_attribution)

print("\n==============================")
print(" AUROC SUMMARY ")
print("==============================")
for method_name in ["jf", "jfinv", "jfconvabsinv"]:
    print(
        f"{method_name:>20} | "
        f"clean={clean_pack['auc'][method_name]:.2f} | "
        f"adv={adv_pack['auc'][method_name]:.2f}"
    )

# Save heatmaps
save_heatmaps("clean", clean_pack, clean_pack["auc"])
save_heatmaps("adv", adv_pack, adv_pack["auc"])

# Save AUROC bar plot
methods = ["jf", "jfinv", "jfconvabsinv"]
x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - width / 2, [clean_pack["auc"][m] for m in methods], width, label="clean")
ax.bar(x + width / 2, [adv_pack["auc"][m] for m in methods], width, label="adv")

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("ROC AUC")
ax.set_title("RatInABox AUROC: clean vs adv")
ax.legend()

auc_path = IMG_DIR / "ratinabox_auc_clean_vs_adv.png"
plt.savefig(auc_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved AUROC plot: {auc_path}")

cleanup(clean_solver, adv_solver, clean_model, adv_model, clean_pack, adv_pack)
print("Done.")
