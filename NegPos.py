"""
Train ONLY one adversarial CEBRA model on Perich C-CO12,
then train ONLY one TwoLayerMLP decoder and report ONLY R^2.

No clean model.
No NegPos model.
No plots.
No PCA.
No extra metrics.
No saved embeddings.

CEBRA settings:
    epsilon = 0.3
    alpha   = epsilon / 5 = 0.06
    steps   = 10
    norm    = linf
    offset  = 1

Decoder:
    TwoLayerMLP
    2500 epochs
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# =====================================================================
# ORIGINAL / MAIN CEBRA FORK
# =====================================================================

from utils.constants import CEBRA_DIR

# Make sure another CEBRA fork is not cached.
for _m in list(sys.modules):
    if _m == "cebra" or _m.startswith("cebra."):
        del sys.modules[_m]

sys.path.insert(0, str(CEBRA_DIR))

import cebra
from cebra import CEBRA

print("Using CEBRA:", cebra.__file__)


# =====================================================================
# CONFIG
# =====================================================================

PERICH_DATA_DIR = "/data/hossein/mm_project/perich_data_valid_final_raw/"
SESSION = "C-CO12"
NPZ_PATH = os.path.join(PERICH_DATA_DIR, f"{SESSION}.npz")

SEED = 42

# CEBRA
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

OFFSET = 1

ADV_EPSILON = 0.5
ADV_ALPHA = ADV_EPSILON / 5.0   # 0.06
ADV_STEPS = 10
ATTACK_NORM = "linf"

# Decoder
DECODER_HIDDEN_DIM = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 1e-4
DECODER_EPOCHS = 2500
DECODER_BATCH_SIZE = 256

TORCH_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Optional: save only the trained models.
OUT_DIR = "ADV_CEBRA_CCO12_R2"
os.makedirs(OUT_DIR, exist_ok=True)

CEBRA_MODEL_PATH = os.path.join(OUT_DIR, "adversarial_cebra.pt")
DECODER_MODEL_PATH = os.path.join(OUT_DIR, "decoder.pt")


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# DATA
# =====================================================================

def load_data():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(NPZ_PATH)

    data = np.load(NPZ_PATH, allow_pickle=True)

    X_train = data["train_data"].astype(np.float32)
    X_test = data["valid_data"].astype(np.float32)

    Y_train = data["train_label"].astype(np.float32)
    Y_test = data["valid_label"].astype(np.float32)

    for name, arr in (
        ("X_train", X_train),
        ("X_test", X_test),
        ("Y_train", Y_train),
        ("Y_test", Y_test),
    ):
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{name} contains NaN or Inf.")

    return X_train, X_test, Y_train, Y_test


# =====================================================================
# ADVERSARIAL CEBRA
# =====================================================================

def build_adv_cebra():
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        model_architecture=MODEL_ARCH,
        time_offsets=OFFSET,
        max_iterations=MAX_ITER,
        output_dimension=LATENT_DIM,
        num_hidden_units=HIDDEN,

        training_mode="adversarial",
        adv_epsilon=ADV_EPSILON,
        adv_alpha=ADV_ALPHA,
        adv_steps=ADV_STEPS,
        attack_norm=ATTACK_NORM,

        device=DEVICE,
        verbose=True,
    )


# =====================================================================
# DECODER
# =====================================================================

class TwoLayerMLP(nn.Module):

    def __init__(
        self,
        input_dim=32,
        hidden_dim=64,
        output_dim=2,
        dropout_rate=0.4,
    ):
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
                nn.init.kaiming_normal_(
                    layer.weight,
                    nonlinearity="relu",
                )

                if layer.bias is not None:
                    nn.init.constant_(
                        layer.bias,
                        0,
                    )

    def forward(self, x):
        return self.net(x)


def align_embedding_and_labels(embedding, labels):
    """
    Normally model.transform keeps the same number of samples because
    CEBRA pads before transform. If lengths differ, crop both equally.
    """
    n = min(len(embedding), len(labels))
    return embedding[:n], labels[:n]


def train_decoder(train_emb, Y_train):
    train_emb, Y_train = align_embedding_and_labels(
        train_emb,
        Y_train,
    )

    if Y_train.ndim == 1:
        Y_train = Y_train[:, None]

    seed_all(SEED)

    decoder = TwoLayerMLP(
        input_dim=train_emb.shape[1],
        hidden_dim=DECODER_HIDDEN_DIM,
        output_dim=Y_train.shape[1],
        dropout_rate=DECODER_DROPOUT,
    ).to(TORCH_DEVICE)

    dataset = TensorDataset(
        torch.from_numpy(train_emb.astype(np.float32)),
        torch.from_numpy(Y_train.astype(np.float32)),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=DECODER_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=DECODER_LR,
        weight_decay=DECODER_WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    for epoch in range(1, DECODER_EPOCHS + 1):
        decoder.train()

        for xb, yb in loader:
            xb = xb.to(TORCH_DEVICE)
            yb = yb.to(TORCH_DEVICE)

            optimizer.zero_grad(set_to_none=True)

            pred = decoder(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

        # Minimal progress only.
        if epoch % 500 == 0 or epoch == 1 or epoch == DECODER_EPOCHS:
            print(f"Decoder epoch {epoch}/{DECODER_EPOCHS}")

    return decoder


# =====================================================================
# R2 ONLY
# =====================================================================

def compute_r2(decoder, test_emb, Y_test):
    test_emb, Y_test = align_embedding_and_labels(
        test_emb,
        Y_test,
    )

    if Y_test.ndim == 1:
        Y_test = Y_test[:, None]

    decoder.eval()

    with torch.no_grad():
        X = torch.from_numpy(
            test_emb.astype(np.float32)
        ).to(TORCH_DEVICE)

        pred = decoder(X).cpu().numpy()

    r2_each = r2_score(
        Y_test,
        pred,
        multioutput="raw_values",
    )

    mean_r2 = float(
        np.mean(r2_each)
    )

    return r2_each, mean_r2


# =====================================================================
# MAIN
# =====================================================================

def main():
    seed_all(SEED)

    X_train, X_test, Y_train, Y_test = load_data()

    # ---------------------------------------------------------------
    # 1) TRAIN ONLY ADVERSARIAL CEBRA
    # ---------------------------------------------------------------
    print("\nTraining adversarial CEBRA...")
    print(
        f"epsilon={ADV_EPSILON}, "
        f"alpha={ADV_ALPHA}, "
        f"steps={ADV_STEPS}, "
        f"norm={ATTACK_NORM}, "
        f"offset={OFFSET}"
    )

    model = build_adv_cebra()
    model.fit(X_train)

    model.save(CEBRA_MODEL_PATH)

    # ---------------------------------------------------------------
    # 2) EMBED TRAIN / TEST
    # ---------------------------------------------------------------
    train_emb = np.asarray(
        model.transform(X_train),
        dtype=np.float32,
    )

    test_emb = np.asarray(
        model.transform(X_test),
        dtype=np.float32,
    )

    # ---------------------------------------------------------------
    # 3) TRAIN ONLY ONE DECODER
    # ---------------------------------------------------------------
    print("\nTraining decoder...")

    decoder = train_decoder(
        train_emb,
        Y_train,
    )

    torch.save(
        decoder.state_dict(),
        DECODER_MODEL_PATH,
    )

    # ---------------------------------------------------------------
    # 4) COMPUTE ONLY R2
    # ---------------------------------------------------------------
    r2_each, mean_r2 = compute_r2(
        decoder,
        test_emb,
        Y_test,
    )

    print("\n" + "=" * 70)
    print("FINAL R2")
    print("=" * 70)

    for i, value in enumerate(r2_each):
        print(f"target {i}: R2 = {value:.6f}")

    print(f"\nMean R2 = {mean_r2:.6f}")


if __name__ == "__main__":
    main()



# """
# Decoder comparison for already-trained C-CO12 CEBRA models.

# Models:
#   1) Normal / vanilla CEBRA
#   2) CEBRA-NegPos

# No CEBRA retraining is done here.

# Pipeline:
#   C-CO12 train neural -> encoder -> train embedding -> train TwoLayerMLP
#   C-CO12 test  neural -> encoder -> test embedding  -> evaluate R^2

# The two decoders:
#   - have exactly the same architecture
#   - use the same random seed / initialization
#   - train for 2500 epochs
#   - are evaluated on exactly the same C-CO12 test labels
# """

# from __future__ import annotations

# import csv
# import random
# import sys
# from pathlib import Path

# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.metrics import r2_score

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt


# # =====================================================================
# # CONFIG
# # =====================================================================

# ROOT = Path(__file__).resolve().parent

# # Modified CEBRA fork
# CEBRA_DIR = ROOT / "CEBRA-NegPos"

# # Perich
# PERICH_DATA_DIR = Path(
#     "/data/hossein/mm_project/perich_data_valid_final_raw/"
# )
# TARGET_SESSION = "C-CO12"
# NPZ_PATH = PERICH_DATA_DIR / f"{TARGET_SESSION}.npz"

# # Models saved by the previous script
# EXPERIMENT_DIR = ROOT / "CEBRA_CCO12_NegPos_Test"
# VANILLA_MODEL_PATH = EXPERIMENT_DIR / "models" / "cebra_vanilla.pt"
# NEGPOS_MODEL_PATH = EXPERIMENT_DIR / "models" / "cebra_negpos.pt"

# # Decoder output
# DECODER_OUT = EXPERIMENT_DIR / "decoder"
# DECODER_OUT.mkdir(parents=True, exist_ok=True)

# SEED = 42

# DECODER_HIDDEN_DIM = 64
# DECODER_DROPOUT = 0.4
# DECODER_EPOCHS = 2500
# DECODER_BATCH_SIZE = 256
# DECODER_LR = 1e-3
# DECODER_WEIGHT_DECAY = 1e-4
# PRINT_EVERY = 250

# TORCH_DEVICE = torch.device(
#     "cuda" if torch.cuda.is_available() else "cpu"
# )


# # =====================================================================
# # IMPORT THE CORRECT CEBRA FORK
# # =====================================================================

# if not CEBRA_DIR.exists():
#     raise FileNotFoundError(
#         f"CEBRA-NegPos not found: {CEBRA_DIR}"
#     )

# for _m in list(sys.modules):
#     if _m == "cebra" or _m.startswith("cebra."):
#         del sys.modules[_m]

# sys.path.insert(0, str(CEBRA_DIR))

# import cebra
# from cebra import CEBRA

# print("\nUsing CEBRA:")
# print(cebra.__file__)


# # =====================================================================
# # REPRODUCIBILITY
# # =====================================================================

# def seed_all(seed=SEED):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# # =====================================================================
# # DATA
# # =====================================================================

# def load_cco12():
#     if not NPZ_PATH.exists():
#         raise FileNotFoundError(NPZ_PATH)

#     data = np.load(NPZ_PATH, allow_pickle=True)

#     X_train = np.asarray(
#         data["train_data"],
#         dtype=np.float32,
#     )
#     X_test = np.asarray(
#         data["valid_data"],
#         dtype=np.float32,
#     )

#     Y_train = np.asarray(
#         data["train_label"],
#         dtype=np.float32,
#     )
#     Y_test = np.asarray(
#         data["valid_label"],
#         dtype=np.float32,
#     )

#     print("\nC-CO12:")
#     print("X_train:", X_train.shape)
#     print("X_test :", X_test.shape)
#     print("Y_train:", Y_train.shape)
#     print("Y_test :", Y_test.shape)

#     for name, arr in (
#         ("X_train", X_train),
#         ("X_test", X_test),
#         ("Y_train", Y_train),
#         ("Y_test", Y_test),
#     ):
#         if not np.isfinite(arr).all():
#             raise RuntimeError(
#                 f"{name} contains NaN/Inf."
#             )

#     return X_train, X_test, Y_train, Y_test


# # =====================================================================
# # DECODER
# # =====================================================================

# class TwoLayerMLP(nn.Module):

#     def __init__(
#         self,
#         input_dim=32,
#         hidden_dim=64,
#         output_dim=2,
#         dropout_rate=0.4,
#     ):
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
#                 nn.init.kaiming_normal_(
#                     layer.weight,
#                     nonlinearity="relu",
#                 )
#                 if layer.bias is not None:
#                     nn.init.constant_(
#                         layer.bias,
#                         0,
#                     )

#     def forward(self, x):
#         return self.net(x)


# # =====================================================================
# # EMBEDDINGS
# # =====================================================================

# def load_models():
#     if not VANILLA_MODEL_PATH.exists():
#         raise FileNotFoundError(
#             VANILLA_MODEL_PATH
#         )

#     if not NEGPOS_MODEL_PATH.exists():
#         raise FileNotFoundError(
#             NEGPOS_MODEL_PATH
#         )

#     vanilla = CEBRA.load(
#         VANILLA_MODEL_PATH
#     )
#     negpos = CEBRA.load(
#         NEGPOS_MODEL_PATH
#     )

#     print("\nLoaded:")
#     print("normal CEBRA:", VANILLA_MODEL_PATH)
#     print("NegPos CEBRA:", NEGPOS_MODEL_PATH)

#     return vanilla, negpos


# def get_embeddings(
#     model,
#     X_train,
#     X_test,
# ):
#     train_emb = np.asarray(
#         model.transform(
#             X_train.astype(np.float32)
#         ),
#         dtype=np.float32,
#     )

#     test_emb = np.asarray(
#         model.transform(
#             X_test.astype(np.float32)
#         ),
#         dtype=np.float32,
#     )

#     return train_emb, test_emb


# def align_embedding_and_labels(
#     embedding,
#     labels,
#     split_name,
# ):
#     """
#     With padded CEBRA transform these lengths should normally already match.

#     If a model configuration produces a small offset-induced length
#     difference, crop both arrays to the same minimum length instead of
#     silently misaligning them.
#     """
#     if len(embedding) == len(labels):
#         return embedding, labels

#     n = min(
#         len(embedding),
#         len(labels),
#     )

#     print(
#         f"[WARN] {split_name}: "
#         f"embedding length={len(embedding)}, "
#         f"label length={len(labels)}. "
#         f"Cropping both to {n}."
#     )

#     return embedding[:n], labels[:n]


# # =====================================================================
# # TRAIN ONE DECODER
# # =====================================================================

# def train_decoder(
#     model_name,
#     train_emb,
#     test_emb,
#     Y_train,
#     Y_test,
# ):
#     print("\n" + "=" * 100)
#     print(f"DECODER -- {model_name}")
#     print("=" * 100)

#     train_emb, Y_train_aligned = align_embedding_and_labels(
#         train_emb,
#         Y_train,
#         f"{model_name} train",
#     )

#     test_emb, Y_test_aligned = align_embedding_and_labels(
#         test_emb,
#         Y_test,
#         f"{model_name} test",
#     )

#     input_dim = train_emb.shape[1]

#     if Y_train_aligned.ndim == 1:
#         Y_train_aligned = Y_train_aligned[:, None]
#         Y_test_aligned = Y_test_aligned[:, None]

#     output_dim = Y_train_aligned.shape[1]

#     print("decoder input_dim :", input_dim)
#     print("decoder hidden_dim:", DECODER_HIDDEN_DIM)
#     print("decoder output_dim:", output_dim)
#     print("decoder epochs    :", DECODER_EPOCHS)
#     print("decoder device    :", TORCH_DEVICE)

#     # Same seed before every model -> same decoder initialization
#     # when input/output dimensions are the same.
#     seed_all(SEED)

#     decoder = TwoLayerMLP(
#         input_dim=input_dim,
#         hidden_dim=DECODER_HIDDEN_DIM,
#         output_dim=output_dim,
#         dropout_rate=DECODER_DROPOUT,
#     ).to(TORCH_DEVICE)

#     Xtr = torch.from_numpy(
#         train_emb.astype(np.float32)
#     )
#     Ytr = torch.from_numpy(
#         Y_train_aligned.astype(np.float32)
#     )

#     dataset = TensorDataset(
#         Xtr,
#         Ytr,
#     )

#     # Separate seeded generator makes batch shuffling reproducible too.
#     generator = torch.Generator()
#     generator.manual_seed(SEED)

#     loader = DataLoader(
#         dataset,
#         batch_size=DECODER_BATCH_SIZE,
#         shuffle=True,
#         num_workers=0,
#         drop_last=False,
#         generator=generator,
#     )

#     optimizer = torch.optim.Adam(
#         decoder.parameters(),
#         lr=DECODER_LR,
#         weight_decay=DECODER_WEIGHT_DECAY,
#     )

#     criterion = nn.MSELoss()

#     loss_history = []

#     for epoch in range(
#         1,
#         DECODER_EPOCHS + 1,
#     ):
#         decoder.train()

#         epoch_loss = 0.0
#         n_seen = 0

#         for xb, yb in loader:
#             xb = xb.to(
#                 TORCH_DEVICE,
#                 non_blocking=True,
#             )
#             yb = yb.to(
#                 TORCH_DEVICE,
#                 non_blocking=True,
#             )

#             optimizer.zero_grad(
#                 set_to_none=True
#             )

#             pred = decoder(xb)

#             loss = criterion(
#                 pred,
#                 yb,
#             )

#             loss.backward()
#             optimizer.step()

#             bs = xb.shape[0]

#             epoch_loss += (
#                 float(loss.item()) * bs
#             )
#             n_seen += bs

#         epoch_loss /= max(
#             n_seen,
#             1,
#         )

#         loss_history.append(
#             epoch_loss
#         )

#         if (
#             epoch == 1
#             or epoch % PRINT_EVERY == 0
#             or epoch == DECODER_EPOCHS
#         ):
#             print(
#                 f"{model_name} | "
#                 f"epoch {epoch:4d}/{DECODER_EPOCHS} | "
#                 f"train MSE={epoch_loss:.6f}"
#             )

#     # ---------------------------------------------------------
#     # TEST
#     # ---------------------------------------------------------
#     decoder.eval()

#     with torch.no_grad():
#         Xte = torch.from_numpy(
#             test_emb.astype(np.float32)
#         ).to(TORCH_DEVICE)

#         pred_test = (
#             decoder(Xte)
#             .cpu()
#             .numpy()
#         )

#     # Per-output R2
#     r2_each = np.asarray(
#         r2_score(
#             Y_test_aligned,
#             pred_test,
#             multioutput="raw_values",
#         )
#     )

#     # Simple mean across all target dimensions
#     mean_r2 = float(
#         np.mean(r2_each)
#     )

#     # Also report sklearn's variance-weighted aggregate
#     weighted_r2 = float(
#         r2_score(
#             Y_test_aligned,
#             pred_test,
#             multioutput="variance_weighted",
#         )
#     )

#     print("\nRESULT")
#     print("model:", model_name)
#     print("R2 per target:", r2_each)
#     print(f"Mean R2:       {mean_r2:.6f}")
#     print(f"Weighted R2:   {weighted_r2:.6f}")

#     # Save decoder weights
#     torch.save(
#         decoder.state_dict(),
#         DECODER_OUT / f"{model_name}_decoder.pt",
#     )

#     # Save predictions
#     np.save(
#         DECODER_OUT / f"{model_name}_test_predictions.npy",
#         pred_test.astype(np.float32),
#     )

#     # Save loss
#     np.save(
#         DECODER_OUT / f"{model_name}_train_loss.npy",
#         np.asarray(
#             loss_history,
#             dtype=np.float32,
#         ),
#     )

#     return {
#         "model_name": model_name,
#         "r2_each": r2_each,
#         "mean_r2": mean_r2,
#         "weighted_r2": weighted_r2,
#         "loss_history": np.asarray(
#             loss_history,
#             dtype=np.float32,
#         ),
#     }


# # =====================================================================
# # SAVE RESULTS
# # =====================================================================

# def save_results(results):
#     csv_path = DECODER_OUT / "decoder_r2_results.csv"

#     max_outputs = max(
#         len(r["r2_each"])
#         for r in results
#     )

#     fieldnames = [
#         "model",
#         "mean_r2",
#         "weighted_r2",
#     ] + [
#         f"r2_target_{i}"
#         for i in range(max_outputs)
#     ]

#     with csv_path.open(
#         "w",
#         newline="",
#     ) as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=fieldnames,
#         )

#         writer.writeheader()

#         for r in results:
#             row = {
#                 "model": r["model_name"],
#                 "mean_r2": r["mean_r2"],
#                 "weighted_r2": r["weighted_r2"],
#             }

#             for i, value in enumerate(
#                 r["r2_each"]
#             ):
#                 row[f"r2_target_{i}"] = float(
#                     value
#                 )

#             writer.writerow(row)

#     print("\nsaved:", csv_path)

#     # ---------------------------------------------------------
#     # Loss curves
#     # ---------------------------------------------------------
#     plt.figure(figsize=(8, 5))

#     for r in results:
#         plt.plot(
#             np.arange(
#                 1,
#                 len(r["loss_history"]) + 1,
#             ),
#             r["loss_history"],
#             label=r["model_name"],
#         )

#     plt.xlabel("Decoder epoch")
#     plt.ylabel("Train MSE")
#     plt.title("Decoder training loss")
#     plt.legend()
#     plt.tight_layout()

#     loss_plot = (
#         DECODER_OUT
#         / "decoder_training_loss.png"
#     )

#     plt.savefig(
#         loss_plot,
#         dpi=200,
#     )
#     plt.close()

#     print("saved:", loss_plot)

#     # ---------------------------------------------------------
#     # R2 comparison
#     # ---------------------------------------------------------
#     names = [
#         r["model_name"]
#         for r in results
#     ]

#     means = [
#         r["mean_r2"]
#         for r in results
#     ]

#     plt.figure(figsize=(6, 5))
#     plt.bar(
#         names,
#         means,
#     )

#     plt.ylabel("Mean test R²")
#     plt.title(
#         "C-CO12 decoder performance"
#     )
#     plt.tight_layout()

#     r2_plot = (
#         DECODER_OUT
#         / "decoder_mean_r2.png"
#     )

#     plt.savefig(
#         r2_plot,
#         dpi=200,
#     )
#     plt.close()

#     print("saved:", r2_plot)


# # =====================================================================
# # MAIN
# # =====================================================================

# def main():
#     seed_all(SEED)

#     (
#         X_train,
#         X_test,
#         Y_train,
#         Y_test,
#     ) = load_cco12()

#     (
#         vanilla_model,
#         negpos_model,
#     ) = load_models()

#     print("\n" + "#" * 100)
#     print("COMPUTE C-CO12 EMBEDDINGS")
#     print("#" * 100)

#     vanilla_train_emb, vanilla_test_emb = get_embeddings(
#         vanilla_model,
#         X_train,
#         X_test,
#     )

#     negpos_train_emb, negpos_test_emb = get_embeddings(
#         negpos_model,
#         X_train,
#         X_test,
#     )

#     print(
#         "normal CEBRA embeddings:",
#         vanilla_train_emb.shape,
#         vanilla_test_emb.shape,
#     )

#     print(
#         "NegPos embeddings:",
#         negpos_train_emb.shape,
#         negpos_test_emb.shape,
#     )

#     np.save(
#         DECODER_OUT / "normal_cebra_train_embedding.npy",
#         vanilla_train_emb,
#     )
#     np.save(
#         DECODER_OUT / "normal_cebra_test_embedding.npy",
#         vanilla_test_emb,
#     )
#     np.save(
#         DECODER_OUT / "negpos_train_embedding.npy",
#         negpos_train_emb,
#     )
#     np.save(
#         DECODER_OUT / "negpos_test_embedding.npy",
#         negpos_test_emb,
#     )

#     normal_result = train_decoder(
#         model_name="normal_cebra",
#         train_emb=vanilla_train_emb,
#         test_emb=vanilla_test_emb,
#         Y_train=Y_train,
#         Y_test=Y_test,
#     )

#     negpos_result = train_decoder(
#         model_name="negpos",
#         train_emb=negpos_train_emb,
#         test_emb=negpos_test_emb,
#         Y_train=Y_train,
#         Y_test=Y_test,
#     )

#     results = [
#         normal_result,
#         negpos_result,
#     ]

#     save_results(results)

#     print("\n" + "#" * 100)
#     print("FINAL R2")
#     print("#" * 100)

#     for r in results:
#         print(
#             f"{r['model_name']:15s} | "
#             f"mean R2={r['mean_r2']:.6f} | "
#             f"weighted R2={r['weighted_r2']:.6f}"
#         )


# if __name__ == "__main__":
#     main()




# # """
# # C-CO12 representation test:
# #   1) Vanilla CEBRA
# #   2) CEBRA-NegPos = clean CEBRA with a fraction of negatives replaced
# #      by neuron-wise-normalized foreign negatives from other C-CO sessions.

# # Both models are trained with C-CO12 train_data as the target training dataset.
# # The NegPos model additionally receives a foreign matrix ONLY for negative
# # sampling through `extra_negatives`.

# # Evaluation (NO further training):
# #   TEST 1: C-CO12 valid_data vs FAKE data
# #   TEST 2: C-CO12 valid_data vs 86 neurons selected from other C-CO sessions

# # For every comparison:
# #   - comparison data are matched NEURON-BY-NEURON to C-CO12
# #   - one PCA is fitted on the UNION of the two embeddings
# #   - both 2D and 3D PCA plots are saved

# # Expected project layout:
# #   ~/sam/result/Aggregate/
# #       this_script.py
# #       CEBRA-NegPos/
# #       utils/
# # """

# # from __future__ import annotations

# # import csv
# # import inspect
# # import os
# # import random
# # import sys
# # from pathlib import Path
# # from typing import Dict, List, Sequence, Tuple

# # import numpy as np
# # import torch

# # # ---------------------------------------------------------------------
# # # Use the modified fork directly.
# # # ---------------------------------------------------------------------
# # CEBRA_DIR = Path(__file__).resolve().parent / "CEBRA-NegPos"

# # if not CEBRA_DIR.exists():
# #     raise FileNotFoundError(f"CEBRA-NegPos fork not found: {CEBRA_DIR}")

# # # Avoid accidentally using a cached/imported different CEBRA installation.
# # for _m in list(sys.modules):
# #     if _m == "cebra" or _m.startswith("cebra."):
# #         del sys.modules[_m]

# # sys.path.insert(0, str(CEBRA_DIR))

# # import cebra
# # from cebra import CEBRA

# # from sklearn.decomposition import PCA

# # import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# # # =====================================================================
# # # CONFIG
# # # =====================================================================

# # PERICH_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")

# # DATASET_NAME = "C-CO"
# # TARGET_DAY = 12
# # TARGET_SESSION = f"{DATASET_NAME}{TARGET_DAY}"

# # N_NEURONS = 86
# # N_CCO_SESSIONS = 53  # C-CO0 ... C-CO52

# # # None = use all other C-CO sessions as candidates.
# # OTHER_SESSION_IDS = None

# # SEED = 42

# # LATENT_DIM = 64
# # HIDDEN = 64
# # BATCH_SIZE = 2048
# # MAX_ITER = 3000
# # TEMPERATURE = 0.4
# # MODEL_ARCH = "offset36-model-more-dropout"
# # DEVICE = "cuda_if_available"
# # OFFSET = 1

# # # Fraction of the normal negative bank replaced by foreign negatives.
# # EXTRA_NEGATIVE_FRACTION = 0.10

# # OUT_DIR = Path("CEBRA_CCO12_NegPos_Test")
# # MODELS_DIR = OUT_DIR / "models"
# # EMB_DIR = OUT_DIR / "embeddings"
# # PLOTS_DIR = OUT_DIR / "plots"
# # FOREIGN_MAP_CSV = OUT_DIR / "foreign_86_mapping.csv"

# # EPS = 1e-8


# # # =====================================================================
# # # BASIC HELPERS
# # # =====================================================================

# # def ensure_dirs():
# #     for d in (OUT_DIR, MODELS_DIR, EMB_DIR, PLOTS_DIR):
# #         d.mkdir(parents=True, exist_ok=True)


# # def seed_all(seed=SEED):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)


# # def session_path(session_name: str) -> Path:
# #     return PERICH_DATA_DIR / f"{session_name}.npz"


# # def load_session_raw(session_name: str):
# #     """Return raw neural train/test arrays."""
# #     path = session_path(session_name)
# #     if not path.exists():
# #         raise FileNotFoundError(path)

# #     data = np.load(path, allow_pickle=True)
# #     X_train = np.asarray(data["train_data"], dtype=np.float32)
# #     X_test = np.asarray(data["valid_data"], dtype=np.float32)

# #     if X_train.ndim != 2 or X_test.ndim != 2:
# #         raise ValueError(
# #             f"{session_name}: expected 2D arrays, got "
# #             f"train={X_train.shape}, test={X_test.shape}"
# #         )

# #     if X_train.shape[1] != X_test.shape[1]:
# #         raise ValueError(
# #             f"{session_name}: train/test neuron count mismatch: "
# #             f"{X_train.shape[1]} vs {X_test.shape[1]}"
# #         )

# #     if not np.isfinite(X_train).all():
# #         raise RuntimeError(f"{session_name}: train_data contains NaN/Inf")
# #     if not np.isfinite(X_test).all():
# #         raise RuntimeError(f"{session_name}: valid_data contains NaN/Inf")

# #     return X_train, X_test


# # # =====================================================================
# # # NEURON-WISE NORMALIZATION
# # # =====================================================================

# # def column_stats(X: np.ndarray):
# #     """Per-neuron mean/std."""
# #     mu = X.mean(axis=0, keepdims=True).astype(np.float32)
# #     sd = X.std(axis=0, keepdims=True).astype(np.float32)
# #     sd = np.maximum(sd, EPS)
# #     return mu, sd


# # def match_neuron_by_neuron(
# #     source: np.ndarray,
# #     target: np.ndarray,
# # ) -> np.ndarray:
# #     """
# #     Match source neuron j to target neuron j.

# #     For every j:
# #         source_j <- (source_j - mean(source_j)) / std(source_j)
# #         source_j <- source_j * std(target_j) + mean(target_j)

# #     After this transform:
# #         mean(source[:, j]) ~= mean(target[:, j])
# #         std(source[:, j])  ~= std(target[:, j])

# #     IMPORTANT:
# #     We do NOT globally z-score all neurons together.
# #     """
# #     source = np.asarray(source, dtype=np.float32)
# #     target = np.asarray(target, dtype=np.float32)

# #     if source.ndim != 2 or target.ndim != 2:
# #         raise ValueError("source and target must be 2D")

# #     if source.shape[1] != target.shape[1]:
# #         raise ValueError(
# #             f"Neuron count mismatch: source={source.shape}, target={target.shape}"
# #         )

# #     src_mu, src_sd = column_stats(source)
# #     tgt_mu, tgt_sd = column_stats(target)

# #     Z = (source - src_mu) / src_sd
# #     matched = Z * tgt_sd + tgt_mu

# #     return matched.astype(np.float32)


# # def print_match_check(name: str, X: np.ndarray, target: np.ndarray):
# #     """Report maximum per-neuron mean/std mismatch."""
# #     X_mu, X_sd = column_stats(X)
# #     T_mu, T_sd = column_stats(target)

# #     mean_err = float(np.max(np.abs(X_mu - T_mu)))
# #     std_err = float(np.max(np.abs(X_sd - T_sd)))

# #     print(
# #         f"{name}: shape={X.shape} | "
# #         f"max |mean_j-target_mean_j|={mean_err:.6f} | "
# #         f"max |std_j-target_std_j|={std_err:.6f}"
# #     )


# # # =====================================================================
# # # SELECT 86 RANDOM NEURONS FROM OTHER C-CO SESSIONS
# # # =====================================================================

# # ForeignNeuron = Tuple[int, int]  # (session_day, neuron_index)


# # def get_other_session_ids() -> List[int]:
# #     if OTHER_SESSION_IDS is not None:
# #         return [
# #             int(day) for day in OTHER_SESSION_IDS
# #             if int(day) != TARGET_DAY
# #         ]

# #     return [
# #         day for day in range(N_CCO_SESSIONS)
# #         if day != TARGET_DAY
# #     ]


# # def load_other_sessions() -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
# #     """Load all usable candidate sessions."""
# #     sessions = {}

# #     for day in get_other_session_ids():
# #         name = f"{DATASET_NAME}{day}"
# #         path = session_path(name)

# #         if not path.exists():
# #             print(f"[SKIP missing] {name}")
# #             continue

# #         try:
# #             X_train, X_test = load_session_raw(name)
# #         except Exception as e:
# #             print(f"[SKIP bad] {name}: {e}")
# #             continue

# #         if X_train.shape[1] == 0:
# #             print(f"[SKIP no-neurons] {name}")
# #             continue

# #         sessions[day] = (X_train, X_test)
# #         print(
# #             f"[OTHER OK] {name}: "
# #             f"train={X_train.shape}, test={X_test.shape}"
# #         )

# #     if len(sessions) == 0:
# #         raise RuntimeError("No usable other C-CO sessions found.")

# #     return sessions


# # def choose_86_random_foreign_neurons(
# #     sessions: Dict[int, Tuple[np.ndarray, np.ndarray]],
# #     seed: int = SEED,
# # ) -> List[ForeignNeuron]:
# #     """
# #     Build a pool of ALL neuron identities from all other sessions,
# #     then randomly select exactly 86 unique (session, neuron) pairs.

# #     Example:
# #         coordinate 0 <- C-CO3 neuron 17
# #         coordinate 1 <- C-CO41 neuron 6
# #         ...
# #         coordinate 85 <- C-CO7 neuron 92
# #     """
# #     pool: List[ForeignNeuron] = []

# #     for day, (X_train, _) in sessions.items():
# #         for neuron_idx in range(X_train.shape[1]):
# #             pool.append((day, neuron_idx))

# #     if len(pool) < N_NEURONS:
# #         raise RuntimeError(
# #             f"Only {len(pool)} foreign neurons available; need {N_NEURONS}."
# #         )

# #     rng = np.random.default_rng(seed)
# #     chosen_idx = rng.choice(
# #         len(pool),
# #         size=N_NEURONS,
# #         replace=False,
# #     )

# #     mapping = [pool[int(i)] for i in chosen_idx]

# #     return mapping


# # def save_foreign_mapping(mapping: Sequence[ForeignNeuron]):
# #     ensure_dirs()

# #     with FOREIGN_MAP_CSV.open("w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow([
# #             "cco12_coordinate",
# #             "foreign_session",
# #             "foreign_neuron_index",
# #         ])

# #         for j, (day, neuron_idx) in enumerate(mapping):
# #             writer.writerow([
# #                 j,
# #                 f"{DATASET_NAME}{day}",
# #                 neuron_idx,
# #             ])

# #     print("saved foreign mapping:", FOREIGN_MAP_CSV)


# # def build_foreign_matrix(
# #     sessions: Dict[int, Tuple[np.ndarray, np.ndarray]],
# #     mapping: Sequence[ForeignNeuron],
# #     split: str,
# # ) -> np.ndarray:
# #     """
# #     Create a (T_min, 86) matrix from the selected foreign neurons.

# #     split="train":
# #         use train_data from each selected neuron

# #     split="test":
# #         use valid_data from each selected neuron

# #     The 86 selected neuron identities are the SAME for train/test.
# #     Because different sessions can have different durations, all selected
# #     traces are cut to the shortest time length.
# #     """
# #     if split not in ("train", "test"):
# #         raise ValueError("split must be 'train' or 'test'")

# #     arr_idx = 0 if split == "train" else 1

# #     traces = []
# #     lengths = []

# #     for day, neuron_idx in mapping:
# #         X = sessions[day][arr_idx]

# #         if neuron_idx >= X.shape[1]:
# #             raise RuntimeError(
# #                 f"C-CO{day}: neuron {neuron_idx} unavailable in {split}."
# #             )

# #         trace = X[:, neuron_idx].astype(np.float32)
# #         traces.append(trace)
# #         lengths.append(len(trace))

# #     T_min = int(min(lengths))

# #     foreign = np.column_stack(
# #         [trace[:T_min] for trace in traces]
# #     ).astype(np.float32)

# #     assert foreign.shape == (T_min, N_NEURONS)

# #     used_sessions = sorted(set(day for day, _ in mapping))

# #     print(
# #         f"foreign_{split}_raw: {foreign.shape} | "
# #         f"T_min={T_min} | "
# #         f"selected neurons from {len(used_sessions)} sessions"
# #     )

# #     return foreign


# # # =====================================================================
# # # FAKE DATASET
# # # =====================================================================

# # def build_fake_neuronwise(X_test: np.ndarray, seed=SEED):
# #     """
# #     Build a structure-destroyed fake dataset and make its finite-sample
# #     mean/std match C-CO12 test EXACTLY neuron-by-neuron (up to fp error).

# #     1) independent Gaussian noise for every neuron/time
# #     2) standardize each fake column
# #     3) rescale each fake column to C-CO12 test neuron's mean/std

# #     No cross-neuron or temporal structure is preserved.
# #     """
# #     rng = np.random.default_rng(seed)

# #     fake0 = rng.normal(
# #         size=X_test.shape
# #     ).astype(np.float32)

# #     fake = match_neuron_by_neuron(
# #         fake0,
# #         X_test,
# #     )

# #     return fake.astype(np.float32)


# # # =====================================================================
# # # CEBRA MODEL BUILD / SAVE
# # # =====================================================================

# # def verify_negpos_fork():
# #     print("\nUsing CEBRA:")
# #     print(cebra.__file__)

# #     params = inspect.signature(CEBRA.__init__).parameters

# #     required = {
# #         "extra_negatives",
# #         "extra_negative_fraction",
# #     }

# #     missing = required.difference(params)

# #     if missing:
# #         raise RuntimeError(
# #             "Wrong CEBRA fork loaded. Missing modified arguments: "
# #             f"{sorted(missing)}"
# #         )

# #     print("[OK] CEBRA-NegPos API detected")


# # def common_cebra_kwargs():
# #     return dict(
# #         batch_size=BATCH_SIZE,
# #         temperature=TEMPERATURE,
# #         model_architecture=MODEL_ARCH,
# #         time_offsets=OFFSET,
# #         max_iterations=MAX_ITER,
# #         output_dimension=LATENT_DIM,
# #         num_hidden_units=HIDDEN,
# #         device=DEVICE,
# #         verbose=True,
# #     )


# # def build_vanilla_cebra():
# #     """
# #     Same modified fork, but extra_negatives=None.
# #     Therefore behavior should be vanilla CEBRA.
# #     """
# #     return CEBRA(
# #         **common_cebra_kwargs(),
# #         extra_negatives=None,
# #         extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
# #     )


# # def build_negpos_cebra(foreign_train_normalized: np.ndarray):
# #     """
# #     CEBRA trained on C-CO12, with some negative samples replaced
# #     by normalized foreign-session negatives.
# #     """
# #     return CEBRA(
# #         **common_cebra_kwargs(),
# #         extra_negatives=foreign_train_normalized,
# #         extra_negative_fraction=EXTRA_NEGATIVE_FRACTION,
# #     )


# # def model_path(name):
# #     return MODELS_DIR / f"{name}.pt"


# # def save_model(model, name):
# #     ensure_dirs()
# #     model.save(model_path(name))
# #     print("saved model:", model_path(name))


# # # =====================================================================
# # # EMBEDDINGS
# # # =====================================================================

# # def emb_path(name):
# #     return EMB_DIR / f"{name}.npy"


# # def save_embedding(emb, name):
# #     ensure_dirs()
# #     arr = np.asarray(emb, dtype=np.float32)
# #     np.save(emb_path(name), arr)
# #     print("saved embedding:", emb_path(name), arr.shape)


# # def embed(model, X):
# #     return np.asarray(
# #         model.transform(X.astype(np.float32)),
# #         dtype=np.float32,
# #     )


# # # =====================================================================
# # # PCA PLOTTING: ALWAYS SAVE BOTH 2D AND 3D
# # # =====================================================================

# # def plot_pca_comparison(
# #     emb_a,
# #     label_a,
# #     color_a,
# #     emb_b,
# #     label_b,
# #     color_b,
# #     title,
# #     out_2d,
# #     out_3d,
# #     point_size=6,
# #     alpha=0.5,
# # ):
# #     """
# #     Fit ONE 3-component PCA on the UNION of emb_a + emb_b.

# #     The same PCA basis is used for:
# #       - 2D plot: PC1 vs PC2
# #       - 3D plot: PC1 vs PC2 vs PC3

# #     This is necessary so red/blue clouds are directly comparable.
# #     """
# #     combined = np.concatenate(
# #         [emb_a, emb_b],
# #         axis=0,
# #     )

# #     pca = PCA(n_components=3)
# #     pca.fit(combined)

# #     proj_a = pca.transform(emb_a)
# #     proj_b = pca.transform(emb_b)

# #     var = pca.explained_variance_ratio_

# #     # ---------------- 2D ----------------
# #     plt.figure(figsize=(7, 6))

# #     plt.scatter(
# #         proj_a[:, 0],
# #         proj_a[:, 1],
# #         s=point_size,
# #         c=color_a,
# #         alpha=alpha,
# #         label=label_a,
# #     )

# #     plt.scatter(
# #         proj_b[:, 0],
# #         proj_b[:, 1],
# #         s=point_size,
# #         c=color_b,
# #         alpha=alpha,
# #         label=label_b,
# #     )

# #     plt.xlabel(f"PC1 ({var[0] * 100:.1f}%)")
# #     plt.ylabel(f"PC2 ({var[1] * 100:.1f}%)")
# #     plt.title(title + " -- PCA 2D")
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(out_2d, dpi=220)
# #     plt.close()

# #     print("saved:", out_2d)

# #     # ---------------- 3D ----------------
# #     fig = plt.figure(figsize=(8, 7))
# #     ax = fig.add_subplot(111, projection="3d")

# #     ax.scatter(
# #         proj_a[:, 0],
# #         proj_a[:, 1],
# #         proj_a[:, 2],
# #         s=point_size,
# #         c=color_a,
# #         alpha=alpha,
# #         label=label_a,
# #     )

# #     ax.scatter(
# #         proj_b[:, 0],
# #         proj_b[:, 1],
# #         proj_b[:, 2],
# #         s=point_size,
# #         c=color_b,
# #         alpha=alpha,
# #         label=label_b,
# #     )

# #     ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
# #     ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
# #     ax.set_zlabel(f"PC3 ({var[2] * 100:.1f}%)")

# #     ax.set_title(title + " -- PCA 3D")
# #     ax.legend()

# #     plt.tight_layout()
# #     plt.savefig(out_3d, dpi=220)
# #     plt.close()

# #     print("saved:", out_3d)


# # # =====================================================================
# # # STAGE 1 -- PREPARE DATA + TRAIN BOTH MODELS ON C-CO12
# # # =====================================================================

# # def stage1_train_models():
# #     print("\n" + "#" * 100)
# #     print("STAGE 1 -- TRAIN VANILLA CEBRA + CEBRA-NEGPOS ON C-CO12")
# #     print("#" * 100)

# #     # ---------------- C-CO12 ----------------
# #     X_train, X_test = load_session_raw(TARGET_SESSION)

# #     print(
# #         f"{TARGET_SESSION} neural train: {X_train.shape} | "
# #         f"test: {X_test.shape}"
# #     )

# #     if X_train.shape[1] != N_NEURONS:
# #         raise RuntimeError(
# #             f"Expected {N_NEURONS} neurons for {TARGET_SESSION}, "
# #             f"got {X_train.shape[1]}"
# #         )

# #     # ---------------- foreign neurons ----------------
# #     other_sessions = load_other_sessions()

# #     mapping = choose_86_random_foreign_neurons(
# #         other_sessions,
# #         seed=SEED + 100,
# #     )

# #     save_foreign_mapping(mapping)

# #     print("\nSelected 86 foreign neurons:")
# #     for j, (day, neuron_idx) in enumerate(mapping):
# #         print(
# #             f"  C-CO12 coordinate {j:02d} <- "
# #             f"C-CO{day} neuron {neuron_idx}"
# #         )

# #     foreign_train_raw = build_foreign_matrix(
# #         other_sessions,
# #         mapping,
# #         split="train",
# #     )

# #     foreign_test_raw = build_foreign_matrix(
# #         other_sessions,
# #         mapping,
# #         split="test",
# #     )

# #     # IMPORTANT:
# #     # extra negatives used in training are matched to C-CO12 TRAIN.
# #     foreign_train_norm = match_neuron_by_neuron(
# #         foreign_train_raw,
# #         X_train,
# #     )

# #     print("\nNeuron-wise normalization checks:")
# #     print_match_check(
# #         "foreign_train_norm vs C-CO12 train",
# #         foreign_train_norm,
# #         X_train,
# #     )
# #     # Test normalization is intentionally deferred until AFTER equal-time
# #     # cropping in TEST 2, so the final plotted arrays match neuron-by-neuron
# #     # exactly after truncation.

# #     # ---------------- model 1 ----------------
# #     print("\n" + "=" * 100)
# #     print("MODEL 1 -- VANILLA CLEAN CEBRA")
# #     print("=" * 100)

# #     seed_all(SEED)

# #     vanilla_model = build_vanilla_cebra()
# #     vanilla_model.fit(X_train)

# #     save_model(
# #         vanilla_model,
# #         "cebra_vanilla",
# #     )

# #     # ---------------- model 2 ----------------
# #     print("\n" + "=" * 100)
# #     print("MODEL 2 -- CLEAN CEBRA + FOREIGN NEGATIVE PAIRS")
# #     print("=" * 100)

# #     n_foreign = round(
# #         EXTRA_NEGATIVE_FRACTION * BATCH_SIZE
# #     )

# #     print(
# #         f"batch_size={BATCH_SIZE} | "
# #         f"extra_negative_fraction={EXTRA_NEGATIVE_FRACTION} | "
# #         f"approximately {n_foreign} foreign negatives per batch"
# #     )

# #     seed_all(SEED)

# #     negpos_model = build_negpos_cebra(
# #         foreign_train_norm,
# #     )

# #     # Main training data are STILL only C-CO12.
# #     negpos_model.fit(X_train)

# #     save_model(
# #         negpos_model,
# #         "cebra_negpos",
# #     )

# #     return (
# #         vanilla_model,
# #         negpos_model,
# #         X_train,
# #         X_test,
# #         foreign_test_raw,
# #     )


# # # =====================================================================
# # # TEST 1 -- C-CO12 TEST vs FAKE
# # # =====================================================================

# # def stage2_test_fake(
# #     vanilla_model,
# #     negpos_model,
# #     X_test,
# # ):
# #     print("\n" + "#" * 100)
# #     print("TEST 1 -- C-CO12 TEST vs NEURON-WISE NORMALIZED FAKE")
# #     print("#" * 100)

# #     fake = build_fake_neuronwise(
# #         X_test,
# #         seed=SEED + 200,
# #     )

# #     print_match_check(
# #         "fake vs C-CO12 test",
# #         fake,
# #         X_test,
# #     )

# #     models = (
# #         ("vanilla", vanilla_model),
# #         ("negpos", negpos_model),
# #     )

# #     for name, model in models:
# #         print(f"\n--- {name.upper()} ---")

# #         real_emb = embed(
# #             model,
# #             X_test,
# #         )

# #         fake_emb = embed(
# #             model,
# #             fake,
# #         )

# #         save_embedding(
# #             real_emb,
# #             f"{name}_cco12_test",
# #         )

# #         save_embedding(
# #             fake_emb,
# #             f"{name}_fake",
# #         )

# #         plot_pca_comparison(
# #             real_emb,
# #             f"{TARGET_SESSION} test",
# #             "red",
# #             fake_emb,
# #             "fake, neuron-wise matched",
# #             "blue",
# #             title=f"{name.upper()} -- C-CO12 test vs fake",
# #             out_2d=PLOTS_DIR / f"{name}_cco12_vs_fake_pca2d.png",
# #             out_3d=PLOTS_DIR / f"{name}_cco12_vs_fake_pca3d.png",
# #         )


# # # =====================================================================
# # # TEST 2 -- C-CO12 TEST vs OTHER-SESSION 86 NEURONS
# # # =====================================================================

# # def stage3_test_other86(
# #     vanilla_model,
# #     negpos_model,
# #     X_test,
# #     foreign_test_raw,
# # ):
# #     print("\n" + "#" * 100)
# #     print("TEST 2 -- C-CO12 TEST vs OTHER-SESSION 86 NEURONS")
# #     print("#" * 100)

# #     # Equal number of timepoints in the two clouds.
# #     T = min(
# #         X_test.shape[0],
# #         foreign_test_raw.shape[0],
# #     )

# #     X_real = X_test[:T]
# #     X_other_raw = foreign_test_raw[:T]

# #     # CRITICAL: normalize AFTER equal-time cropping.
# #     # coordinate j of foreign data is matched to neuron j of C-CO12 test.
# #     X_other = match_neuron_by_neuron(
# #         X_other_raw,
# #         X_real,
# #     )

# #     print(
# #         f"comparison length T={T} | "
# #         f"C-CO12={X_real.shape} | "
# #         f"other86={X_other.shape}"
# #     )

# #     print_match_check(
# #         "final other86 vs final C-CO12 test",
# #         X_other,
# #         X_real,
# #     )

# #     models = (
# #         ("vanilla", vanilla_model),
# #         ("negpos", negpos_model),
# #     )

# #     for name, model in models:
# #         print(f"\n--- {name.upper()} ---")

# #         real_emb = embed(
# #             model,
# #             X_real,
# #         )

# #         other_emb = embed(
# #             model,
# #             X_other,
# #         )

# #         save_embedding(
# #             real_emb,
# #             f"{name}_cco12_test_other86_equalT",
# #         )

# #         save_embedding(
# #             other_emb,
# #             f"{name}_other86_test",
# #         )

# #         plot_pca_comparison(
# #             real_emb,
# #             f"{TARGET_SESSION} test",
# #             "red",
# #             other_emb,
# #             "other-session 86, neuron-wise matched",
# #             "blue",
# #             title=f"{name.upper()} -- C-CO12 test vs other-session 86",
# #             out_2d=PLOTS_DIR / f"{name}_cco12_vs_other86_pca2d.png",
# #             out_3d=PLOTS_DIR / f"{name}_cco12_vs_other86_pca3d.png",
# #         )


# # # =====================================================================
# # # MAIN
# # # =====================================================================

# # def main():
# #     ensure_dirs()
# #     seed_all(SEED)
# #     verify_negpos_fork()

# #     (
# #         vanilla_model,
# #         negpos_model,
# #         X_train,
# #         X_test,
# #         foreign_test_raw,
# #     ) = stage1_train_models()

# #     # From here onward: TEST ONLY. No fit() calls.
# #     stage2_test_fake(
# #         vanilla_model,
# #         negpos_model,
# #         X_test,
# #     )

# #     stage3_test_other86(
# #         vanilla_model,
# #         negpos_model,
# #         X_test,
# #         foreign_test_raw,
# #     )

# #     print("\n" + "#" * 100)
# #     print("ALL DONE")
# #     print("#" * 100)

# #     print("models:")
# #     print(" ", model_path("cebra_vanilla"))
# #     print(" ", model_path("cebra_negpos"))

# #     print("\nplots:")
# #     print(" ", PLOTS_DIR / "vanilla_cco12_vs_fake_pca2d.png")
# #     print(" ", PLOTS_DIR / "vanilla_cco12_vs_fake_pca3d.png")
# #     print(" ", PLOTS_DIR / "negpos_cco12_vs_fake_pca2d.png")
# #     print(" ", PLOTS_DIR / "negpos_cco12_vs_fake_pca3d.png")
# #     print(" ", PLOTS_DIR / "vanilla_cco12_vs_other86_pca2d.png")
# #     print(" ", PLOTS_DIR / "vanilla_cco12_vs_other86_pca3d.png")
# #     print(" ", PLOTS_DIR / "negpos_cco12_vs_other86_pca2d.png")
# #     print(" ", PLOTS_DIR / "negpos_cco12_vs_other86_pca3d.png")

# #     print("\nforeign mapping:")
# #     print(" ", FOREIGN_MAP_CSV)


# # if __name__ == "__main__":
# #     main()
