#normal decoder

import sys
import gc
import math
import random
import numbers
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import r2_score

from utils.constants import CEBRA_DIR
from utils.min_distance import min_l2_distance

# ============================================================
# USE THE UNMODIFIED / ORIGINAL CEBRA FORK
# ============================================================
CEBRA_ORIGINAL_DIR = Path(CEBRA_DIR).resolve().parent / "CEBRA-original"

if not CEBRA_ORIGINAL_DIR.exists():
    raise FileNotFoundError(
        f"Could not find original CEBRA fork at: {CEBRA_ORIGINAL_DIR}\n"
        "Expected it next to the custom CEBRA fork and named exactly 'CEBRA-original'."
    )

for module_name in list(sys.modules):
    if module_name == "cebra" or module_name.startswith("cebra."):
        del sys.modules[module_name]

sys.path.insert(0, str(CEBRA_ORIGINAL_DIR))

import cebra
from cebra.solver.single_session import SingleSessionSolver
from cebra.models.jacobian_regularizer import JacobianReg

print("\nUsing ORIGINAL CEBRA from:")
print(cebra.__file__)
print("CEBRA version:", getattr(cebra, "__version__", "unknown"))
print("Original repo:", CEBRA_ORIGINAL_DIR)

# ============================================================
# CONFIG
# ============================================================
NWB_PATH = "data/Area2_Bump/sub-Han_desc-train_behavior+ecephys.nwb"
OUT = "Area2_Bump_Full_JR_Decoder"
os.makedirs(OUT, exist_ok=True)

# Neural preprocessing
BIN_MS = 50.0
BIN_SEC = BIN_MS / 1000.0
SMOOTH_SD_MS = 100.0
SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
SMOOTH_KERNEL_SIZE = 17

# Train/test
TRAIN_FRAC = 0.80

# CEBRA / ACORN
SEED = 42
LATENT_DIM = 128
NUM_HIDDEN_UNITS = 128
BATCH_SIZE = 512
MAX_ITER = 3000
LEARNING_RATE = 3e-4
TEMPERATURE = 0.4
TIME_OFFSETS = 4
MODEL_ARCH = "offset36-model-more-dropout"

# xCEBRA-style Jacobian regularization
JR_LAMBDA_MAX = 0.10
JR_N_PROJ = 1
JR_SUBBATCH = 512

# Preserve 2500 warmup + 2500 ramp over 20000 steps proportionally.
JR_WARMUP_STEPS = max(1, int(round(MAX_ITER * 2500 / 20000)))
JR_RAMP_STEPS = max(1, int(round(MAX_ITER * 2500 / 20000)))

# ACORN / PGD
ADV_STEPS = 10
ATTACK_NORM = "linf"
ADV_AGGREGATE_TRAJECTORY = True

# Decoder
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_EPOCHS = 6000
DECODER_LR = 1e-3
DECODER_WEIGHT_DECAY = 2e-4

TRANSFORM_BATCH_SIZE = 4096

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", DEVICE)

# ============================================================
# CHECK ARCHITECTURE
# ============================================================
model_options = list(cebra.models.get_options())
if MODEL_ARCH not in model_options:
    raise RuntimeError(
        f"MODEL_ARCH='{MODEL_ARCH}' is not registered in CEBRA-original.\n"
        "Do NOT silently replace it, because that would change the experiment.\n"
        f"Available model options include: {model_options[:20]}"
    )

# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


seed_all(SEED)

# ============================================================
# GAUSSIAN SMOOTHING -- NO NORMALIZATION
# ============================================================
class GaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma, dim=1):
        super().__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        kernel = 1.0
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="ij",
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2.0
            kernel *= (
                1.0
                / (std * math.sqrt(2.0 * math.pi))
                * torch.exp(-0.5 * ((mgrid - mean) / std) ** 2)
            )
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer("weight", kernel)
        self.groups = channels

    def forward(self, x):
        # B x T x N -> B x N x T
        x = x.permute(0, 2, 1)
        x = F.conv1d(x, weight=self.weight, groups=self.groups, padding="same")
        # B x N x T -> B x T x N
        return x.permute(0, 2, 1)


def smooth_neural(X):
    print("\n" + "=" * 100)
    print("GAUSSIAN SMOOTHING")
    print("=" * 100)
    print("Input:", X.shape)
    print("SD:", SMOOTH_SD_MS, "ms")
    print("Sigma:", SMOOTH_SIGMA_BINS, "bins")
    print("Kernel:", SMOOTH_KERNEL_SIZE)

    x_t = torch.from_numpy(X.astype(np.float32, copy=False)).unsqueeze(0)
    smoother = GaussianSmoothing(
        channels=X.shape[1],
        kernel_size=SMOOTH_KERNEL_SIZE,
        sigma=SMOOTH_SIGMA_BINS,
        dim=1,
    )
    smoother.eval()
    with torch.no_grad():
        X_smooth = smoother(x_t).squeeze(0).cpu().numpy().astype(np.float32)

    print("Smoothed:", X_smooth.shape)
    print("NO Z-SCORE / NO NORMALIZATION")
    return X_smooth

# ============================================================
# LOAD AREA2
# ============================================================
def load_area2():
    print("\n" + "=" * 100)
    print("LOADING AREA2 BUMP")
    print("=" * 100)

    with h5py.File(NWB_PATH, "r") as f:
        vel_group = f["processing/behavior/hand_vel"]
        vel_ds = vel_group["data"]
        starting_time_ds = vel_group["starting_time"]

        t_start = float(starting_time_ds[()])
        behavior_rate = float(starting_time_ds.attrs["rate"])
        n_behavior_samples = vel_ds.shape[0]

        samples_per_bin = int(round(behavior_rate * BIN_SEC))
        n_bins = n_behavior_samples // samples_per_bin
        usable_samples = n_bins * samples_per_bin
        t_stop = t_start + n_bins * BIN_SEC

        vel_raw = np.asarray(vel_ds[:usable_samples], dtype=np.float32)
        if vel_raw.ndim != 2 or vel_raw.shape[1] != 2:
            raise RuntimeError(f"Expected hand_vel shape (T,2), got {vel_raw.shape}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            Y = np.nanmean(
                vel_raw.reshape(n_bins, samples_per_bin, 2), axis=1
            ).astype(np.float32)

        units = f["units"]
        unit_ids = np.asarray(units["id"][:], dtype=np.int64)
        spike_times = units["spike_times"]
        spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)

        n_units = len(unit_ids)
        X_counts = np.zeros((n_bins, n_units), dtype=np.float32)

        print("Behavior rate:", behavior_rate)
        print("50-ms bins:", n_bins)
        print("Neurons:", n_units)
        print("Binning spikes...")

        edges = t_start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SEC
        for neuron_idx in range(n_units):
            end_idx = int(spike_index[neuron_idx])
            start_idx = 0 if neuron_idx == 0 else int(spike_index[neuron_idx - 1])
            spikes = np.asarray(spike_times[start_idx:end_idx], dtype=np.float64)
            spikes = spikes[(spikes >= t_start) & (spikes < t_stop)]
            X_counts[:, neuron_idx] = np.histogram(spikes, bins=edges)[0].astype(np.float32)

    assert X_counts.shape[0] == Y.shape[0]
    print("X_counts:", X_counts.shape)
    print("Y:", Y.shape)
    print("Y NaNs:", int(np.isnan(Y).sum()))
    return X_counts, Y, unit_ids

# ============================================================
# JR + MANUAL ACORN ON ORIGINAL CEBRA
# ============================================================
def lambda_jr_at(step: int) -> float:
    if step < JR_WARMUP_STEPS:
        return 0.0
    frac = (step - JR_WARMUP_STEPS) / float(JR_RAMP_STEPS)
    return JR_LAMBDA_MAX * min(1.0, max(0.0, frac))


def _autocast_bf16(x: torch.Tensor):
    return torch.autocast(
        device_type=x.device.type,
        dtype=torch.bfloat16,
        enabled=(x.device.type == "cuda"),
    )


def _subsample_for_jr(x: torch.Tensor) -> torch.Tensor:
    if JR_SUBBATCH and x.shape[0] > JR_SUBBATCH:
        idx = torch.randperm(x.shape[0], device=x.device)[:JR_SUBBATCH]
        return x[idx]
    return x


def _jacobian_reg_fp32(model: nn.Module, reg: nn.Module, x: torch.Tensor):
    """Official CEBRA JacobianReg in FP32.

    R_J(x) = 1/2 ||df(x)/dx||_F^2 (stochastic projection for n=1).
    The input is detached from PGD so JR does not alter attack construction.
    """
    xj = _subsample_for_jr(x.detach()).float().requires_grad_(True)
    with torch.autocast(device_type=xj.device.type, enabled=False):
        zj = model(xj)
        jr = reg(xj, zj)
    return jr


class RegularizedCleanSolver(SingleSessionSolver):
    """CLEAN: InfoNCE(clean) + lambda_t * JR(clean)."""

    def __post_init__(self):
        super().__post_init__()
        self.jacobian_reg = JacobianReg(n=JR_N_PROJ).to(self.device)
        self.outer_step = 0
        self.log.setdefault("nce", [])
        self.log.setdefault("jr", [])
        self.log.setdefault("lambda_jr", [])
        self.log.setdefault("objective", [])

    def step(self, batch):
        lam = lambda_jr_at(self.outer_step)
        self.optimizer.zero_grad()

        with _autocast_bf16(batch.reference):
            pred = self._inference(batch)
            nce_loss, align, uniform = self.criterion(
                pred.reference, pred.positive, pred.negative
            )

        if lam > 0.0:
            jr = _jacobian_reg_fp32(self.model, self.jacobian_reg, batch.reference)
        else:
            jr = torch.zeros((), device=batch.reference.device, dtype=torch.float32)

        objective = nce_loss.float() + lam * jr
        objective.backward()
        self.optimizer.step()

        nce_v = float(nce_loss.detach().float().item())
        jr_v = float(jr.detach().float().item())
        obj_v = float(objective.detach().float().item())

        self.history.append(obj_v)
        stats = {
            "pos": float(align.detach().float().item()),
            "neg": float(uniform.detach().float().item()),
            "total": obj_v,
            "temperature": self.criterion.temperature,
            "nce": nce_v,
            "jr": jr_v,
            "lambda_jr": lam,
        }
        for k, v in stats.items():
            self.log.setdefault(k, []).append(v)
        self.log["objective"].append(obj_v)
        self.outer_step += 1
        return stats


class ACORNAdversarialRegularizedSolver(SingleSessionSolver):
    """ACORN + JR.

    Per outer iteration:
      1) InfoNCE(clean) + lambda_t*JR(clean), optimizer.step()
      2) Build Linf PGD using PURE InfoNCE only
      3) mean InfoNCE over all 10 PGD trajectory points
         + lambda_t*JR(final adversarial input), optimizer.step()
    """

    def __post_init__(self):
        super().__post_init__()
        self.jacobian_reg = JacobianReg(n=JR_N_PROJ).to(self.device)
        self.outer_step = 0
        self.adv_epsilon = 0.0
        self.adv_alpha = 0.0
        self.adv_steps = ADV_STEPS

        for key in (
            "clean_nce", "clean_jr", "clean_objective",
            "adv_nce", "adv_jr", "adv_objective", "lambda_jr",
        ):
            self.log.setdefault(key, [])

    def _clean_regularized_update(self, batch, lam):
        self.optimizer.zero_grad()

        with _autocast_bf16(batch.reference):
            pred = self._inference(batch)
            nce_loss, align, uniform = self.criterion(
                pred.reference, pred.positive, pred.negative
            )

        if lam > 0.0:
            jr = _jacobian_reg_fp32(self.model, self.jacobian_reg, batch.reference)
        else:
            jr = torch.zeros((), device=batch.reference.device, dtype=torch.float32)

        objective = nce_loss.float() + lam * jr
        objective.backward()
        self.optimizer.step()
        return nce_loss, align, uniform, jr, objective

    def _build_pgd_trajectory(self, batch):
        if ATTACK_NORM != "linf":
            raise NotImplementedError("This experiment is configured for Linf ACORN.")

        x_base = batch.reference.detach()
        eps = float(self.adv_epsilon)
        alpha = float(self.adv_alpha)
        n_steps = int(self.adv_steps)

        x_adv = x_base.clone().detach() + torch.empty_like(x_base).uniform_(-eps, eps)
        x_adv.requires_grad_(True)
        adv_traj = []

        for _ in range(n_steps):
            adv_batch = cebra.data.Batch(
                reference=x_adv,
                positive=batch.positive,
                negative=batch.negative,
            )

            with _autocast_bf16(x_adv):
                adv_output = self._inference(adv_batch)
                adv_loss = self.criterion(
                    adv_output.reference,
                    adv_output.positive,
                    adv_output.negative,
                )[0]

            (grad_x,) = torch.autograd.grad(
                adv_loss,
                x_adv,
                retain_graph=False,
                create_graph=False,
            )

            with torch.no_grad():
                x_adv = x_adv + alpha * grad_x.sign()
                x_adv = torch.max(
                    torch.min(x_adv, x_base + eps),
                    x_base - eps,
                )

            x_adv = x_adv.detach().requires_grad_(True)
            if ADV_AGGREGATE_TRAJECTORY:
                adv_traj.append(x_adv.detach())

        if ADV_AGGREGATE_TRAJECTORY:
            if len(adv_traj) != n_steps:
                raise RuntimeError(
                    f"Expected {n_steps} adversarial trajectory points, got {len(adv_traj)}"
                )
        else:
            adv_traj = [x_adv.detach()]

        return adv_traj

    def _adversarial_regularized_update(self, batch, adv_traj, lam):
        self.optimizer.zero_grad()

        weight = 1.0 / len(adv_traj)
        adv_nce_total = None

        for x_i in adv_traj:
            adv_batch = cebra.data.Batch(
                reference=x_i,
                positive=batch.positive,
                negative=batch.negative,
            )

            with _autocast_bf16(x_i):
                output = self._inference(adv_batch)
                loss_i, _, _ = self.criterion(
                    output.reference,
                    output.positive,
                    output.negative,
                )

            term = weight * loss_i.float()
            adv_nce_total = term if adv_nce_total is None else adv_nce_total + term

        # PGD is fully finished first; only now evaluate JR on its final endpoint.
        x_adv_final = adv_traj[-1]
        if lam > 0.0:
            adv_jr = _jacobian_reg_fp32(
                self.model,
                self.jacobian_reg,
                x_adv_final,
            )
        else:
            adv_jr = torch.zeros((), device=x_adv_final.device, dtype=torch.float32)

        adv_objective = adv_nce_total + lam * adv_jr
        adv_objective.backward()
        self.optimizer.step()
        return adv_nce_total, adv_jr, adv_objective

    def step(self, batch):
        lam = lambda_jr_at(self.outer_step)

        clean_nce, align, uniform, clean_jr, clean_obj = (
            self._clean_regularized_update(batch, lam)
        )

        adv_traj = self._build_pgd_trajectory(batch)

        adv_nce, adv_jr, adv_obj = self._adversarial_regularized_update(
            batch, adv_traj, lam
        )

        clean_nce_v = float(clean_nce.detach().float().item())
        clean_jr_v = float(clean_jr.detach().float().item())
        clean_obj_v = float(clean_obj.detach().float().item())
        adv_nce_v = float(adv_nce.detach().float().item())
        adv_jr_v = float(adv_jr.detach().float().item())
        adv_obj_v = float(adv_obj.detach().float().item())

        self.history.append(clean_obj_v + adv_obj_v)

        stats = {
            "pos": float(align.detach().float().item()),
            "neg": float(uniform.detach().float().item()),
            "total": clean_obj_v,
            "temperature": self.criterion.temperature,
            "clean_nce": clean_nce_v,
            "clean_jr": clean_jr_v,
            "adv_nce": adv_nce_v,
            "adv_jr": adv_jr_v,
            "adv_total": adv_obj_v,
            "lambda_jr": lam,
        }

        for k, v in stats.items():
            self.log.setdefault(k, []).append(v)
        self.log["clean_objective"].append(clean_obj_v)
        self.log["adv_objective"].append(adv_obj_v)

        self.outer_step += 1
        return stats

# ============================================================
# EPSILON
# ============================================================
def compute_adv_epsilon(X_train):
    train_tensor = torch.from_numpy(X_train.astype(np.float32, copy=False)).float()
    eps = float(min_l2_distance(train_tensor)) / 2.0
    eps = max(eps, 1e-6)
    print("adv_epsilon:", eps)
    print("adv_alpha  :", eps / 5.0)
    return eps

# ============================================================
# BUILD + TRAIN REPRESENTATION
# ============================================================
def train_representation(X_train, adversarial, name):
    print("\n" + "#" * 110)
    print(name)
    print("#" * 110)
    print("X_train:", X_train.shape)
    print("JR lambda max:", JR_LAMBDA_MAX)
    print("JR warmup/ramp:", JR_WARMUP_STEPS, "/", JR_RAMP_STEPS)
    print("JR projections:", JR_N_PROJ)
    print("Adversarial:", adversarial)
    if adversarial:
        print("ADV trajectory aggregation:", ADV_AGGREGATE_TRAJECTORY)
        print("ACORN ADV JR location: FINAL adversarial input AFTER PGD")

    seed_all(SEED)
    X_train = X_train.astype(np.float32, copy=False)

    neural_tensor = torch.from_numpy(X_train).float()

    # Original low-level TensorDataset requires an index. This chronological
    # dummy index is NOT a behavioral label because the loader is conditional='time'.
    time_index = torch.arange(X_train.shape[0], dtype=torch.float32).unsqueeze(1)

    dataset = cebra.data.TensorDataset(
        neural=neural_tensor,
        continuous=time_index,
    )

    model = cebra.models.init(
        name=MODEL_ARCH,
        num_neurons=dataset.input_dimension,
        num_units=NUM_HIDDEN_UNITS,
        num_output=LATENT_DIM,
        normalize=True,
    ).to(DEVICE)

    dataset.configure_for(model)
    dataset.to(DEVICE)

    loader = cebra.data.single_session.ContinuousDataLoader(
        dataset=dataset,
        time_offset=TIME_OFFSETS,
        num_steps=MAX_ITER,
        batch_size=BATCH_SIZE,
        conditional="time",
    ).to(DEVICE)

    criterion = cebra.models.criterions.FixedCosineInfoNCE(
        temperature=TEMPERATURE
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )

    if adversarial:
        eps = compute_adv_epsilon(X_train)
        solver = ACORNAdversarialRegularizedSolver(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            tqdm_on=True,
        ).to(DEVICE)
        solver.adv_epsilon = eps
        solver.adv_alpha = eps / 5.0
        solver.adv_steps = ADV_STEPS
    else:
        solver = RegularizedCleanSolver(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            tqdm_on=True,
        ).to(DEVICE)

    solver.fit(loader=loader)

    del loader, dataset
    cleanup()
    return solver

# ============================================================
# EMBEDDINGS
# ============================================================
def get_embeddings(solver, X):
    X_t = torch.from_numpy(X.astype(np.float32, copy=False)).float().to(DEVICE)
    with torch.no_grad():
        z = solver.transform(
            X_t,
            pad_before_transform=True,
            batch_size=TRANSFORM_BATCH_SIZE,
        )
    z = z.detach().cpu().numpy().astype(np.float32)
    del X_t
    cleanup()
    return z

# ============================================================
# DECODER -- SAME 6000-EPOCH FULL-BATCH DECODER
# ============================================================
class TwoLayerMLP(nn.Module):
    def __init__(
        self,
        input_dim=128,
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
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)


def train_decoder(z_train_np, y_train_np, z_test_np, y_test_np, model_name):
    train_mask = np.isfinite(z_train_np).all(axis=1) & np.isfinite(y_train_np).all(axis=1)
    test_mask = np.isfinite(z_test_np).all(axis=1) & np.isfinite(y_test_np).all(axis=1)

    z_train_np = z_train_np[train_mask]
    y_train_np = y_train_np[train_mask]
    z_test_np = z_test_np[test_mask]
    y_test_np = y_test_np[test_mask]

    assert len(z_train_np) > 0 and len(z_test_np) > 0

    print(f"{model_name} valid decoder samples: train={len(z_train_np)} test={len(z_test_np)}")

    seed_all(SEED)
    decoder = TwoLayerMLP(
        input_dim=z_train_np.shape[1],
        hidden_dim=DECODER_HIDDEN,
        output_dim=2,
        dropout_rate=DECODER_DROPOUT,
    ).to(DEVICE)

    z_train = torch.from_numpy(z_train_np).float().to(DEVICE)
    y_train = torch.from_numpy(y_train_np).float().to(DEVICE)
    z_test = torch.from_numpy(z_test_np).float().to(DEVICE)
    y_test = torch.from_numpy(y_test_np).float().to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=DECODER_LR,
        weight_decay=DECODER_WEIGHT_DECAY,
    )

    print("\n" + "=" * 100)
    print("DECODER —", model_name)
    print("=" * 100)

    for epoch in range(DECODER_EPOCHS):
        decoder.train()
        optimizer.zero_grad()
        pred = decoder(z_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch == 0 or (epoch + 1) % 1000 == 0:
            print(
                f"{model_name} | Epoch {epoch+1}/{DECODER_EPOCHS} | "
                f"train MSE={loss.item():.8f}"
            )

    decoder.eval()
    with torch.no_grad():
        pred = decoder(z_test).cpu().numpy()
        true = y_test.cpu().numpy()

    mse = float(np.mean((true - pred) ** 2))
    r2_vx = float(r2_score(true[:, 0], pred[:, 0]))
    r2_vy = float(r2_score(true[:, 1], pred[:, 1]))
    mean_r2 = float((r2_vx + r2_vy) / 2.0)

    print(
        f"{model_name} | MSE={mse:.6f} | "
        f"R2 vx={r2_vx:.6f} | R2 vy={r2_vy:.6f} | Mean R2={mean_r2:.6f}"
    )

    del decoder, optimizer, z_train, y_train, z_test, y_test
    cleanup()

    return {
        "model": model_name,
        "mse": mse,
        "r2_vx": r2_vx,
        "r2_vy": r2_vy,
        "mean_r2": mean_r2,
    }

# ============================================================
# MAIN -- ONLY THE TWO FULL MODELS + DECODERS
# ============================================================
def main():
    print("\n" + "=" * 110)
    print("AREA2 BUMP — FULL CLEAN+JR vs FULL ACORN+JR + DECODER")
    print("=" * 110)
    print("Original CEBRA repo:", CEBRA_ORIGINAL_DIR)
    print("No Top-K / no attribution in this script")
    print("Decoder epochs:", DECODER_EPOCHS)

    X_counts, Y, unit_ids = load_area2()
    X = smooth_neural(X_counts)
    del X_counts
    cleanup()

    split_idx = int(TRAIN_FRAC * len(X))
    X_train = X[:split_idx].astype(np.float32, copy=False)
    X_test = X[split_idx:].astype(np.float32, copy=False)
    Y_train = Y[:split_idx].astype(np.float32, copy=False)
    Y_test = Y[split_idx:].astype(np.float32, copy=False)

    print("\nTemporal 80/20 split")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("Y_train:", Y_train.shape)
    print("Y_test :", Y_test.shape)
    print("Neurons:", len(unit_ids))

    rows = []

    # --------------------------------------------------------
    # FULL CLEAN + JR -> embeddings -> decoder
    # --------------------------------------------------------
    clean_solver = train_representation(
        X_train,
        adversarial=False,
        name="FULL CLEAN + JR",
    )

    z_train_clean = get_embeddings(clean_solver, X_train)
    z_test_clean = get_embeddings(clean_solver, X_test)

    print("FULL CLEAN embeddings:", z_train_clean.shape, z_test_clean.shape)

    clean_metrics = train_decoder(
        z_train_clean,
        Y_train,
        z_test_clean,
        Y_test,
        model_name="FULL CLEAN + JR",
    )
    rows.append(clean_metrics)

    del clean_solver, z_train_clean, z_test_clean
    cleanup()

    # --------------------------------------------------------
    # FULL ACORN + JR -> embeddings -> decoder
    # --------------------------------------------------------
    acorn_solver = train_representation(
        X_train,
        adversarial=True,
        name="FULL ACORN + JR",
    )

    z_train_acorn = get_embeddings(acorn_solver, X_train)
    z_test_acorn = get_embeddings(acorn_solver, X_test)

    print("FULL ACORN embeddings:", z_train_acorn.shape, z_test_acorn.shape)

    acorn_metrics = train_decoder(
        z_train_acorn,
        Y_train,
        z_test_acorn,
        Y_test,
        model_name="FULL ACORN + JR",
    )
    rows.append(acorn_metrics)

    del acorn_solver, z_train_acorn, z_test_acorn
    cleanup()

    # --------------------------------------------------------
    # FINAL TABLE + CSV
    # --------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT, "Full_CLEAN_ACORN_JR_Decoder.csv")
    df.to_csv(csv_path, index=False, float_format="%.8f")

    print("\n\n" + "=" * 100)
    print("FULL MODEL DECODER RESULTS")
    print("=" * 100)
    print(
        f"{'MODEL':24s} {'MSE':>12s} {'R2 vx':>12s} "
        f"{'R2 vy':>12s} {'Mean R2':>12s}"
    )
    print("-" * 100)
    for row in rows:
        print(
            f"{row['model']:24s} "
            f"{row['mse']:12.6f} "
            f"{row['r2_vx']:12.6f} "
            f"{row['r2_vy']:12.6f} "
            f"{row['mean_r2']:12.6f}"
        )

    print("\nSaved:", csv_path)


if __name__ == "__main__":
    main()

#topk
# import os
# import sys
# import gc
# import math
# import random
# import numbers
# import warnings
# from pathlib import Path

# import h5py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from sklearn.metrics import r2_score

# from utils.constants import CEBRA_DIR
# from utils.min_distance import min_l2_distance

# # ============================================================
# # IMPORTANT: USE THE UNMODIFIED / ORIGINAL CEBRA FORK
# # ============================================================
# CEBRA_ORIGINAL_DIR = Path(CEBRA_DIR).resolve().parent / "CEBRA-original"

# if not CEBRA_ORIGINAL_DIR.exists():
#     raise FileNotFoundError(
#         f"Could not find original CEBRA fork at: {CEBRA_ORIGINAL_DIR}\n"
#         "Expected it next to the custom CEBRA fork and named exactly 'CEBRA-Orginal'."
#     )

# # Remove any already imported custom-fork CEBRA modules.
# for module_name in list(sys.modules):
#     if module_name == "cebra" or module_name.startswith("cebra."):
#         del sys.modules[module_name]

# sys.path.insert(0, str(CEBRA_ORIGINAL_DIR))

# import cebra
# import cebra.attribution
# from cebra.solver.single_session import SingleSessionSolver
# from cebra.models.jacobian_regularizer import JacobianReg

# print("\nUsing ORIGINAL CEBRA from:")
# print(cebra.__file__)
# print("CEBRA version:", getattr(cebra, "__version__", "unknown"))
# print("Original repo:", CEBRA_ORIGINAL_DIR)

# # ============================================================
# # CONFIG
# # ============================================================
# NWB_PATH = "data/Area2_Bump/sub-Han_desc-train_behavior+ecephys.nwb"
# OUT = "Area2_Bump_TopK_JR_AdvFirst_Aggregated"
# os.makedirs(OUT, exist_ok=True)

# # Neural preprocessing
# BIN_MS = 50.0
# BIN_SEC = BIN_MS / 1000.0
# SMOOTH_SD_MS = 100.0
# SMOOTH_SIGMA_BINS = SMOOTH_SD_MS / BIN_MS
# SMOOTH_KERNEL_SIZE = 17

# # Train/test
# TRAIN_FRAC = 0.80

# # CEBRA / ACORN
# SEED = 42
# LATENT_DIM = 128
# NUM_HIDDEN_UNITS = 128
# BATCH_SIZE = 512
# MAX_ITER = 3000
# LEARNING_RATE = 3e-4
# TEMPERATURE = 0.4
# TIME_OFFSETS = 4
# MODEL_ARCH = "offset36-model-more-dropout"

# # xCEBRA-style Jacobian regularization.
# # We use ORIGINAL CEBRA's official JacobianReg implementation, but we control
# # exactly WHEN it is evaluated so that ACORN is: clean update -> PGD attack ->
# # trajectory-aggregated adversarial InfoNCE + JR(final adversarial input).
# JR_LAMBDA_MAX = 0.10
# JR_N_PROJ = 1
# JR_SUBBATCH = 512

# # xCEBRA paper/demo schedule shape: 2500 warmup + 2500 ramp over 20000 steps.
# # We preserve those proportions at MAX_ITER=3000: 375 warmup + 375 ramp.
# JR_WARMUP_STEPS = max(1, int(round(MAX_ITER * 2500 / 20000)))
# JR_RAMP_STEPS = max(1, int(round(MAX_ITER * 2500 / 20000)))

# # ACORN / PGD -- same epsilon/alpha/steps geometry as prior experiment.
# ADV_STEPS = 10
# ATTACK_NORM = "linf"

# # IMPORTANT: target the advisor-described fork revision, where the FINAL
# # adversarial model update averages InfoNCE across all PGD trajectory points.
# # The earlier uploaded snapshot in this chat showed a local hard-coded False;
# # keep this True to match the advisor-described authoritative revision.
# ADV_AGGREGATE_TRAJECTORY = True

# # Attribution
# ATTR_N_CHUNKS = 16
# ATTR_CHUNK_LEN = 128
# ATTR_BATCH_SIZE = 16

# # Decoder
# DECODER_HIDDEN = 64
# DECODER_DROPOUT = 0.4
# DECODER_EPOCHS = 6000
# DECODER_LR = 1e-3
# DECODER_WEIGHT_DECAY = 2e-4

# # Transform batch size (only inference)
# TRANSFORM_BATCH_SIZE = 4096

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Torch device:", DEVICE)

# # ============================================================
# # CHECK ORIGINAL CEBRA HAS THE REQUIRED MODEL ARCHITECTURE
# # ============================================================
# model_options = list(cebra.models.get_options())
# if MODEL_ARCH not in model_options:
#     raise RuntimeError(
#         f"MODEL_ARCH='{MODEL_ARCH}' is not registered in CEBRA-Orginal.\n"
#         "Do NOT silently replace it, because that would change the experiment.\n"
#         "Register/copy only the model architecture definition into CEBRA-Orginal (or into this script), "
#         "while keeping the original solver code untouched.\n"
#         f"Available model options include: {model_options[:20]}"
#     )

# # ============================================================
# # REPRODUCIBILITY
# # ============================================================
# def seed_all(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def cleanup():
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# seed_all(SEED)

# # ============================================================
# # GAUSSIAN SMOOTHING -- SAME AS BEFORE, NO NORMALIZATION
# # ============================================================
# class GaussianSmoothing(nn.Module):
#     def __init__(self, channels, kernel_size, sigma, dim=1):
#         super().__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim

#         kernel = 1.0
#         meshgrids = torch.meshgrid(
#             [torch.arange(size, dtype=torch.float32) for size in kernel_size],
#             indexing="ij",
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2.0
#             kernel *= (
#                 1.0
#                 / (std * math.sqrt(2.0 * math.pi))
#                 * torch.exp(-0.5 * ((mgrid - mean) / std) ** 2)
#             )
#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
#         self.register_buffer("weight", kernel)
#         self.groups = channels

#     def forward(self, x):
#         # B x T x N -> B x N x T
#         x = x.permute(0, 2, 1)
#         x = F.conv1d(x, weight=self.weight, groups=self.groups, padding="same")
#         # B x N x T -> B x T x N
#         return x.permute(0, 2, 1)


# def smooth_neural(X):
#     print("\n" + "=" * 100)
#     print("GAUSSIAN SMOOTHING")
#     print("=" * 100)
#     print("Input:", X.shape)
#     print("SD:", SMOOTH_SD_MS, "ms")
#     print("Sigma:", SMOOTH_SIGMA_BINS, "bins")
#     print("Kernel:", SMOOTH_KERNEL_SIZE)

#     x_t = torch.from_numpy(X.astype(np.float32, copy=False)).unsqueeze(0)
#     smoother = GaussianSmoothing(
#         channels=X.shape[1],
#         kernel_size=SMOOTH_KERNEL_SIZE,
#         sigma=SMOOTH_SIGMA_BINS,
#         dim=1,
#     )
#     smoother.eval()
#     with torch.no_grad():
#         X_smooth = smoother(x_t).squeeze(0).cpu().numpy().astype(np.float32)

#     print("Smoothed:", X_smooth.shape)
#     print("NO Z-SCORE / NO NORMALIZATION")
#     return X_smooth

# # ============================================================
# # LOAD AREA2
# # ============================================================
# def load_area2():
#     print("\n" + "=" * 100)
#     print("LOADING AREA2 BUMP")
#     print("=" * 100)

#     with h5py.File(NWB_PATH, "r") as f:
#         vel_group = f["processing/behavior/hand_vel"]
#         vel_ds = vel_group["data"]
#         starting_time_ds = vel_group["starting_time"]

#         t_start = float(starting_time_ds[()])
#         behavior_rate = float(starting_time_ds.attrs["rate"])
#         n_behavior_samples = vel_ds.shape[0]

#         samples_per_bin = int(round(behavior_rate * BIN_SEC))
#         n_bins = n_behavior_samples // samples_per_bin
#         usable_samples = n_bins * samples_per_bin
#         t_stop = t_start + n_bins * BIN_SEC

#         vel_raw = np.asarray(vel_ds[:usable_samples], dtype=np.float32)
#         if vel_raw.ndim != 2 or vel_raw.shape[1] != 2:
#             raise RuntimeError(f"Expected hand_vel shape (T,2), got {vel_raw.shape}")

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)
#             Y = np.nanmean(
#                 vel_raw.reshape(n_bins, samples_per_bin, 2), axis=1
#             ).astype(np.float32)

#         units = f["units"]
#         unit_ids = np.asarray(units["id"][:], dtype=np.int64)
#         spike_times = units["spike_times"]
#         spike_index = np.asarray(units["spike_times_index"][:], dtype=np.int64)

#         n_units = len(unit_ids)
#         X_counts = np.zeros((n_bins, n_units), dtype=np.float32)

#         print("Behavior rate:", behavior_rate)
#         print("50-ms bins:", n_bins)
#         print("Neurons:", n_units)
#         print("Binning spikes...")

#         # Same 50 ms binning logic.
#         edges = t_start + np.arange(n_bins + 1, dtype=np.float64) * BIN_SEC
#         for neuron_idx in range(n_units):
#             end_idx = int(spike_index[neuron_idx])
#             start_idx = 0 if neuron_idx == 0 else int(spike_index[neuron_idx - 1])
#             spikes = np.asarray(spike_times[start_idx:end_idx], dtype=np.float64)
#             spikes = spikes[(spikes >= t_start) & (spikes < t_stop)]
#             X_counts[:, neuron_idx] = np.histogram(spikes, bins=edges)[0].astype(np.float32)

#     assert X_counts.shape[0] == Y.shape[0]
#     print("X_counts:", X_counts.shape)
#     print("Y:", Y.shape)
#     print("Y NaNs:", int(np.isnan(Y).sum()))
#     return X_counts, Y, unit_ids

# # ============================================================
# # xCEBRA-STYLE JACOBIAN REGULARIZER + MANUAL ACORN ON ORIGINAL CEBRA
# # ============================================================

# def lambda_jr_at(step: int) -> float:
#     """xCEBRA-style warmup/ramp schedule, advanced once per OUTER iteration."""
#     if step < JR_WARMUP_STEPS:
#         return 0.0
#     frac = (step - JR_WARMUP_STEPS) / float(JR_RAMP_STEPS)
#     return JR_LAMBDA_MAX * min(1.0, max(0.0, frac))


# def _autocast_bf16(x: torch.Tensor):
#     """Match the fork's bfloat16 contrastive/PGD path on CUDA.

#     Jacobian regularization itself is intentionally computed in FP32 because it
#     requires differentiating a gradient (second-order graph).
#     """
#     return torch.autocast(
#         device_type=x.device.type,
#         dtype=torch.bfloat16,
#         enabled=(x.device.type == "cuda"),
#     )


# def _subsample_for_jr(x: torch.Tensor) -> torch.Tensor:
#     if JR_SUBBATCH and x.shape[0] > JR_SUBBATCH:
#         idx = torch.randperm(x.shape[0], device=x.device)[:JR_SUBBATCH]
#         return x[idx]
#     return x


# def _jacobian_reg_fp32(model: nn.Module, reg: nn.Module, x: torch.Tensor):
#     """Compute official CEBRA JacobianReg on x in FP32.

#     This is R_J(x) = 1/2 ||d f(x) / d x||_F^2 (stochastic projection for n=1).
#     The input is detached from any PGD graph: JR regularizes model parameters,
#     but it does NOT alter the attack construction/direction.
#     """
#     xj = _subsample_for_jr(x.detach()).float().requires_grad_(True)
#     # Explicitly disable autocast for the second-order regularizer.
#     with torch.autocast(device_type=xj.device.type, enabled=False):
#         zj = model(xj)
#         jr = reg(xj, zj)
#     return jr


# class RegularizedCleanSolver(SingleSessionSolver):
#     """Original CEBRA single-session solver + xCEBRA Jacobian penalty.

#     One outer iteration:
#         L = InfoNCE(clean) + lambda_t * JR(clean)
#         optimizer.step()
#     """

#     def __post_init__(self):
#         super().__post_init__()
#         self.jacobian_reg = JacobianReg(n=JR_N_PROJ).to(self.device)
#         self.outer_step = 0
#         self.log.setdefault("nce", [])
#         self.log.setdefault("jr", [])
#         self.log.setdefault("lambda_jr", [])
#         self.log.setdefault("objective", [])

#     def step(self, batch):
#         lam = lambda_jr_at(self.outer_step)
#         self.optimizer.zero_grad()

#         # CEBRA contrastive path: bfloat16 on CUDA, as in the adversarial fork.
#         with _autocast_bf16(batch.reference):
#             pred = self._inference(batch)
#             nce_loss, align, uniform = self.criterion(
#                 pred.reference,
#                 pred.positive,
#                 pred.negative,
#             )

#         if lam > 0.0:
#             jr = _jacobian_reg_fp32(
#                 self.model,
#                 self.jacobian_reg,
#                 batch.reference,
#             )
#         else:
#             jr = torch.zeros((), device=batch.reference.device, dtype=torch.float32)

#         objective = nce_loss.float() + lam * jr
#         objective.backward()
#         self.optimizer.step()

#         nce_v = float(nce_loss.detach().float().item())
#         jr_v = float(jr.detach().float().item())
#         obj_v = float(objective.detach().float().item())

#         self.history.append(obj_v)
#         stats = {
#             "pos": float(align.detach().float().item()),
#             "neg": float(uniform.detach().float().item()),
#             "total": obj_v,
#             "temperature": self.criterion.temperature,
#             "nce": nce_v,
#             "jr": jr_v,
#             "lambda_jr": lam,
#         }

#         for k, v in stats.items():
#             self.log.setdefault(k, []).append(v)
#         self.log["objective"].append(obj_v)
#         self.outer_step += 1
#         return stats


# class ACORNAdversarialRegularizedSolver(SingleSessionSolver):
#     """Original CEBRA + manual ACORN + xCEBRA-style Jacobian penalty.

#     IMPORTANT ORDERING (one OUTER iteration):

#       1) Clean regularized update:
#          L_clean = InfoNCE(x) + lambda_t * JR(x)

#       2) Build ACORN PGD adversarial trajectory AFTER the clean update.
#          The attack maximizes PURE InfoNCE only; JR is not part of PGD.

#       3) Adversarial regularized update:
#          L_adv = mean_k InfoNCE(x_adv^(k))
#                  + lambda_t * JR(x_adv^(K))

#          So the attack is generated FIRST, and only AFTER it is finished do we
#          evaluate the Jacobian-Frobenius penalty on the FINAL adversarial input.

#     PGD geometry matches the target ACORN revision:
#       * random Linf start in [-eps, eps]
#       * reference only (positive/negative stay clean)
#       * alpha = eps / 5
#       * 10 sign-gradient steps
#       * projection to the Linf epsilon-ball
#       * NO clamp to [0,1]
#       * two optimizer.step() calls per outer iteration
#       * bfloat16 contrastive/PGD forward path on CUDA
#       * final adversarial InfoNCE aggregated over the full PGD trajectory
#     """

#     def __post_init__(self):
#         super().__post_init__()
#         self.jacobian_reg = JacobianReg(n=JR_N_PROJ).to(self.device)
#         self.outer_step = 0

#         # Assigned after construction by train_representation().
#         self.adv_epsilon = 0.0
#         self.adv_alpha = 0.0
#         self.adv_steps = ADV_STEPS

#         for key in (
#             "clean_nce", "clean_jr", "clean_objective",
#             "adv_nce", "adv_jr", "adv_objective", "lambda_jr",
#         ):
#             self.log.setdefault(key, [])

#     def _clean_regularized_update(self, batch, lam):
#         self.optimizer.zero_grad()

#         with _autocast_bf16(batch.reference):
#             pred = self._inference(batch)
#             nce_loss, align, uniform = self.criterion(
#                 pred.reference,
#                 pred.positive,
#                 pred.negative,
#             )

#         if lam > 0.0:
#             jr = _jacobian_reg_fp32(
#                 self.model,
#                 self.jacobian_reg,
#                 batch.reference,
#             )
#         else:
#             jr = torch.zeros((), device=batch.reference.device, dtype=torch.float32)

#         objective = nce_loss.float() + lam * jr
#         objective.backward()
#         self.optimizer.step()

#         return nce_loss, align, uniform, jr, objective

#     def _build_pgd_trajectory(self, batch):
#         if ATTACK_NORM != "linf":
#             raise NotImplementedError(
#                 "This Area2 experiment is configured for the Linf ACORN attack."
#             )

#         x_base = batch.reference.detach()
#         eps = float(self.adv_epsilon)
#         alpha = float(self.adv_alpha)
#         n_steps = int(self.adv_steps)

#         # Same random start as the fork.
#         x_adv = x_base.clone().detach() + torch.empty_like(x_base).uniform_(-eps, eps)
#         x_adv.requires_grad_(True)

#         adv_traj = []

#         for _ in range(n_steps):
#             adv_batch = cebra.data.Batch(
#                 reference=x_adv,
#                 positive=batch.positive,
#                 negative=batch.negative,
#             )

#             # Same bfloat16 contrastive forward as fork on CUDA.
#             with _autocast_bf16(x_adv):
#                 adv_output = self._inference(adv_batch)
#                 adv_loss = self.criterion(
#                     adv_output.reference,
#                     adv_output.positive,
#                     adv_output.negative,
#                 )[0]

#             (grad_x,) = torch.autograd.grad(
#                 adv_loss,
#                 x_adv,
#                 retain_graph=False,
#                 create_graph=False,
#             )

#             with torch.no_grad():
#                 x_adv = x_adv + alpha * grad_x.sign()
#                 x_adv = torch.max(
#                     torch.min(x_adv, x_base + eps),
#                     x_base - eps,
#                 )

#             x_adv = x_adv.detach().requires_grad_(True)

#             if ADV_AGGREGATE_TRAJECTORY:
#                 # Same position in the loop as the ACORN fork: append AFTER
#                 # each PGD update/projection.
#                 adv_traj.append(x_adv.detach())

#         if ADV_AGGREGATE_TRAJECTORY:
#             if len(adv_traj) != n_steps:
#                 raise RuntimeError(
#                     f"Expected {n_steps} adversarial trajectory points, got {len(adv_traj)}"
#                 )
#         else:
#             adv_traj = [x_adv.detach()]

#         return adv_traj

#     def _adversarial_regularized_update(self, batch, adv_traj, lam):
#         """Attack is already complete here. JR is evaluated only afterwards."""
#         self.optimizer.zero_grad()

#         weight = 1.0 / len(adv_traj)
#         adv_nce_total = None

#         # Faithful ACORN trajectory aggregation for the adversarial InfoNCE term.
#         for x_i in adv_traj:
#             adv_batch = cebra.data.Batch(
#                 reference=x_i,
#                 positive=batch.positive,
#                 negative=batch.negative,
#             )

#             with _autocast_bf16(x_i):
#                 output = self._inference(adv_batch)
#                 loss_i, _, _ = self.criterion(
#                     output.reference,
#                     output.positive,
#                     output.negative,
#                 )

#             term = weight * loss_i.float()
#             adv_nce_total = term if adv_nce_total is None else adv_nce_total + term

#         # FIRST finish PGD; THEN regularize the FINAL adversarial input.
#         x_adv_final = adv_traj[-1]
#         if lam > 0.0:
#             adv_jr = _jacobian_reg_fp32(
#                 self.model,
#                 self.jacobian_reg,
#                 x_adv_final,
#             )
#         else:
#             adv_jr = torch.zeros((), device=x_adv_final.device, dtype=torch.float32)

#         adv_objective = adv_nce_total + lam * adv_jr
#         adv_objective.backward()
#         self.optimizer.step()

#         return adv_nce_total, adv_jr, adv_objective

#     def step(self, batch):
#         lam = lambda_jr_at(self.outer_step)

#         # ----------------------------------------------------
#         # UPDATE 1: CLEAN + JR(clean)
#         # ----------------------------------------------------
#         clean_nce, align, uniform, clean_jr, clean_obj = (
#             self._clean_regularized_update(batch, lam)
#         )

#         # ----------------------------------------------------
#         # ATTACK: AFTER CLEAN UPDATE, PURE InfoNCE PGD
#         # ----------------------------------------------------
#         adv_traj = self._build_pgd_trajectory(batch)

#         # ----------------------------------------------------
#         # UPDATE 2: trajectory-mean adversarial InfoNCE
#         #           + JR(FINAL ADVERSARIAL INPUT)
#         # ----------------------------------------------------
#         adv_nce, adv_jr, adv_obj = self._adversarial_regularized_update(
#             batch,
#             adv_traj,
#             lam,
#         )

#         clean_nce_v = float(clean_nce.detach().float().item())
#         clean_jr_v = float(clean_jr.detach().float().item())
#         clean_obj_v = float(clean_obj.detach().float().item())
#         adv_nce_v = float(adv_nce.detach().float().item())
#         adv_jr_v = float(adv_jr.detach().float().item())
#         adv_obj_v = float(adv_obj.detach().float().item())

#         # Store one history value per OUTER iteration.
#         self.history.append(clean_obj_v + adv_obj_v)

#         stats = {
#             "pos": float(align.detach().float().item()),
#             "neg": float(uniform.detach().float().item()),
#             "total": clean_obj_v,
#             "temperature": self.criterion.temperature,
#             "clean_nce": clean_nce_v,
#             "clean_jr": clean_jr_v,
#             "adv_nce": adv_nce_v,
#             "adv_jr": adv_jr_v,
#             "adv_total": adv_obj_v,
#             "lambda_jr": lam,
#         }

#         for k, v in stats.items():
#             self.log.setdefault(k, []).append(v)
#         self.log["clean_objective"].append(clean_obj_v)
#         self.log["adv_objective"].append(adv_obj_v)

#         self.outer_step += 1
#         return stats

# # ============================================================
# # EPSILON -- RECOMPUTED FOR EACH INPUT SUBSET
# # ============================================================
# def compute_adv_epsilon(X_train):
#     train_tensor = torch.from_numpy(X_train.astype(np.float32, copy=False)).float()
#     eps = float(min_l2_distance(train_tensor)) / 2.0
#     eps = max(eps, 1e-6)
#     print("adv_epsilon:", eps)
#     print("adv_alpha  :", eps / 5.0)
#     return eps

# # ============================================================
# # BUILD + TRAIN ON ORIGINAL CEBRA
# # ============================================================
# def train_representation(X_train, adversarial, name):
#     print("\n" + "#" * 110)
#     print(name)
#     print("#" * 110)
#     print("X_train:", X_train.shape)
#     print("JR lambda max:", JR_LAMBDA_MAX)
#     print("JR warmup/ramp:", JR_WARMUP_STEPS, "/", JR_RAMP_STEPS)
#     print("JR projections:", JR_N_PROJ)
#     print("Adversarial:", adversarial)
#     if adversarial:
#         print("ADV trajectory aggregation:", ADV_AGGREGATE_TRAJECTORY)
#         print("ACORN ADV JR location: FINAL adversarial input AFTER PGD")

#     seed_all(SEED)
#     X_train = X_train.astype(np.float32, copy=False)

#     # --------------------------------------------------------
#     # ORIGINAL CEBRA TensorDataset + time-contrastive loader
#     # --------------------------------------------------------
#     neural_tensor = torch.from_numpy(X_train).float()
    
#     # Low-level TensorDataset in original CEBRA requires at least one
#     # continuous/discrete index.
#     #
#     # IMPORTANT:
#     # This is ONLY a dummy chronological index required by TensorDataset.
#     # Since the loader below uses conditional="time", this index is NOT
#     # used as a behavioral label for positive-pair sampling.
#     time_index = torch.arange(
#         X_train.shape[0],
#         dtype=torch.float32
#     ).unsqueeze(1)
    
#     dataset = cebra.data.TensorDataset(
#         neural=neural_tensor,
#         continuous=time_index,
#     )
    
#     model = cebra.models.init(
#         name=MODEL_ARCH,
#         num_neurons=dataset.input_dimension,
#         num_units=NUM_HIDDEN_UNITS,
#         num_output=LATENT_DIM,
#         normalize=True,
#     ).to(DEVICE)

#     dataset.configure_for(model)
#     dataset.to(DEVICE)

#     loader = cebra.data.single_session.ContinuousDataLoader(
#         dataset=dataset,
#         time_offset=TIME_OFFSETS,
#         num_steps=MAX_ITER,
#         batch_size=BATCH_SIZE,
#         conditional="time",
#     ).to(DEVICE)

#     criterion = cebra.models.criterions.FixedCosineInfoNCE(
#         temperature=TEMPERATURE
#     ).to(DEVICE)

#     optimizer = torch.optim.Adam(
#         list(model.parameters()) + list(criterion.parameters()),
#         lr=LEARNING_RATE,
#         weight_decay=0.0,
#     )

#     if adversarial:
#         eps = compute_adv_epsilon(X_train)
#         solver = ACORNAdversarialRegularizedSolver(
#             model=model,
#             criterion=criterion,
#             optimizer=optimizer,
#             tqdm_on=True,
#         ).to(DEVICE)
#         solver.adv_epsilon = eps
#         solver.adv_alpha = eps / 5.0
#         solver.adv_steps = ADV_STEPS
#     else:
#         solver = RegularizedCleanSolver(
#             model=model,
#             criterion=criterion,
#             optimizer=optimizer,
#             tqdm_on=True,
#         ).to(DEVICE)

#     solver.fit(loader=loader)

#     del loader, dataset
#     cleanup()
#     return solver

# # ============================================================
# # TRANSFORM
# # ============================================================
# def get_embeddings(solver, X):
#     X_t = torch.from_numpy(X.astype(np.float32, copy=False)).float().to(DEVICE)
#     with torch.no_grad():
#         z = solver.transform(
#             X_t,
#             pad_before_transform=True,
#             batch_size=TRANSFORM_BATCH_SIZE,
#         )
#     z = z.detach().cpu().numpy().astype(np.float32)
#     del X_t
#     cleanup()
#     return z

# # ============================================================
# # ATTRIBUTION HELPERS
# # ============================================================
# def _matrix_from_attr(arr, n_neurons, inverse=False):
#     a = np.abs(np.asarray(arr))

#     # Average sample/chunk dimensions until a matrix remains.
#     while a.ndim > 2:
#         a = a.mean(axis=0)

#     if a.ndim != 2:
#         raise RuntimeError(f"Unexpected attribution shape after reduction: {a.shape}")

#     if not inverse:
#         # Want latent x neuron
#         if a.shape == (LATENT_DIM, n_neurons):
#             return a
#         if a.shape == (n_neurons, LATENT_DIM):
#             return a.T
#     else:
#         # Want neuron x latent
#         if a.shape == (n_neurons, LATENT_DIM):
#             return a
#         if a.shape == (LATENT_DIM, n_neurons):
#             return a.T

#     raise RuntimeError(
#         f"Cannot orient attribution matrix {a.shape}; "
#         f"expected latent={LATENT_DIM}, neurons={n_neurons}."
#     )


# def _get_inverse_from_result(result):
#     candidates = [
#         "jf-inv-svd",
#         "jf-inv",
#         "jfinv",
#         "jf-pinv",
#         "jf_pinv",
#         "jf-inv-lsq",
#     ]
#     for key in candidates:
#         if key in result:
#             return result[key], key
#     raise KeyError(
#         "Could not find inverse Jacobian in attribution result. "
#         f"Available keys: {list(result.keys())}"
#     )


# def compute_full_attribution(solver, X_train, tag):
#     print("\n" + "=" * 100)
#     print(f"ATTRIBUTION — {tag}")
#     print("=" * 100)

#     net = solver.model
#     net.eval()
#     if hasattr(net, "split_outputs"):
#         net.split_outputs = False

#     n_neurons = X_train.shape[1]
#     max_start = max(0, len(X_train) - ATTR_CHUNK_LEN)
#     starts = np.linspace(0, max_start, ATTR_N_CHUNKS).astype(int)

#     jf_maps = []
#     jfinv_maps = []

#     for chunk_id, start in enumerate(starts, 1):
#         chunk = X_train[start : start + ATTR_CHUNK_LEN].astype(np.float32, copy=False)
#         inp = (
#             torch.from_numpy(chunk)
#             .float()
#             .to(DEVICE)
#             .detach()
#             .requires_grad_(True)
#         )

#         method = cebra.attribution.init(
#             name="jacobian-based-batched",
#             model=net,
#             input_data=inp,
#             output_dimension=LATENT_DIM,
#         )
#         with torch.enable_grad():
#             result = method.compute_attribution_map(
#                 batch_size=ATTR_BATCH_SIZE
#             )

#         jf = _matrix_from_attr(result["jf"], n_neurons=n_neurons, inverse=False)
#         inv_raw, inv_key = _get_inverse_from_result(result)
#         jfinv = _matrix_from_attr(inv_raw, n_neurons=n_neurons, inverse=True)

#         jf_maps.append(jf)
#         jfinv_maps.append(jfinv)

#         print(
#             f"chunk {chunk_id:02d}/{ATTR_N_CHUNKS} | "
#             f"start={start} | inverse_key={inv_key}"
#         )
#         del inp, method, result
#         cleanup()

#     JF = np.mean(np.stack(jf_maps, axis=0), axis=0)
#     JFINV = np.mean(np.stack(jfinv_maps, axis=0), axis=0)

#     print("JF final    :", JF.shape)
#     print("JFINV final :", JFINV.shape)
#     return JF, JFINV


# def topk_from_maps(JF, JFINV, k):
#     # JF: latent x neuron
#     jf_scores = JF.mean(axis=0)
#     # JFINV: neuron x latent
#     jfinv_scores = JFINV.mean(axis=1)

#     top_jf = np.argsort(jf_scores)[::-1][:k].astype(np.int64)
#     top_jfinv = np.argsort(jfinv_scores)[::-1][:k].astype(np.int64)
#     return top_jf, top_jfinv, jf_scores, jfinv_scores

# # ============================================================
# # SAVE ONLY FULL-MODEL JF / JFINV COMPARISON FIGURES
# # ============================================================
# def save_attribution_plots(clean_jf, acorn_jf, clean_inv, acorn_inv):
#     # JF
#     vmax = max(float(clean_jf.max()), float(acorn_jf.max()))
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
#     im0 = axes[0].imshow(clean_jf, aspect="auto", vmin=0, vmax=vmax)
#     axes[0].set_title("CLEAN + JR — Forward Jacobian")
#     axes[0].set_xlabel("Neuron")
#     axes[0].set_ylabel("Latent dimension")
#     axes[1].imshow(acorn_jf, aspect="auto", vmin=0, vmax=vmax)
#     axes[1].set_title("ACORN + JR — Forward Jacobian")
#     axes[1].set_xlabel("Neuron")
#     axes[1].set_ylabel("Latent dimension")
#     fig.colorbar(im0, ax=axes, shrink=0.85)
#     jf_path = os.path.join(OUT, "JF_REG_CLEAN_vs_REG_ACORN.png")
#     fig.savefig(jf_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     # JFINV
#     vmax = max(float(clean_inv.max()), float(acorn_inv.max()))
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
#     im0 = axes[0].imshow(clean_inv, aspect="auto", vmin=0, vmax=vmax)
#     axes[0].set_title("CLEAN + JR — Inverse Jacobian")
#     axes[0].set_xlabel("Latent dimension")
#     axes[0].set_ylabel("Neuron")
#     axes[1].imshow(acorn_inv, aspect="auto", vmin=0, vmax=vmax)
#     axes[1].set_title("ACORN + JR — Inverse Jacobian")
#     axes[1].set_xlabel("Latent dimension")
#     axes[1].set_ylabel("Neuron")
#     fig.colorbar(im0, ax=axes, shrink=0.85)
#     inv_path = os.path.join(OUT, "JFINV_REG_CLEAN_vs_REG_ACORN.png")
#     fig.savefig(inv_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     print("Saved:", jf_path)
#     print("Saved:", inv_path)

# # ============================================================
# # DECODER
# # ============================================================
# class TwoLayerMLP(nn.Module):
#     def __init__(
#         self,
#         input_dim=128,
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
#                 nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
#                 if layer.bias is not None:
#                     nn.init.constant_(layer.bias, 0)

#     def forward(self, x):
#         return self.net(x)


# def train_decoder(z_train_np, y_train_np, z_test_np, y_test_np, model_name):
#     train_mask = np.isfinite(z_train_np).all(axis=1) & np.isfinite(y_train_np).all(axis=1)
#     test_mask = np.isfinite(z_test_np).all(axis=1) & np.isfinite(y_test_np).all(axis=1)

#     z_train_np = z_train_np[train_mask]
#     y_train_np = y_train_np[train_mask]
#     z_test_np = z_test_np[test_mask]
#     y_test_np = y_test_np[test_mask]

#     assert len(z_train_np) > 0 and len(z_test_np) > 0

#     seed_all(SEED)
#     decoder = TwoLayerMLP(
#         input_dim=z_train_np.shape[1],
#         hidden_dim=DECODER_HIDDEN,
#         output_dim=2,
#         dropout_rate=DECODER_DROPOUT,
#     ).to(DEVICE)

#     z_train = torch.from_numpy(z_train_np).float().to(DEVICE)
#     y_train = torch.from_numpy(y_train_np).float().to(DEVICE)
#     z_test = torch.from_numpy(z_test_np).float().to(DEVICE)
#     y_test = torch.from_numpy(y_test_np).float().to(DEVICE)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(
#         decoder.parameters(),
#         lr=DECODER_LR,
#         weight_decay=DECODER_WEIGHT_DECAY,
#     )

#     print("\n" + "=" * 100)
#     print("DECODER —", model_name)
#     print("=" * 100)

#     for epoch in range(DECODER_EPOCHS):
#         decoder.train()
#         optimizer.zero_grad()
#         pred = decoder(z_train)
#         loss = criterion(pred, y_train)
#         loss.backward()
#         optimizer.step()

#         if epoch == 0 or (epoch + 1) % 1000 == 0:
#             print(
#                 f"{model_name} | Epoch {epoch+1}/{DECODER_EPOCHS} | "
#                 f"train MSE={loss.item():.8f}"
#             )

#     decoder.eval()
#     with torch.no_grad():
#         pred = decoder(z_test).cpu().numpy()
#         true = y_test.cpu().numpy()

#     mse = float(np.mean((true - pred) ** 2))
#     r2_vx = float(r2_score(true[:, 0], pred[:, 0]))
#     r2_vy = float(r2_score(true[:, 1], pred[:, 1]))
#     mean_r2 = float((r2_vx + r2_vy) / 2.0)

#     print(
#         f"{model_name} | MSE={mse:.6f} | "
#         f"R2 vx={r2_vx:.6f} | R2 vy={r2_vy:.6f} | Mean R2={mean_r2:.6f}"
#     )

#     del decoder, optimizer, z_train, y_train, z_test, y_test
#     cleanup()

#     return {
#         "mse": mse,
#         "r2_vx": r2_vx,
#         "r2_vy": r2_vy,
#         "mean_r2": mean_r2,
#     }

# # ============================================================
# # RUN ONE REDUCED CONDITION
# # ============================================================
# def run_reduced_condition(
#     selector_name,
#     neuron_indices,
#     X_train,
#     X_test,
#     Y_train,
#     Y_test,
#     unit_ids,
# ):
#     neuron_indices = np.asarray(neuron_indices, dtype=np.int64)
#     Xtr = X_train[:, neuron_indices].astype(np.float32, copy=False)
#     Xte = X_test[:, neuron_indices].astype(np.float32, copy=False)

#     print("\n" + "#" * 110)
#     print("SELECTOR:", selector_name)
#     print("Indices :", neuron_indices.tolist())
#     print("Unit IDs:", unit_ids[neuron_indices].tolist())
#     print("#" * 110)

#     rows = []
#     for adversarial in (False, True):
#         retrained_name = "ACORN" if adversarial else "CEBRA"
#         condition = f"{selector_name}__{retrained_name}"

#         solver = train_representation(
#             Xtr,
#             adversarial=adversarial,
#             name=condition + " + JR",
#         )

#         z_train = get_embeddings(solver, Xtr)
#         z_test = get_embeddings(solver, Xte)

#         metrics = train_decoder(
#             z_train,
#             Y_train,
#             z_test,
#             Y_test,
#             model_name=condition,
#         )

#         rows.append(
#             {
#                 "condition": condition,
#                 "selector": selector_name,
#                 "retrained_model": retrained_name,
#                 "n_neurons": len(neuron_indices),
#                 "neuron_indices": ",".join(map(str, neuron_indices.tolist())),
#                 "unit_ids": ",".join(map(str, unit_ids[neuron_indices].tolist())),
#                 "mse": metrics["mse"],
#                 "r2_vx": metrics["r2_vx"],
#                 "r2_vy": metrics["r2_vy"],
#                 "mean_r2": metrics["mean_r2"],
#             }
#         )

#         del solver, z_train, z_test
#         cleanup()

#     return rows

# # ============================================================
# # MAIN
# # ============================================================
# def main():
#     print("\n" + "=" * 110)
#     print("AREA2 BUMP — TOP-K | ORIGINAL CEBRA | JR + FAITHFUL ACORN")
#     print("=" * 110)
#     print("Original CEBRA repo:", CEBRA_ORIGINAL_DIR)
#     print("JR lambda max:", JR_LAMBDA_MAX)
#     print("Full CLEAN: original CEBRA + explicit xCEBRA-style JR")
#     print("Full ACORN: original CEBRA + faithful manual ACORN + JR(final adv)")
#     print("All 8 reduced models: same regularized CLEAN/ACORN objectives")
#     print("Decoder epochs:", DECODER_EPOCHS)

#     # --------------------------------------------------------
#     # 1) LOAD + PREPROCESS
#     # --------------------------------------------------------
#     X_counts, Y, unit_ids = load_area2()
#     X = smooth_neural(X_counts)
#     del X_counts
#     cleanup()

#     split_idx = int(TRAIN_FRAC * len(X))
#     X_train = X[:split_idx].astype(np.float32, copy=False)
#     X_test = X[split_idx:].astype(np.float32, copy=False)
#     Y_train = Y[:split_idx].astype(np.float32, copy=False)
#     Y_test = Y[split_idx:].astype(np.float32, copy=False)

#     print("\nTemporal 80/20 split")
#     print("X_train:", X_train.shape)
#     print("X_test :", X_test.shape)
#     print("Y_train:", Y_train.shape)
#     print("Y_test :", Y_test.shape)

#     n_neurons = X_train.shape[1]
#     k = max(1, min(n_neurons, int(np.sqrt(n_neurons))))
#     print("\nTop-K = floor(sqrt(N))")
#     print("N =", n_neurons, "K =", k)

#     # --------------------------------------------------------
#     # 2) FULL REGULARIZED CLEAN
#     # --------------------------------------------------------
#     clean_full = train_representation(
#         X_train,
#         adversarial=False,
#         name="FULL CLEAN + JR",
#     )

#     clean_jf, clean_inv = compute_full_attribution(
#         clean_full,
#         X_train,
#         tag="FULL CLEAN + JR",
#     )
#     clean_top_jf, clean_top_inv, clean_jf_scores, clean_inv_scores = topk_from_maps(
#         clean_jf, clean_inv, k
#     )

#     print("\nCLEAN + JR Top-JF    :", clean_top_jf.tolist())
#     print("CLEAN + JR Top-JFINV :", clean_top_inv.tolist())

#     # --------------------------------------------------------
#     # 3) FULL REGULARIZED ACORN
#     # --------------------------------------------------------
#     acorn_full = train_representation(
#         X_train,
#         adversarial=True,
#         name="FULL ACORN + JR",
#     )

#     acorn_jf, acorn_inv = compute_full_attribution(
#         acorn_full,
#         X_train,
#         tag="FULL ACORN + JR",
#     )
#     acorn_top_jf, acorn_top_inv, acorn_jf_scores, acorn_inv_scores = topk_from_maps(
#         acorn_jf, acorn_inv, k
#     )

#     print("\nACORN + JR Top-JF    :", acorn_top_jf.tolist())
#     print("ACORN + JR Top-JFINV :", acorn_top_inv.tolist())

#     # Save only the two full-model attribution comparison plots.
#     save_attribution_plots(
#         clean_jf,
#         acorn_jf,
#         clean_inv,
#         acorn_inv,
#     )

#     # Save selector indices for reproducibility.
#     selector_rows = []
#     for selector, inds in {
#         "CLEAN_topJF": clean_top_jf,
#         "ACORN_topJF": acorn_top_jf,
#         "CLEAN_topJFINV": clean_top_inv,
#         "ACORN_topJFINV": acorn_top_inv,
#     }.items():
#         selector_rows.append(
#             {
#                 "selector": selector,
#                 "k": k,
#                 "neuron_indices": ",".join(map(str, inds.tolist())),
#                 "unit_ids": ",".join(map(str, unit_ids[inds].tolist())),
#             }
#         )
#     pd.DataFrame(selector_rows).to_csv(
#         os.path.join(OUT, "TopK8_REG_selectors.csv"), index=False
#     )

#     # Full solvers are no longer needed after attribution / selection.
#     del clean_full, acorn_full
#     cleanup()

#     # --------------------------------------------------------
#     # 4) FOUR SELECTORS x TWO REGULARIZED RETRAINED MODELS
#     #    = EIGHT REDUCED CONDITIONS
#     # --------------------------------------------------------
#     reduced_sets = {
#         "CLEAN_topJF": clean_top_jf,
#         "ACORN_topJF": acorn_top_jf,
#         "CLEAN_topJFINV": clean_top_inv,
#         "ACORN_topJFINV": acorn_top_inv,
#     }

#     all_rows = []
#     for selector_name, inds in reduced_sets.items():
#         all_rows.extend(
#             run_reduced_condition(
#                 selector_name=selector_name,
#                 neuron_indices=inds,
#                 X_train=X_train,
#                 X_test=X_test,
#                 Y_train=Y_train,
#                 Y_test=Y_test,
#                 unit_ids=unit_ids,
#             )
#         )

#     # --------------------------------------------------------
#     # 5) FINAL TABLE + CSV
#     # --------------------------------------------------------
#     df = pd.DataFrame(all_rows)
#     result_csv = os.path.join(OUT, "TopK8_REG_8conditions.csv")
#     df.to_csv(result_csv, index=False, float_format="%.8f")

#     print("\n\n" + "=" * 116)
#     print("FINAL 8 REDUCED CONDITIONS — ALL WITH JACOBIAN REGULARIZATION")
#     print("=" * 116)
#     print(
#         f"{'CONDITION':44s} {'N':>4s} {'MSE':>12s} "
#         f"{'R2 vx':>12s} {'R2 vy':>12s} {'Mean R2':>12s}"
#     )
#     print("-" * 116)

#     for row in all_rows:
#         print(
#             f"{row['condition']:44s} "
#             f"{row['n_neurons']:4d} "
#             f"{row['mse']:12.6f} "
#             f"{row['r2_vx']:12.6f} "
#             f"{row['r2_vy']:12.6f} "
#             f"{row['mean_r2']:12.6f}"
#         )

#     best = max(all_rows, key=lambda r: r["mean_r2"])
#     print("\nBEST REDUCED CONDITION:")
#     print(best["condition"], "Mean R2 =", f"{best['mean_r2']:.6f}")

#     print("\nSAVED:")
#     print(os.path.join(OUT, "JF_REG_CLEAN_vs_REG_ACORN.png"))
#     print(os.path.join(OUT, "JFINV_REG_CLEAN_vs_REG_ACORN.png"))
#     print(os.path.join(OUT, "TopK8_REG_selectors.csv"))
#     print(result_csv)


# if __name__ == "__main__":
#     main()
