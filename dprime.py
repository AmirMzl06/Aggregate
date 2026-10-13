"""C-CO12: compare raw PGD (L2 / Linf, epsilon=3) against Acorn-dprime ideas.

Place NEXT TO the folder Acorn-dprime and run:
    python run_CCO12_Acorn_dprime_all.py

By default the comprehensive suite runs. Optional:
    python run_CCO12_Acorn_dprime_all.py --suite core
    python run_CCO12_Acorn_dprime_all.py --list
    python run_CCO12_Acorn_dprime_all.py --only DPRIME_AUTO_INSTANCE DPRIME_TRIAL_MASK
    python run_CCO12_Acorn_dprime_all.py --report-eps

The results are printed as each run finishes. No models, checkpoints, CSVs,
plots, predictions or other result files are saved by this runner.

IMPORTANT: Acorn-dprime implements d' attacks using a WHITENED L2 budget.
The `attack_norm` parameter is NOT USED in d' mode. Therefore it is wrong
(and redundant) to run d' as if it supported both ordinary L2 and Linf.
Both norms ARE exercised for the original PGD epsilon=3 baselines.

The encoder is fitted on x_train ALONE (time-only). Behavioral labels are
used only by the decoder, unlike the reference SVD runner which passed them
into CEBRA.fit(). This is intentional to avoid behavioral-label leakage.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import gc
import inspect
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True

import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
DEFAULT_FORK_DIR = ROOT / "Acorn-dprime"
DEFAULT_DATA_DIR = Path("/data/hossein/mm_project/perich_data_valid_final_raw/")
DEFAULT_SESSION = "C-CO12"

SEED = 42
PGD_EPS = 3.0
ATTACK_STEPS = 10
PGD_ALPHA = PGD_EPS / 5.0

LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
ARCH = "offset36-model-more-dropout"
DEVICE = "cuda_if_available"

MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192


@dataclass(frozen=True)
class Experiment:
    name: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    core: bool = False


def experiments() -> List[Experiment]:
    """One-factor-at-a-time comparisons and a few faithful combined methods.

    Fixed d' epsilon=3 is NOT equivalent to raw L2/Linf epsilon=3: the units
    and feasible sets differ. Auto-epsilon runs include detector calibration.
    """
    base = [
        Experiment("PGD_L2_EPS3", {"adv_epsilon_mode": "constant", "attack_norm": "l2"}, True),
        Experiment("PGD_LINF_EPS3", {"adv_epsilon_mode": "constant", "attack_norm": "linf"}, True),
    ]

    # d' defaults: stationary, APGD, Madry, auto epsilon, instance-adaptive,
    # Ledoit-Wolf + Welch, guards off, time-only calibration, 1 restart.
    d = dict(
        adv_epsilon_mode="dprime",
        adv_dprime_layer="stationary",
        adv_dprime_epsilon="auto",
        adv_dprime_jnd=1.0,
        adv_dprime_residuals="whole",  # No behavior labels anywhere in CEBRA.
        adv_dprime_spatial="ledoit_wolf",
        adv_dprime_temporal="welch",
        adv_dprime_optimizer="apgd",
        adv_dprime_restarts=1,
        adv_dprime_objective="madry",
        adv_dprime_instance_adaptive=True,
        adv_dprime_guards=(),
        adv_dprime_guard_mode="off",
        adv_dprime_warmup=0.10,
        adv_dprime_curriculum=0.50,
    )

    def arm(name: str, core: bool = False, **kw: Any) -> Experiment:
        return Experiment(name, {**d, **kw}, core)

    return base + [
        # Algorithm and step adaptation, with the same fixed whitened L2 budget.
        arm("DPRIME_STATIONARY_PGD_FIXED3", True,
            adv_dprime_epsilon=3.0, adv_dprime_optimizer="pgd",
            adv_dprime_instance_adaptive=False),
        arm("DPRIME_STATIONARY_APGD_FIXED3", True,
            adv_dprime_epsilon=3.0, adv_dprime_instance_adaptive=False),
        arm("DPRIME_APGD_RESTARTS3_FIXED3",
            adv_dprime_epsilon=3.0, adv_dprime_instance_adaptive=False,
            adv_dprime_restarts=3),
        # The full 3-criterion (JND/NN/detector) epsilon protocol.
        arm("DPRIME_AUTO_GLOBAL", True, adv_dprime_instance_adaptive=False),
        arm("DPRIME_AUTO_INSTANCE", True),
        arm("DPRIME_AUTO_INSTANCE_NO_CURRICULUM", adv_dprime_curriculum=0.0),
        # TRADES is a DIFFERENT training objective than the original Madry AT.
        arm("DPRIME_TRADES_AUTO_INSTANCE", adv_dprime_objective="trades"),
        # Spatial and temporal ablations of stationary NoiseModel.
        arm("DPRIME_SPATIAL_OAS", adv_dprime_spatial="oas"),
        arm("DPRIME_SPATIAL_TYLER", adv_dprime_spatial="tyler"),
        arm("DPRIME_SPATIAL_EMPIRICAL", adv_dprime_spatial="empirical"),
        arm("DPRIME_TEMPORAL_AR4", adv_dprime_temporal="ar", adv_dprime_ar_order=4),
        arm("DPRIME_TEMPORAL_WHITE", adv_dprime_temporal="white"),
        arm("DPRIME_TEMPORAL_VANROSSUM", True,
            adv_dprime_temporal="vanrossum", adv_dprime_vanrossum_tau="auto"),
        # Nonstationary, trial-specific time-frequency coloring (still whitened L2).
        arm("DPRIME_TRIAL_MASK", True, adv_dprime_layer="trial_mask"),
        arm("DPRIME_TRIAL_MASK_PGD", adv_dprime_layer="trial_mask",
            adv_dprime_optimizer="pgd"),
        # Layer 3, evaluated separately from the basic metric for ablations.
        arm("DPRIME_GUARD_RIEMANN", adv_dprime_guards=("riemann",),
            adv_dprime_guard_mode="backtrack"),
        arm("DPRIME_GUARD_ARTIFACT", adv_dprime_guards=("artifact",),
            adv_dprime_guard_mode="backtrack"),
        arm("DPRIME_GUARD_BAND_POWER", adv_dprime_guards=("band_power",),
            adv_dprime_guard_mode="backtrack"),
        arm("DPRIME_GUARD_APERIODIC", adv_dprime_guards=("aperiodic",),
            adv_dprime_guard_mode="backtrack"),
        arm("DPRIME_GUARDS_ALL", adv_dprime_guards=("band_power", "aperiodic", "riemann", "artifact"),
            adv_dprime_guard_mode="backtrack"),
        arm("DPRIME_GUARDS_FINAL_ONLY", adv_dprime_guards=("riemann",),
            adv_dprime_guard_mode="final"),
        arm("DPRIME_GUARD_SOFT_RIEMANN", adv_dprime_guards=("riemann",),
            adv_dprime_guard_mode="backtrack", adv_dprime_soft_riemann=1.0),
    ]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_fork(fork_dir: Path):
    package = fork_dir / "cebra" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(
            f"Expected fork at {fork_dir}. Place this runner next to Acorn-dprime "
            "or set --fork-dir PATH."
        )
    for key in list(sys.modules):
        if key == "cebra" or key.startswith("cebra."):
            del sys.modules[key]
    sys.path.insert(0, str(fork_dir.resolve()))
    import cebra

    actual = Path(cebra.__file__).resolve()
    if not actual.is_relative_to(fork_dir.resolve()):
        raise RuntimeError(f"Wrong CEBRA imported: {actual}")
    params = inspect.signature(cebra.CEBRA.__init__).parameters
    required = {
        "training_mode", "adv_epsilon", "adv_alpha", "adv_steps", "attack_norm",
        "adv_epsilon_mode", "adv_dprime_epsilon", "adv_dprime_layer",
        "adv_dprime_optimizer", "adv_dprime_instance_adaptive", "adv_dprime_temporal",
        "adv_dprime_spatial", "adv_dprime_guards", "adv_dprime_objective",
    }
    missing = sorted(required - set(params))
    if missing:
        raise RuntimeError(f"This fork lacks Acorn-dprime parameters: {missing}")
    return cebra


def load_data(data_path: Path):
    if not data_path.is_file():
        raise FileNotFoundError(f"No session file at: {data_path}")
    with np.load(data_path, allow_pickle=False) as f:
        x_train, x_valid, y_train, y_valid = (
            np.asarray(f[k], dtype=np.float32)
            for k in ("train_data", "valid_data", "train_label", "valid_label")
        )

    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    for key, arr in (("train_data", x_train), ("valid_data", x_valid),
                     ("train_label", y_train), ("valid_label", y_valid)):
        if arr.ndim != 2 or 0 in arr.shape or not np.isfinite(arr).all():
            raise ValueError(f"Invalid shape / nonfinite {key}: {arr.shape}")
    if (x_train.shape[1] != x_valid.shape[1]
            or y_train.shape[1] != y_valid.shape[1]):
        raise ValueError("Train/valid feature or label dimensions differ")
    if len(x_train) != len(y_train) or len(x_valid) != len(y_valid):
        raise ValueError("Neural data and decoder labels have different lengths")
    if min(len(x_train), len(x_valid)) < 36:
        raise ValueError("The offset36 model needs at least 36 bins in each split")
    for name, x in (("train_data", x_train), ("valid_data", x_valid)):
        if np.any(x < 0) or not np.array_equal(x, np.rint(x)):
            raise ValueError(f"{name} must be RAW nonnegative integer spike counts")
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))


def encoder_config(exp: Experiment, iterations: int, batch_size: int,
                   device: str, seed: int, pgd_eps: float) -> Dict[str, Any]:
    overrides = dict(exp.overrides)
    return {
        "model_architecture": ARCH,
        "batch_size": batch_size,
        "temperature": TEMPERATURE,
        "time_offsets": 1,
        "max_iterations": iterations,
        "output_dimension": LATENT_DIM,
        "num_hidden_units": HIDDEN,
        "learning_rate": 3e-4,
        "temperature_mode": "constant",
        "training_mode": "adversarial",
        "adv_epsilon": pgd_eps,
        "adv_alpha": pgd_eps / 5.0,
        "adv_steps": ATTACK_STEPS,
        "attack_norm": "l2",   # A placeholder ignored by d' training.
        "adv_dprime_seed": seed,
        "device": device,
        "verbose": False,
        **overrides,
    }


class TwoLayerMLP(nn.Module):
    def __init__(self, dim: int, out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, MLP_HIDDEN),
            nn.LayerNorm(MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(MLP_DROP),
            nn.Linear(MLP_HIDDEN, out),
        )

    def forward(self, x):
        return self.net(x)


def train_decoder(z_train, y_train, seed: int, device, epochs: int):
    seed_all(seed)
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    criterion = nn.MSELoss()
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    decoder.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = criterion(decoder(z), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}")
        loss.backward()
        opt.step()
    return decoder


def predict_decoder(decoder, embeddings):
    decoder.eval()
    device = next(decoder.parameters()).device
    preds = []
    with torch.inference_mode():
        for first in range(0, len(embeddings), EVAL_BATCH_SIZE):
            z = torch.as_tensor(embeddings[first:first + EVAL_BATCH_SIZE],
                                dtype=torch.float32, device=device)
            preds.append(decoder(z).cpu().numpy())
    return np.concatenate(preds, axis=0)


def epsilon_summary(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "no-dprime-report"
    e = report.get("epsilon", {})
    nn = report.get("eps_nn", {})
    t = report.get("training", {})
    return (f"eps_star={e.get('eps_star')} "
            f"eps_NN={nn.get('value')} "
            f"mean_dprime={t.get('mean_dprime')} "
            f"NN_binding={t.get('eps_nn_binding_rate')} "
            f"guard_accept={t.get('accept_rate')}")


def run_one(cebra, exp: Experiment, arrays, args) -> Tuple[float, Optional[Dict[str, Any]]]:
    x_train, x_valid, y_train, y_valid = arrays
    encoder = decoder = z_train = z_valid = None
    report = None
    try:
        seed_all(args.seed)
        encoder = cebra.CEBRA(**encoder_config(exp, args.max_iterations, args.batch_size,
                                               args.device, args.seed, args.epsilon))
        # TIME-ONLY training: y_train is deliberately NOT passed here.
        encoder.fit(x_train)
        report = getattr(encoder, "dprime_report_", None)
        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        if (len(z_train) != len(x_train) or len(z_valid) != len(x_valid)
                or not np.isfinite(z_train).all() or not np.isfinite(z_valid).all()):
            raise ValueError(f"Bad embedding shape / alignment for {exp.name}")
        decoder = train_decoder(z_train, y_train, args.seed + 100_000,
                                torch.device(encoder.device_), args.mlp_epochs)
        predictions = predict_decoder(decoder, z_valid)
        if predictions.shape != y_valid.shape or not np.isfinite(predictions).all():
            raise ValueError(f"Bad decoder predictions for {exp.name}")
        value = float(np.atleast_1d(r2_score(y_valid, predictions,
                                             multioutput="raw_values")).mean())
        if not np.isfinite(value):
            raise ValueError("R2 is nonfinite; check validation label variability")
        return value, report
    finally:
        del encoder, decoder, z_train, z_valid
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fork-dir", type=Path, default=DEFAULT_FORK_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--session", default=DEFAULT_SESSION)
    p.add_argument("--suite", choices=("all", "core"), default="all")
    p.add_argument("--only", nargs="+", help="Run named arms; use --list to inspect names")
    p.add_argument("--list", action="store_true", help="List all experiments and exit")
    p.add_argument("--dry-run", action="store_true", help="List selected arms without fitting")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epsilon", type=float, default=PGD_EPS,
                   help="Epsilon for the ORIGINAL L2/Linf PGD only (also sets alpha=epsilon/5)")
    p.add_argument("--max-iterations", type=int, default=MAX_ITER)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--mlp-epochs", type=int, default=MLP_EPOCHS)
    p.add_argument("--device", default=DEVICE)
    p.add_argument("--report-eps", action="store_true",
                   help="Print dprime calibration diagnostics to stderr after each run")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep running remaining arms if an individual arm fails")
    return p


def main():
    args = build_parser().parse_args()
    all_exp = experiments()
    names = {e.name for e in all_exp}
    if args.list:
        for e in all_exp:
            print(f"{e.name} [{'core' if e.core else 'all'}]")
        return 0
    if args.only:
        unknown = sorted(set(args.only) - names)
        if unknown:
            raise ValueError(f"Unknown experiments: {unknown}; inspect --list")
        selected = [e for e in all_exp if e.name in set(args.only)]
    elif args.suite == "core":
        selected = [e for e in all_exp if e.core]
    else:
        selected = all_exp

    if args.dry_run:
        for e in selected:
            print(e.name)
        return 0
    if args.max_iterations < 1 or args.mlp_epochs < 1 or args.batch_size < 1:
        raise ValueError("Iterations, MLP epochs and batch size must be >= 1")
    if not np.isfinite(args.epsilon) or args.epsilon < 0:
        raise ValueError("PGD epsilon must be finite and nonnegative")

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            cebra = load_fork(args.fork_dir)
            arrays = load_data(args.data_dir / f"{args.session}.npz")

    failed = 0
    for exp in selected:
        try:
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    r2, report = run_one(cebra, exp, arrays, args)
            print(f"{exp.name}: R2 = {r2:.6f}", flush=True)
            if args.report_eps and exp.overrides.get("adv_epsilon_mode") == "dprime":
                print(f"  [{exp.name}] {epsilon_summary(report)}", file=sys.stderr, flush=True)
        except Exception as exc:
            failed += 1
            print(f"{exp.name}: ERROR ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
