"""Standalone C-CO12 CEBRA+LABEL experiment runner.
Uses train_data/valid_data and ALL train_label/valid_label columns from NPZ.
No resplitting or normalization. Reports CLEAN-INPUT decoding R2, not robust R2.
The attack implementations must already exist in the selected fork.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import gc
import inspect
import json
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
PERICH_DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw/')
SESSION = 'C-CO12'
NPZ_PATH = PERICH_DATA_DIR / f'{SESSION}.npz'
LATENT_DIM = 64
HIDDEN = 64
BATCH_SIZE = 2048
MAX_ITER = 3000
TEMPERATURE = 0.4
MODEL_ARCH = 'offset36-model-more-dropout'
TIME_OFFSETS = 1
LEARNING_RATE = 3e-4
DEVICE = 'cuda_if_available'
ADV_STEPS = 10
ATTACK_NORM = 'linf'
CONSTANT_EPSILON = 5.0  # Preserved from your supplied runner.
ADV_ALPHA = 1.0        # Used by the ordinary additive PGD baseline.
ADV_ALPHA_RATIO = 0.2  # Fraction of local budget; aligns with 1 / 5 above.
SEEDS = (42, 43)
SAVE_CHECKPOINTS = True
MLP_EPOCHS = 2500
MLP_HIDDEN = 64
MLP_DROP = 0.4
MLP_LR = 1e-3
EVAL_BATCH_SIZE = 8192

CEBRA_DIR = ROOT / 'Acorn-structured-attack'
OUT_ROOT = ROOT / 'ACORN_STRUCTURED_ATTACK_CCO12'
GAIN_EPSILON = 0.1
BASELINE_EPSILON = 0.5
BASELINE_SCALE = 'std'
BASELINE_FLOOR = 0.1
DEFAULT_ARMS = ('clean', 'constant', 'gain_only', 'baseline_only', 'gain_baseline')
AVAILABLE_ARMS = DEFAULT_ARMS + ('gain_baseline_shared',)


def idea_config(arm, args):
    if arm == 'clean':
        return {}
    return dict(
        adv_attack_type='additive' if arm == 'constant' else 'structured',
        adv_gain_epsilon=0.0 if arm == 'baseline_only' else args.gain_epsilon,
        adv_baseline_epsilon=0.0 if arm == 'gain_only' else args.baseline_epsilon,
        adv_baseline_scale=args.baseline_scale,
        adv_baseline_floor=args.baseline_floor,
        adv_structured_shared=(arm == 'gain_baseline_shared'),
        adv_alpha_ratio=args.alpha_ratio,
    )


def add_idea_arguments(parser):
    parser.add_argument('--gain-epsilon', type=float, default=GAIN_EPSILON)
    parser.add_argument('--baseline-epsilon', type=float, default=BASELINE_EPSILON)
    parser.add_argument('--baseline-scale', choices=['std', 'none'], default=BASELINE_SCALE)
    parser.add_argument('--baseline-floor', type=float, default=BASELINE_FLOOR)


def validate_idea(args):
    vals = [args.gain_epsilon, args.baseline_epsilon, args.baseline_floor]
    if not np.isfinite(vals).all() or min(vals) < 0:
        raise ValueError('Gain/baseline budgets and floor must be finite and nonnegative.')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_data():
    if not NPZ_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {NPZ_PATH}")
    with np.load(NPZ_PATH, allow_pickle=False) as data:
        arrays = [
            np.asarray(data[key], dtype=np.float32)
            for key in ("train_data", "valid_data", "train_label", "valid_label")
        ]
    x_train, x_valid, y_train, y_valid = arrays
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    if y_valid.ndim == 1:
        y_valid = y_valid[:, None]
    for name, values in zip(
        ("X_train", "X_valid", "Y_train", "Y_valid"),
        (x_train, x_valid, y_train, y_valid),
    ):
        if values.ndim != 2 or min(values.shape) == 0:
            raise ValueError(f"{name} must be a nonempty 2D array; got {values.shape}.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
    if len(x_train) != len(y_train):
        raise ValueError(
            f"Training feature/label lengths differ: {len(x_train)} vs {len(y_train)}."
        )
    if len(x_valid) != len(y_valid):
        raise ValueError(
            f"Validation feature/label lengths differ: {len(x_valid)} vs {len(y_valid)}."
        )
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError("Training and validation neuron counts do not match.")
    if y_train.shape[1] != y_valid.shape[1]:
        raise ValueError("Training and validation label dimensions do not match.")
    if len(x_train) < 36 or len(x_valid) < 36:
        raise ValueError("The offset36 encoder requires at least 36 timepoints per split.")
    return tuple(np.ascontiguousarray(a) for a in (x_train, x_valid, y_train, y_valid))

class TwoLayerMLP(nn.Module):
    def __init__(self, dim, out):
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

def train_decoder(z_train, y_train, seed, device):
    seed_all(seed)
    decoder = TwoLayerMLP(z_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=MLP_LR)
    criterion = nn.MSELoss()
    z = torch.as_tensor(z_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    losses = []
    decoder.train()
    for epoch in range(MLP_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(decoder(z), y)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Nonfinite decoder loss at epoch {epoch + 1}.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
        if (epoch + 1) % 500 == 0 or epoch + 1 == MLP_EPOCHS:
            print(f"Decoder {epoch + 1}/{MLP_EPOCHS}: train MSE={losses[-1]:.6f}", flush=True)
    return decoder, losses

def predict_decoder(decoder, embeddings):
    decoder.eval()
    device = next(decoder.parameters()).device
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(embeddings), EVAL_BATCH_SIZE):
            z = torch.as_tensor(
                embeddings[start:start + EVAL_BATCH_SIZE],
                dtype=torch.float32,
                device=device,
            )
            predictions.append(decoder(z).cpu().numpy())
    return np.concatenate(predictions, axis=0)

def compute_r2(labels, predictions):
    if predictions.shape != labels.shape:
        raise ValueError(f"Prediction/label shape mismatch: {predictions.shape} vs {labels.shape}.")
    if not np.isfinite(predictions).all():
        raise ValueError("Decoder predictions contain NaN/Inf.")
    per_output = np.atleast_1d(r2_score(labels, predictions, multioutput="raw_values"))
    return float(per_output.mean()), per_output.tolist()

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', default=SESSION)
    parser.add_argument('--data-dir', type=Path, default=PERICH_DATA_DIR)
    parser.add_argument('--cebra-dir', type=Path, default=CEBRA_DIR)
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--arms', nargs='+', choices=AVAILABLE_ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    parser.add_argument('--max-iter', type=int, default=MAX_ITER)
    parser.add_argument('--decoder-epochs', type=int, default=MLP_EPOCHS)
    parser.add_argument('--epsilon', type=float, default=CONSTANT_EPSILON)
    parser.add_argument('--alpha', type=float, default=ADV_ALPHA)
    parser.add_argument('--adv-steps', type=int, default=ADV_STEPS)
    parser.add_argument('--alpha-ratio', type=float, default=ADV_ALPHA_RATIO)
    parser.add_argument('--device', default=DEVICE)
    parser.add_argument('--no-checkpoints', action='store_true', default=not SAVE_CHECKPOINTS)
    parser.add_argument('--dry-run', action='store_true', help='Validate fork API, NPZ and configs; do not train.')
    add_idea_arguments(parser)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error('Duplicate seeds or arms are not allowed.')
    if any(s < 0 or s + 100_000 >= 2**32 for s in args.seeds):
        parser.error('Seeds must be >= 0 and seed + 100000 < 2**32.')
    if min(args.max_iter, args.decoder_epochs, args.adv_steps) < 1:
        parser.error('Training iterations and attack steps must be positive.')
    if not np.isfinite([args.epsilon, args.alpha, args.alpha_ratio]).all() or min(args.epsilon, args.alpha, args.alpha_ratio) <= 0:
        parser.error('Attack budgets/step sizes must be finite and positive.')
    validate_idea(args)
    return args


def load_cebra_fork(path):
    path = path.expanduser().resolve()
    if not (path / 'cebra' / '__init__.py').is_file():
        raise FileNotFoundError(f'CEBRA checkout not found: {path}. Use --cebra-dir /path/to/fork')
    for name in list(sys.modules):
        if name == 'cebra' or name.startswith('cebra.'):
            del sys.modules[name]
    sys.path.insert(0, str(path))
    import cebra
    imported = Path(cebra.__file__).resolve()
    if path not in imported.parents:
        raise RuntimeError(f'Wrong import: {imported}; expected under {path}')
    print('Using fork:', imported, flush=True)
    return cebra


def experiment_configs(args, parameters):
    result = []
    for arm in args.arms:
        cfg = dict(batch_size=BATCH_SIZE, temperature=TEMPERATURE,
                   model_architecture=MODEL_ARCH, time_offsets=TIME_OFFSETS,
                   max_iterations=args.max_iter, output_dimension=LATENT_DIM,
                   num_hidden_units=HIDDEN, learning_rate=LEARNING_RATE,
                   temperature_mode='constant', device=args.device, verbose=True,
                   training_mode='standard' if arm == 'clean' else 'adversarial')
        if arm != 'clean':
            cfg.update(adv_epsilon=args.epsilon, adv_alpha=args.alpha,
                       adv_steps=args.adv_steps, attack_norm=ATTACK_NORM)
            # These controls were present in some previous PGD forks only.
            for name, value in [('attack_target', 'reference'),
                                ('adv_restarts', 1), ('adv_best_iterate', False)]:
                if name in parameters:
                    cfg[name] = value
            cfg.update(idea_config(arm, args))
        missing = sorted(set(cfg) - set(parameters))
        if missing:
            raise RuntimeError(f'{arm}: constructor does not expose {missing}. Check the selected fork.')
        result.append((arm, cfg))
    return result


def run_arm(cebra, arm, config, seed, arrays, out, args):
    x_train, x_valid, y_train, y_valid = arrays
    folder = out / f'seed_{seed}' / arm
    folder.mkdir(parents=True, exist_ok=False)
    encoder = decoder = z_train = z_valid = None
    started = time.perf_counter()
    try:
        print(f'\nSEED {seed} | ARM {arm}\n{json.dumps(config, indent=2)}', flush=True)
        seed_all(seed)  # Reset PER ARM: --arms ordering does not advance this seed.
        encoder = cebra.CEBRA(**config)
        encoder.fit(x_train, y_train)  # Supervised CEBRA: ALL label columns.
        z_train = np.asarray(encoder.transform(x_train), dtype=np.float32)
        z_valid = np.asarray(encoder.transform(x_valid), dtype=np.float32)
        for split, z, x in [('train', z_train, x_train), ('valid', z_valid, x_valid)]:
            if z.ndim != 2 or z.shape != (len(x), LATENT_DIM) or not np.isfinite(z).all():
                raise ValueError(f'{split}: invalid embedding shape/values: {z.shape}')
        encoder_seconds = time.perf_counter() - started
        decoder, losses = train_decoder(z_train, y_train, seed + 100_000,
                                        torch.device(encoder.device_))
        train_pred = predict_decoder(decoder, z_train)
        valid_pred = predict_decoder(decoder, z_valid)
        train_r2, train_outputs = compute_r2(y_train, train_pred)
        valid_r2, valid_outputs = compute_r2(y_valid, valid_pred)
        row = dict(session=args.session, seed=seed, arm=arm,
                   train_mean_r2=train_r2, valid_mean_r2=valid_r2,
                   train_r2_per_output=train_outputs, valid_r2_per_output=valid_outputs,
                   encoder_seconds=encoder_seconds, total_seconds=time.perf_counter()-started,
                   encoder_config=config, decoder_seed=seed+100_000,
                   evaluated_inputs='clean; no evaluation-time attack')
        save_json(folder / 'metrics.json', row)
        np.save(folder / 'decoder_train_mse.npy', np.asarray(losses, dtype=np.float32))
        np.savez_compressed(folder / 'validation_predictions.npz', y_true=y_valid, y_pred=valid_pred)
        if not args.no_checkpoints:
            encoder.save(str(folder / 'cebra.pt'), backend='sklearn')
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in decoder.state_dict().items()},
                            input_dim=LATENT_DIM, output_dim=int(y_train.shape[1]),
                            hidden=MLP_HIDDEN, dropout=MLP_DROP, seed=seed+100_000), folder / 'decoder.pt')
        print(f'{arm}: TRAIN mean R2={train_r2:.6f} | VALID mean R2={valid_r2:.6f}', flush=True)
        print('VALID per-output R2:', valid_outputs, flush=True)
        return row
    except Exception as exc:
        save_json(folder / 'failure.json', dict(arm=arm, seed=seed, error=repr(exc), config=config))
        raise
    finally:
        del encoder, decoder, z_train, z_valid
        cleanup()


def summarize(rows, out, args):
    with (out / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        n = len(rows[0]['valid_r2_per_output'])
        writer.writerow(['session','seed','arm','train_mean_r2','valid_mean_r2'] +
                        [f'valid_r2_output_{i}' for i in range(n)])
        for r in rows:
            writer.writerow([r['session'],r['seed'],r['arm'],r['train_mean_r2'],r['valid_mean_r2']] + r['valid_r2_per_output'])
    summary = {}
    print('\nFINAL CLEAN-INPUT VALIDATION R2 (all label columns)', flush=True)
    for arm in args.arms:
        group = [r for r in rows if r['arm'] == arm]
        if not group:
            continue
        scores = np.array([r['valid_mean_r2'] for r in group])
        item = dict(mean=float(scores.mean()), sample_std=float(scores.std(ddof=1)) if len(scores)>1 else None,
                    per_seed={str(r['seed']):r['valid_mean_r2'] for r in group},
                    mean_r2_per_output=np.mean([r['valid_r2_per_output'] for r in group], axis=0).tolist())
        summary[arm] = item
        print(f"{arm:<25} R2={item['mean']:.6f} std={item['sample_std']} seeds={item['per_seed']}", flush=True)
    save_json(out / 'summary.json', dict(session=args.session, seeds=args.seeds, by_arm=summary))


def main():
    global SESSION, NPZ_PATH, MLP_EPOCHS
    args = parse_args()
    SESSION = args.session
    NPZ_PATH = args.data_dir.expanduser() / f'{SESSION}.npz'
    MLP_EPOCHS = args.decoder_epochs
    cebra = load_cebra_fork(args.cebra_dir)
    params = inspect.signature(cebra.CEBRA.__init__).parameters
    configs = experiment_configs(args, params)
    for arm, cfg in configs:
        cebra.CEBRA(**cfg)  # Fail before training if constructor rejects any config.
    arrays = load_data()
    x_train, x_valid, y_train, y_valid = arrays
    print('DATA:', NPZ_PATH, '\nShapes:', [a.shape for a in arrays], flush=True)
    print('NPZ split preserved. Encoder and decoder use ALL', y_train.shape[1], 'labels.', flush=True)
    print('Input range (train):', float(x_train.min()), float(x_train.max()), flush=True)
    for name in ('attack_target', 'adv_restarts', 'adv_best_iterate'):
        if name not in params:
            print(f'NOTE: {name} is not exposed; the fork internal behavior applies.', flush=True)
    print('Arms:', args.arms, '| seeds:', args.seeds, flush=True)
    print('Baseline epsilon/alpha:', args.epsilon, args.alpha,
          '| new-attack alpha ratio:', args.alpha_ratio, flush=True)
    if args.dry_run:
        print(json.dumps(dict(configs), indent=2), flush=True)
        print('Preflight passed. Attack execution was not tested.', flush=True)
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    out = args.out_root.expanduser() / f'{SESSION}_{stamp}'
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'run_config.json', dict(
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        fork=str(Path(cebra.__file__).resolve()),
        cebra_version=str(getattr(cebra,'__version__','unknown')),
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        data_shapes=[list(a.shape) for a in arrays], evaluation_split='valid_data',
        labels='all columns', configs=dict(configs),
        decoder=dict(epochs=MLP_EPOCHS, hidden=MLP_HIDDEN, dropout=MLP_DROP,
                     learning_rate=MLP_LR, batch_mode='full', eval_mode=True),
        constructor_defaults={k:repr(v.default) for k,v in params.items()}
    ))
    print('OUTPUT:', out, flush=True)
    rows = []
    for seed in args.seeds:
        for arm, cfg in configs:
            rows.append(run_arm(cebra, arm, cfg, seed, arrays, out, args))
            save_json(out / 'results_so_far.json', rows)
            summarize(rows, out, args)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
