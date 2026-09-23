import argparse
import gc
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path('/data/hossein/mm_project/perich_data_valid_final_raw/')
WINDOW_SIZE = 10
LATENT_DIM = 64
ENCODER_HIDDEN = 64
DECODER_EPOCHS = 2500
DECODER_HIDDEN = 64
DECODER_DROPOUT = 0.4
DECODER_LR = 1e-3
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
_EPS = 1e-8


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return value


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_cebra_fork(directory):
    directory = directory.expanduser().resolve()
    if not (directory / 'cebra' / '__init__.py').is_file():
        raise FileNotFoundError(
            f'CEBRA checkout not found at {directory}. Expected '
            'CEBRA-original/cebra/__init__.py. Pass --cebra-dir /actual/path/CEBRA-original.')
    for name in list(sys.modules):
        if name == 'cebra' or name.startswith('cebra.'):
            del sys.modules[name]
    sys.path.insert(0, str(directory))
    importlib.invalidate_caches()
    cebra = importlib.import_module('cebra')
    source = Path(cebra.__file__).resolve()
    if directory not in source.parents:
        raise RuntimeError(f'Wrong CEBRA imported: {source}; expected checkout {directory}.')
    revision = None
    try:
        result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0:
            revision = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    print(f'Using CEBRA: {source}', flush=True)
    return cebra, dict(path=str(source), version=str(getattr(cebra, '__version__', 'unknown')),
                       git_commit=revision)


def load_session(path):
    required = ('train_data', 'train_label', 'valid_data', 'valid_label')
    with np.load(path, allow_pickle=False) as handle:
        missing = [key for key in required if key not in handle.files]
        if missing:
            raise KeyError(f'{path.name}: missing keys {missing}; found {handle.files}')
        arrays = [np.asarray(handle[key], dtype=np.float32) for key in required]
    splits = []
    original_label_counts = []
    for name, spikes, labels in (('train', arrays[0], arrays[1]),
                                ('valid', arrays[2], arrays[3])):
        if labels.ndim == 1:
            labels = labels[:, None]
        if spikes.ndim != 2 or labels.ndim != 2:
            raise ValueError(f'{name}: expected 2D spikes/labels, got {spikes.shape}/{labels.shape}')
        if len(spikes) != len(labels) and spikes.shape[1] == len(labels):
            spikes = spikes.T
        if len(spikes) != len(labels):
            raise ValueError(f'{name}: spikes and labels have different lengths.')
        if labels.shape[1] < 2:
            raise ValueError(f'{name}: need at least two label columns, got {labels.shape[1]}.')
        if len(spikes) < WINDOW_SIZE + 2:
            raise ValueError(f'{name}: not enough bins for decoding with window={WINDOW_SIZE}.')
        if not (np.isfinite(spikes).all() and np.isfinite(labels).all()):
            raise ValueError(f'{name}: data contains NaN or Inf.')
        original_label_counts.append(int(labels.shape[1]))
        splits.append((spikes, labels[:, :2]))
    x_train, y_train = splits[0]
    x_valid, y_valid = splits[1]
    if x_train.shape[1] != x_valid.shape[1]:
        raise ValueError('Train/valid neuron counts differ.')
    alive = x_train.std(0) > 0
    if not alive.any():
        raise ValueError('No varying neurons in train_data.')
    data = tuple(np.ascontiguousarray(a) for a in
                 (x_train[:, alive], y_train, x_valid[:, alive], y_valid))
    info = dict(train_bins=len(x_train), valid_bins=len(x_valid),
                original_neurons=int(x_train.shape[1]), kept_neurons=int(alive.sum()),
                neuron_indices=np.flatnonzero(alive).tolist(), label_columns=[0, 1],
                original_label_counts=original_label_counts, split='NPZ train_data / valid_data',
                trial_boundaries='Not supplied: each NPZ split treated as one continuous array.')
    return data, info


def matching_update_budget(train_bins, jigsaw_epochs):
    # Exactly the span and minibatch arithmetic of the supplied Jigsaw fit loop.
    span = 4 * WINDOW_SIZE + 3 * 8 + 1
    spans = train_bins - span + 1
    if spans < 2:
        raise ValueError('Too few bins to match the old puzzle budget; use --max-iterations.')
    full, remainder = divmod(spans, 512)
    steps_per_epoch = full + int(remainder >= 2)  # old loop skips a singleton tail
    return steps_per_epoch * jigsaw_epochs, dict(
        reference_jigsaw_epochs=jigsaw_epochs, reference_n_tiles=4,
        reference_tile_gap=[1, 8], reference_batch_size=512,
        reference_span=span, reference_n_spans=spans,
        reference_updates_per_epoch=steps_per_epoch)


class Decoder(nn.Module):
    def __init__(self, dimension, targets, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, hidden), nn.LayerNorm(hidden),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, targets))

    def forward(self, x):
        return self.net(x)


def r2_raw(truth, prediction):
    residual = np.square(truth - prediction).sum(0)
    total = np.square(truth - truth.mean(0, keepdims=True)).sum(0)
    return float(np.mean(1.0 - residual / np.maximum(total, 1e-12)))


def r2_per_label(truth, prediction):
    residual = np.square(truth - prediction).sum(0)
    total = np.square(truth - truth.mean(0, keepdims=True)).sum(0)
    return (1.0 - residual / np.maximum(total, 1e-12)).tolist()


def mlp_r2(x_train, y_train, x_test, y_test, *, epochs, hidden, dropout,
           learning_rate, seed, device, artifact_dir):
    # Same training/selection math as run_jigsaw.py. Added reporting and saving
    # perform no optimizer steps and draw no random numbers.
    torch.manual_seed(seed)
    mean, scale = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True) + 1e-8
    cut = max(1, int(0.9 * len(x_train)))
    tensors = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
               for a in ((x_train - mean) / scale, y_train, (x_test - mean) / scale, y_test)]
    xt, yt, xv, yv = tensors
    inner_x, inner_y, hold_x, hold_y = xt[:cut], yt[:cut], xt[cut:], yt[cut:]
    if len(hold_x) < 2:
        inner_x, inner_y, hold_x, hold_y = xt, yt, xt, yt
    model = Decoder(xt.shape[1], yt.shape[1], hidden, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state, best_score, best_epoch = None, -float('inf'), 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        nn.functional.mse_loss(model(inner_x), inner_y).backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            model.eval()
            with torch.no_grad():
                score = r2_raw(hold_y.cpu().numpy(), model(hold_x).cpu().numpy())
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % 250 == 0 or epoch + 1 == epochs:
                print(f'Decoder {epoch + 1}/{epochs}: inner holdout R2={score:.6f}; '
                      f'best={best_score:.6f} at {best_epoch}', flush=True)
    if best_state is None:
        raise FloatingPointError('No finite decoder checkpoint could be selected.')
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_valid = model(xv).cpu().numpy()
        p_train = model(xt).cpu().numpy()
        test = r2_raw(yv.cpu().numpy(), p_valid)
        train = r2_raw(yt.cpu().numpy(), p_train)
    metrics = dict(r2=test, r2_train=train, r2_internal_holdout=best_score,
                   best_epoch=best_epoch, r2_per_dimension=r2_per_label(y_test, p_valid),
                   r2_train_per_dimension=r2_per_label(y_train, p_train))
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    feature_mean=torch.from_numpy(mean), feature_scale=torch.from_numpy(scale),
                    input_dimension=int(xt.shape[1]), targets=2, hidden=hidden, dropout=dropout,
                    best_epoch=best_epoch, seed=seed), artifact_dir / 'decoder.pt')
    np.savez_compressed(artifact_dir / 'decoder_predictions.npz',
                        train_prediction=p_train, valid_prediction=p_valid,
                        train_truth=y_train, valid_truth=y_test)
    del model, optimizer, best_state, tensors, xt, yt, xv, yv
    cleanup()
    return metrics


def _participation_ratio(features):
    centered = features - features.mean(0, keepdims=True)
    values = np.linalg.eigvalsh(np.cov(centered, rowvar=False)
                                + 1e-12 * np.eye(centered.shape[1]))
    values = np.clip(values, 0, None)
    total = values.sum()
    return float(total ** 2 / np.square(values).sum()) if total > 0 else 0.0


def _ridge_r2(train_features, train_targets, test_features, test_targets, alphas):
    """Historical Jigsaw helper, unchanged, including inner-fit centering caveat."""
    mean = train_features.mean(0, keepdims=True)
    scale = train_features.std(0, keepdims=True) + _EPS
    a_train = (train_features - mean) / scale
    a_test = (test_features - mean) / scale
    cut = max(1, int(0.8 * len(a_train)))
    inner_x, inner_y, hold_x, hold_y = a_train[:cut], train_targets[:cut], \
        a_train[cut:], train_targets[cut:]
    if len(hold_x) < 2:
        inner_x, inner_y, hold_x, hold_y = a_train, train_targets, a_train, train_targets

    def solve(x, y, alpha):
        offset = y.mean(0, keepdims=True)
        return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y - offset)), offset

    def r2(y_true, y_hat):
        residual = np.square(y_true - y_hat).sum(0)
        total = np.square(y_true - y_true.mean(0, keepdims=True)).sum(0)
        return 1.0 - residual / np.maximum(total, 1e-12)

    scored = [(float(np.mean(r2(hold_y, hold_x @ w + b))), alpha)
              for alpha in alphas for w, b in [solve(inner_x, inner_y, alpha)]]
    best = max(scored)[1]
    weights, offset = solve(a_train, train_targets, best)
    per_dimension = r2(test_targets, a_test @ weights + offset)
    return dict(r2=float(np.mean(per_dimension)), r2_per_dimension=per_dimension.tolist(),
                alpha=float(best), participation_ratio=_participation_ratio(train_features))


def safe_ridge(x_train, y_train, x_test, y_test):
    if (x_train.size + x_test.size) * 8 >= 1.5e9:
        cast = np.float32
    else:
        cast = np.float64
    return _ridge_r2(x_train.astype(cast), y_train.astype(np.float64),
                     x_test.astype(cast), y_test.astype(np.float64), RIDGE_ALPHAS)


def transform_interior(model, x):
    # Use CEBRA's own public transform, with pad_before_transform=False.
    # This yields the same natural windows and center indices as Jigsaw pad=False.
    solver = getattr(model, 'solver_', None)
    network = getattr(solver, 'model', None)
    if network is None:
        network = getattr(model, 'model_', None)
    if network is None or not hasattr(network, 'get_offset'):
        raise RuntimeError('Cannot inspect the fitted CEBRA receptive field in this fork.')
    offset = network.get_offset()
    if (int(offset.left), int(offset.right)) != (5, 5):
        raise RuntimeError(f'Expected offset=(5,5) for offset10-model, got {offset}.')
    if model.pad_before_transform:
        raise RuntimeError('pad_before_transform must be False for this comparison.')
    z = np.asarray(model.transform(x), dtype=np.float32)
    expected = (len(x) - WINDOW_SIZE + 1, LATENT_DIM)
    if z.shape != expected:
        raise RuntimeError(f'Unexpected transform shape {z.shape}; expected {expected}. '
                           'Do not silently truncate embeddings or labels.')
    if not np.isfinite(z).all():
        raise FloatingPointError('Nonfinite CEBRA embeddings.')
    indices = np.arange(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + len(z), dtype=np.int64)
    return np.ascontiguousarray(z), indices


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', default='C-CO0')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--cebra-dir', type=Path, default=ROOT / 'CEBRA-original')
    parser.add_argument('--out-dir', type=Path, default=ROOT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 8])
    parser.add_argument('--max-iterations', type=positive_int, default=None,
                        help='Explicit CEBRA optimizer updates; overrides budget matching.')
    parser.add_argument('--jigsaw-epochs', type=positive_int, default=10000,
                        help='Reference budget only; no Jigsaw training is performed.')
    parser.add_argument('--time-offset', type=positive_int, default=10,
                        help='Temporal lag for positive pairs, in bins; not receptive-field length.')
    parser.add_argument('--batch-size', type=positive_int, default=512)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--decoder-epochs', type=positive_int, default=DECODER_EPOCHS)
    parser.add_argument('--device', default='cuda_if_available')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error('Seeds must lie between 0 and 2**32 - 1.')
    if not all(math.isfinite(v) and v > 0 for v in (args.learning_rate, args.temperature)):
        parser.error('Learning rate and temperature must be positive finite numbers.')
    seeds = list(dict.fromkeys(args.seeds))
    data_path = (args.data_dir.expanduser() / f'{args.session}.npz').resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f'Dataset not found: {data_path}')
    cebra, source_info = load_cebra_fork(args.cebra_dir)
    (x_train, y_train, x_valid, y_valid), data_info = load_session(data_path)
    if len(x_train) <= WINDOW_SIZE + args.time_offset:
        raise ValueError('Not enough training bins for the requested time offset.')
    if args.max_iterations is None:
        max_iterations, budget = matching_update_budget(len(x_train), args.jigsaw_epochs)
        budget['mode'] = 'match_old_jigsaw_optimizer_update_count'
    else:
        max_iterations = args.max_iterations
        budget = dict(mode='explicit_CEBRA_iterations')
    budget['cebra_optimizer_updates'] = max_iterations
    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
        if args.device == 'cuda_if_available' else args.device
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = args.out_dir.expanduser().resolve() / f'CEBRA_TIME_{args.session}_2LABELS_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    settings = dict(model_architecture='offset10-model', batch_size=args.batch_size,
                    learning_rate=args.learning_rate, temperature=args.temperature,
                    temperature_mode='constant', max_iterations=max_iterations,
                    conditional='time', time_offsets=args.time_offset,
                    output_dimension=LATENT_DIM, num_hidden_units=ENCODER_HIDDEN,
                    distance='cosine', criterion='infonce', hybrid=False,
                    pad_before_transform=False, device=device, verbose=not args.quiet)
    payload = dict(session=args.session, seeds=seeds, data_path=str(data_path),
                   data_sha256=file_sha256(data_path), data=data_info, cebra=source_info,
                   torch_version=str(torch.__version__), numpy_version=np.__version__,
                   cebra_settings=settings, budget=budget, window_size=WINDOW_SIZE,
                   label_alignment='window starting j -> label j+5; no padded windows',
                   decoder=dict(epochs=args.decoder_epochs, hidden=DECODER_HIDDEN,
                                dropout=DECODER_DROPOUT, learning_rate=DECODER_LR,
                                selection='last 10% of training embeddings, every 25 epochs',
                                evaluation='NPZ valid_data; never used to select decoder weights'),
                   ridge_protocol='historical Jigsaw helper; inner-fit centering caveat preserved',
                   runs={})
    save_json(output / 'results.json', payload)
    print(f'DATA: {data_path}\ntrain={x_train.shape}, valid={x_valid.shape}; '
          f'behavior columns [0,1]\nCEBRA updates={max_iterations:,}; '
          f'time lag={args.time_offset} bins; receptive field={WINDOW_SIZE} bins\n'
          f'Output: {output}', flush=True)
    for seed in seeds:
        seed_dir = output / f'seed_{seed}'
        seed_dir.mkdir()
        seed_all(seed)
        started = time.time()
        print(f'\nTRAIN CEBRA-TIME | seed={seed} | NO behavior labels passed to fit', flush=True)
        model = cebra.CEBRA(**settings)
        model.fit(x_train)  # Intentionally no y, timestamps, or validation data.
        model.save(str(seed_dir / 'cebra_time.pt'))
        z_train, i_train = transform_interior(model, x_train)
        z_valid, i_valid = transform_interior(model, x_valid)
        train_targets, valid_targets = y_train[i_train], y_valid[i_valid]
        np.savez_compressed(seed_dir / 'embeddings.npz', train=z_train, valid=z_valid,
                            train_indices=i_train, valid_indices=i_valid,
                            train_label=train_targets, valid_label=valid_targets)
        print(f'Embeddings: train={z_train.shape}, valid={z_valid.shape}; '
              f'valid label indices {i_valid[0]}..{i_valid[-1]}', flush=True)
        mlp = mlp_r2(z_train, train_targets, z_valid, valid_targets,
                     epochs=args.decoder_epochs, hidden=DECODER_HIDDEN,
                     dropout=DECODER_DROPOUT, learning_rate=DECODER_LR,
                     seed=seed, device=device, artifact_dir=seed_dir)
        ridge = safe_ridge(z_train, train_targets, z_valid, valid_targets)
        row = dict(seed=seed, mlp=mlp, ridge=ridge, seconds=time.time() - started)
        payload['runs'][str(seed)] = row
        save_json(output / 'results.json', payload)
        save_json(seed_dir / 'metrics.json', row)
        print(f"TRAIN MLP mean R2: {mlp['r2_train']:.6f}\n"
              f"VALID MLP mean R2: {mlp['r2']:.6f}; per-label={mlp['r2_per_dimension']}\n"
              f"VALID Ridge mean R2: {ridge['r2']:.6f}; per-label={ridge['r2_per_dimension']}",
              flush=True)
        del model, z_train, z_valid
        cleanup()
    summary = {}
    print('\nSUMMARY: validation R2, mean over behavior columns [0,1]')
    print(f"{'seed':>8}{'MLP valid':>14}{'MLP train':>14}{'Ridge valid':>14}")
    for row in payload['runs'].values():
        print(f"{row['seed']:>8}{row['mlp']['r2']:>14.6f}"
              f"{row['mlp']['r2_train']:>14.6f}{row['ridge']['r2']:>14.6f}")
    for name in ('mlp', 'ridge'):
        values = [row[name]['r2'] for row in payload['runs'].values()]
        summary[name] = dict(mean=float(np.mean(values)), spread=float(np.ptp(values)),
                             per_seed=dict(zip(map(str, seeds), values)))
        print(f"{name.upper()}: mean={summary[name]['mean']:.6f}; "
              f"spread(max-min)={summary[name]['spread']:.6f}")
    payload['summary'] = summary
    save_json(output / 'results.json', payload)
    print(f'Saved all results: {output}', flush=True)


if __name__ == '__main__':
    main()
