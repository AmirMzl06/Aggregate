#!/usr/bin/env python3
"""Audit existing NTT encoder checkpoints WITHOUT retraining, changing, or resuming them.

Example:
 python ntt_audit_existing.py --run /path/to/NTT_CCO36_RESULTS/C-CO36_<timestamp> --device cuda

Outputs: posthoc_audit_v2/checkpoint_pair_metrics.csv, shortcut_audit_v2.json,
posthoc_audit_v2/<architecture>/seed_.../<arm>/step_.../pretext_v2.json.
These are diagnostics on the already-used NPZ validation split, NOT final test.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ntt_architectures import EncoderConfig, TournamentEncoder
from run_ntt_cco12 import (clean_json, fit_shortcut_audit, load_data,
                            pretext_diagnostics, save_json, seed_all)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path, help='Existing run folder containing run_config.json')
    parser.add_argument('--data-dir', type=Path, default=None, help='Override original NPZ directory if moved')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--step', type=int, default=3000, help='Saved encoder training step')
    parser.add_argument('--arms', nargs='*', default=None, help='Optional arms, default all available')
    parser.add_argument('--seeds', type=int, nargs='*', default=None, help='Optional seeds, default all available')
    parser.add_argument('--diagnostic-samples', type=int, default=None)
    cli = parser.parse_args()
    root = cli.run.expanduser().resolve()
    config_file = root / 'run_config.json'
    if not config_file.is_file():
        parser.error(f'No run_config.json at {config_file}')
    config = json.loads(config_file.read_text(encoding='utf-8'))
    args = SimpleNamespace(**config['args'])
    args.data_dir = cli.data_dir if cli.data_dir is not None else Path(args.data_dir)
    if cli.diagnostic_samples is not None:
        if cli.diagnostic_samples < 8:
            parser.error('--diagnostic-samples must be >= 8')
        args.diagnostic_samples = cli.diagnostic_samples
    torch.set_num_threads(min(getattr(args, 'threads', 4), 8))
    data = load_data(args)
    expected_digest = config.get('dataset_sha256')
    observed_digest = data['audit']['sha256_loaded_arrays']
    if expected_digest and observed_digest != expected_digest:
        raise RuntimeError('Input dataset SHA256 differs from original run. Refusing mismatched audit.')
    device = torch.device(cli.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    out = root / 'posthoc_audit_v2'
    out.mkdir(exist_ok=True)
    seed_all(42)
    shortcut = fit_shortcut_audit(data, args, seed=int(args.seeds[0]))
    save_json(out / 'shortcut_audit_v2.json', shortcut)
    print('Split shapes:', tuple(data['xt'].shape), tuple(data['xv'].shape), flush=True)
    print('SHORTCUT feature families:')
    for group, scores in shortcut['feature_family_probes'].items():
        r = scores['valid']['ranking']
        print(f'  {group:<24} valid pair (ties=0.5)={r["pair_accuracy_ties_half"]:.4f}'
              f'  ties={r["tie_fraction"]:.4f}'
              f'  valid full={r["full_order_accuracy"]:.4f}', flush=True)
    rows = []
    checkpoints = sorted(root.glob(f'*/seed_*/*/step_{cli.step:06d}/encoder.pt'))
    if not checkpoints:
        raise FileNotFoundError(f'No encoder.pt checkpoints for step {cli.step} in {root}')
    for ckpt_path in checkpoints:
        arch, seed_name, arm = ckpt_path.relative_to(root).parts[:3]
        if arm == 'init' or (cli.arms is not None and arm not in cli.arms):
            continue
        seed_match = re.fullmatch('seed_(\\d+)', seed_name)
        if seed_match is None:
            continue
        seed = int(seed_match.group(1))
        if cli.seeds is not None and seed not in cli.seeds:
            continue
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model = TournamentEncoder(EncoderConfig(**checkpoint['config'])).to(device)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        result = pretext_diagnostics(model, data, args, seed, device)
        target = out / arch / seed_name / arm / f'step_{cli.step:06d}'
        target.mkdir(parents=True, exist_ok=True)
        save_json(target / 'pretext_v2.json', result)
        row = dict(architecture=arch, seed=seed, arm=arm, iteration=cli.step)
        for split in ('heldout_seen_strengths', 'heldout_unseen_strengths',
                      'valid_seen_strengths', 'valid_unseen_strengths'):
            entry = result.get(split, {})
            if 'z_ranking' not in entry:
                continue
            for measure, obj in [('z', entry['z_ranking']),
                                  ('h', entry['h_ranking']),
                                  ('raw_l2', entry['raw_l2_ranking']),
                                  ('spectrum', entry['power_spectrum_distance_ranking'])]:
                row[split + '_' + measure + '_pair'] = obj['pair_accuracy']
                row[split + '_' + measure + '_full_order'] = obj['full_order_accuracy']
            for rep in ('z', 'h'):
                geom = entry[rep + '_clean_geometry']
                row[split + '_' + rep + '_clean_mean_std'] = geom['mean_std']
                row[split + '_' + rep + '_clean_effective_rank'] = geom['effective_rank']
        rows.append(row)
        vs = result['valid_seen_strengths']
        vu = result['valid_unseen_strengths']
        print(f'{arch}/{seed_name}/{arm} step={cli.step}: '
              f'VALID z_pair={vs["z_ranking"]["pair_accuracy"]:.4f} '
              f'VALID unseen_z_pair={vu["z_ranking"]["pair_accuracy"]:.4f} '
              f'raw_l2_pair={vs["raw_l2_ranking"]["pair_accuracy"]:.4f} '
              f'clean_z_effective_rank={vs["z_clean_geometry"]["effective_rank"]:.3f}', flush=True)
        del model
    if not rows:
        raise ValueError('No checkpoints matched the requested arms/seeds/step.')
    fields = list(dict.fromkeys(key for r in rows for key in r))
    with (out / 'checkpoint_pair_metrics.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    save_json(out / 'audit_metadata.json', {
        'source_run': str(root), 'step': cli.step,
        'dataset_sha256_verified': expected_digest == observed_digest,
        'assurance': 'NPZ valid was previously used for downstream R2; not an independent final test',
        'notes': 'No optimization, decoder fit, or change to source checkpoints was performed.',
    })
    print(f'\nSaved diagnostic CSV and JSONs to: {out}', flush=True)


if __name__ == '__main__':
    main()
