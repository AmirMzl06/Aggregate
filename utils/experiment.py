import os

import torch
from sklearn.model_selection import train_test_split

from CEBRA_Parallel.run.models import ExperimentConfig
from utils.dataset_loader import DatasetLoader

dataset_loader = DatasetLoader()


def get_experiment_config(
        training_mode,
        dataset_name,
        normalize=None,
        day_idx=0,
        model_architecture='offset36-model-more-dropout',
        batch_size=2048,
        learning_rate=3e-4,
        temperature=0.4,
        output_dimension=48,
        max_iterations=5000,
        max_adapt_iterations=500,
        distance='cosine',
        conditional='time_delta',
        adv_steps=10,
        attack_norm='l2',
        verbose=True,
        time_offsets=4,
        random_seed=42,
        mul=None
) -> ExperimentConfig:
    day_data, day_label = dataset_loader.load_dataset_day(day_idx, dataset_name)
    day_data = torch.from_numpy(day_data).float()
    day_label = torch.from_numpy(day_label).float()
    day_train_data, _, _, _ = train_test_split(day_data, day_label, test_size=0.20, shuffle=False)

    if normalize is None:
        normalize = training_mode == "adversarial"
    if not mul:
        if normalize:
            epsilon = 2.5
            alpha = 0.7
        else:
            epsilon = 0.2
            alpha = 0.03
    else:
        epsilon = mul
        alpha = 2.0 * mul / 10
    train_ranges = [(0, day_train_data.shape[0])]

    if training_mode != "standard":
        max_iterations //= 2
        max_adapt_iterations //= 2

    dataset_name = dataset_name
    experiment_cfg = ExperimentConfig(
        dataset_name=dataset_name,
        train_ranges=train_ranges,
        normalize=normalize,
        day_idx=day_idx,
        model_architecture=model_architecture,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        output_dimension=output_dimension,
        max_iterations=max_iterations,
        max_adapt_iterations=max_adapt_iterations,
        distance=distance,  # type: ignore
        conditional=conditional,  # type: ignore
        device='cuda_if_available',
        training_mode=training_mode,
        adv_epsilon=epsilon, adv_alpha=alpha, adv_steps=adv_steps,
        attack_norm=attack_norm,  # type: ignore
        verbose=verbose,
        time_offsets=time_offsets,
        random_seed=random_seed,
    )
    return experiment_cfg


def get_experiment_configs(combination_dicts, get_all: bool = False):
    train_configs = []
    for combination_dict in combination_dicts:
        experiment_cfg = get_experiment_config(**combination_dict)
        if not get_all and experiment_cfg.trained_model_exists():
            print(f"Model for {combination_dict} already exists. Skipping.")
            continue
        train_configs.append(experiment_cfg)
    return train_configs
