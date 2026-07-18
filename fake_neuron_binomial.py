import sys
import os
import itertools
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

if "cebra" in sys.modules:
    del sys.modules["cebra"]

from utils.constants import CEBRA_DIR
sys.path.insert(0, str(CEBRA_DIR))
import cebra 
from cebra import CEBRA
from utils.dataset_loader import DatasetLoader
from torch.utils.data import DataLoader, TensorDataset
from utils.normalization import normalize_to_target
from utils.min_distance import min_l2_distance

# names = ['buddy']
names = [
    'achilles',
    # 'gatsby',
    # 'buddy',
    # 'cicero'
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_FAKE_NEURONS = 5

for name in names:
    print(f"\n========== Processing Rat: {name} ==========")
    
    # --- 1. Load Dataset and Inject Fake Neurons ---
    dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
    
    neural_data = dataset.neural.clone() if torch.is_tensor(dataset.neural) else torch.tensor(dataset.neural)
    num_samples, num_real_neurons = neural_data.shape
    
    fake_data = torch.tensor(np.random.binomial(n=1, p=0.5, size=(num_samples, NUM_FAKE_NEURONS)), dtype=neural_data.dtype)
    
    total_neurons = num_real_neurons + NUM_FAKE_NEURONS
    fake_indices = np.sort(np.random.choice(total_neurons, NUM_FAKE_NEURONS, replace=False))
    real_indices = np.setdiff1d(np.arange(total_neurons), fake_indices)
    
    combined_neural = torch.zeros((num_samples, total_neurons), dtype=neural_data.dtype)
    combined_neural[:, real_indices] = neural_data
    combined_neural[:, fake_indices] = fake_data
    
    print(f"Added {NUM_FAKE_NEURONS} fake neurons at indices: {fake_indices.tolist()}")
    
    split_idx = int(0.8 * len(combined_neural))
    train_data = combined_neural[:split_idx]
    valid_data = combined_neural[split_idx:]
    
    continuous_index = dataset.continuous_index.clone() if torch.is_tensor(dataset.continuous_index) else torch.tensor(dataset.continuous_index)
    train_continuous_label = continuous_index[:split_idx, :2].numpy()
    valid_continuous_label = continuous_index[split_idx:, :2].numpy()

    save_dir = os.path.join("images", name)
    os.makedirs(save_dir, exist_ok=True)
    
    results = {}

    for adv in [False, True]:
        model_name = "ACORN" if adv else "CEBRA"
        adv_epsilon = 0.5
        epochs = 1500
      
        print(f"\n--- Training {model_name} (adv = {adv}) ---")
        
        model = CEBRA(
            batch_size=1024,
            temperature=0.4,
            model_architecture="offset36-model-more-dropout",
            time_offsets=4,
            max_iterations=epochs,
            output_dimension=48,
            verbose=True,
            training_mode='adversarial' if adv else 'clean',
            adv_alpha=adv_epsilon / 5 if adv else 0.0,
            adv_epsilon=adv_epsilon if adv else 0.0,
            adv_steps=10 if adv else 0, 
            attack_norm="l2"
        )
        
        path = f"{name}_adv.pth" if adv else f"{name}.pth"
        
        # Fit & Save
        model.fit(train_data, train_continuous_label)
        model.save(path)
        
        # Load & Attribution
        model = CEBRA.load(path, weights_only=False)
        model = model.solver_.model.to(device)
        
        input_data = train_data.clone().detach().to(device).requires_grad_(True)
        method = cebra.attribution.init(
            name="jacobian-based",
            model=model,
            input_data=input_data,
            output_dimension=model.num_output
        )
        
        print(f"Computing Jacobian Map for {model_name}...")
        result = method.compute_attribution_map()
        results[adv] = result

        jf = abs(result['jf']).mean(0)
        jf_normalized = jf / jf.sum()
        
        fake_jf = jf_normalized[:, fake_indices] 
        mean_fake_latents = fake_jf.mean(axis=0)
        
        print(f"\n>>> [{model_name}] Average Latent Attribution for Fake Neurons:")
        for idx_order, global_idx in enumerate(fake_indices):
            print(f"    Fake Neuron #{idx_order+1} (Index: {global_idx}): {mean_fake_latents[idx_order]:.6e}")

        fig, ax = plt.subplots(figsize=(15, 8))
        im = ax.matshow(
            jf_normalized,
            aspect="auto",
            cmap="cividis",
        )
        ax.set_title(f"Rat: {name} | Model: {model_name}", fontsize=14)
        
        for global_idx in fake_indices:
            ax.axvline(x=global_idx, color='red', linestyle='--', alpha=0.8, linewidth=1)
            
        fig.colorbar(im, ax=ax)
        
        plot_path = os.path.join(save_dir, f"{model_name}_jacobian.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved successfully at: {plot_path}")
        plt.show()
