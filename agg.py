import sys
if "cebra" in sys.modules:
    del sys.modules["cebra"]

from utils.constants import CEBRA_DIR
sys.path.insert(0, str(CEBRA_DIR))
import cebra 
import itertools
import torch
from cebra import CEBRA
import torch.nn as nn
from utils.dataset_loader import DatasetLoader
from torch.utils.data import DataLoader, TensorDataset
from utils.normalization import normalize_to_target
from utils.min_distance import min_l2_distance
import os
import numpy as np
# names = ['buddy']
names = ['achilles',
         # 'gatsby',
         # 'buddy',
         # 'cicero'
        ]
adv =  True

for adv in [True, False]:
    adv_epsilon = 0.5
    epochs = 1500 if adv else 1500
    for name in names:
        print(f"rat = {rat} , adv_mode = {adv}")
        
        model = CEBRA(
            batch_size=1024,
            temperature=0.4,
            model_architecture="offset36-model-more-dropout",
            time_offsets=4,
                
            max_iterations=epochs,
            output_dimension=48,
            verbose=True,
            training_mode='clean' if not adv else 'adversarial',
            adv_alpha=adv_epsilon / 5,
            adv_epsilon=adv_epsilon,
            adv_steps=10, 
            attack_norm="l2"
        )
        dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
        split_idx = int(0.8 * len(dataset.neural))

        train_data = dataset.neural[:split_idx]
        valid_data = dataset.neural[split_idx:]

        train_continuous_label = dataset.continuous_index.numpy()[:split_idx, :2]
        valid_continuous_label = dataset.continuous_index.numpy()[split_idx:, :2]
        path = name
        if adv:
            path += '_adv'
        path += '.pth'
        model.fit(train_data, train_continuous_label)
        model.save(path)



device = 'cuda'
adv =  not True
results = {}

for name in names:
    dataset = cebra.datasets.init(f'rat-hippocampus-single-{name}')
    split_idx = int(0.8 * len(dataset.neural))

    train_data = dataset.neural[:split_idx]    
    valid_data = dataset.neural[split_idx:]
   
    

    train_continuous_label = dataset.continuous_index.numpy()[:split_idx, :2]
    valid_continuous_label = dataset.continuous_index.numpy()[split_idx:, :2]
    path = name 
    if adv:
        path += '_adv'
    path += '.pth'
    model = CEBRA.load(path, weights_only=False)
    model = model.solver_.model.to(device)
    
    method = cebra.attribution.init(
        name="jacobian-based",
        model=model,
        input_data=torch.from_numpy(train_data).requires_grad_(True),
        output_dimension=model.num_output
    )
    result = method.compute_attribution_map()
    results[name] = result
# names = list(map(str, [False, True]))


import os
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

fig, axes = plt.subplots(1, 1, figsize=(15, 8))
axes = [axes]

model_name = "ACORN" if adv else "CEBRA"

ims = []
for ax, name in zip(axes, names):
    result = results[name]
    jf = abs(result['jf']).mean(0)
    _, N = jf.shape

    im = ax.matshow(
        jf / jf.sum(),
        aspect="auto",
        cmap="cividis",
    )
    ims.append(im)
    ax.set_title(name)

fig.subplots_adjust(right=0.85, top=0.9)
cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
fig.colorbar(ims[0], cax=cbar_ax)

fig.suptitle(model_name, fontsize=14)

plt.savefig(os.path.join("images", f"{model_name}x.png"),
            dpi=300,
            bbox_inches="tight")

plt.show()
