import copy
import random
import sys
import tempfile
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch import optim

from utils.constants import CEBRA_DIR

sys.path.append(CEBRA_DIR)
from cebra import CEBRA
from utils.dataset_loader import DatasetLoader
from utils.normalization import normalize_to_target, DataNormalizationLiteral

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# toggles for tf32 support
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # faster convs for fixed shapes
# PyTorch 2.x: "high" uses TF32 on matmuls; "medium" is a bit stricter
torch.set_float32_matmul_precision('high')

dataset_loader = DatasetLoader()

DECODER_ITERS = 10000  # leave it to the early stopping mechanism


class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=2, dropout_rate=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        self._initialize_weights()
        self.random_id = random.randint(0, 1000)

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)

    def fit(self, dataset_name: str, model: CEBRA, verbose=True, day_num=0,
            data_normalization: DataNormalizationLiteral = 'none', target_day_dist=0):
        self.train()
        day_data, day_label = dataset_loader.load_dataset_day(day_num, dataset_name)
        day_data = torch.from_numpy(day_data).float()
        day_label = torch.from_numpy(day_label).float()
        (
            neural_train,
            neural_test,
            label_train,
            label_test,
        ) = train_test_split(day_data, day_label, test_size=0.20, random_state=42, shuffle=False)

        # Time-aware: last 10% of train is val (small val for early stopping)
        val_size = max(1, int(0.125 * len(neural_train)))  # 10% of whole data for validation
        neural_val, label_val = neural_train[-val_size:], label_train[-val_size:]
        neural_train, label_train = neural_train[:-val_size], label_train[:-val_size]

        neural_train = normalize_to_target(neural_train, dataset_name, cur_day=day_num,
                                           data_normalization=data_normalization, target_day=target_day_dist)
        neural_val = normalize_to_target(neural_val, dataset_name, cur_day=day_num,
                                         data_normalization=data_normalization, target_day=target_day_dist)
        neural_test = normalize_to_target(neural_test, dataset_name, cur_day=day_num,
                                          data_normalization=data_normalization, target_day=target_day_dist)

        model.model_.eval()
        cebra_posdir_train = model.transform(neural_train)  # already used
        cebra_posdir_val = model.transform(neural_val.cpu()).copy()

        train_x = torch.from_numpy(cebra_posdir_train).to(device).float()
        train_y = label_train.cuda().float()

        val_x = torch.from_numpy(cebra_posdir_val).to(device).float()
        val_y = label_val.cuda().float()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=2e-4)

        num_epochs = DECODER_ITERS
        r2_score_best = -1e9
        best_epoch = 0
        temp_path = tempfile.gettempdir()

        # Save initial weights so we can retrain on train+val later
        initial_state = copy.deepcopy(self.state_dict())  # NEW

        # -------- Phase 1: train using small val for early stopping --------
        patience, bad, min_epochs = 1000, 0, 4000
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self(train_x)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()

            r2 = self.score(val_x, val_y, model=None)  # model unused now
            if r2 > r2_score_best:
                r2_score_best = r2
                best_epoch = epoch + 1
                bad = 0
                torch.save(self.state_dict(), temp_path + f"/linear_model_{self.random_id}.pt")
            else:
                if epoch > min_epochs - patience:
                    bad += 1
                if bad >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            if (epoch + 1) % 1000 == 0:
                verbose and print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.3f}, r\u00B2: {r2:.3f}')

        # -------- Phase 2: retrain on TRAIN+VAL union for best_epoch --------
        # combine already-normalized tensors
        neural_train_full = torch.cat([neural_train, neural_val], dim=0)
        label_train_full = torch.cat([label_train, label_val], dim=0)
        cebra_posdir_train_full = model.transform(neural_train_full.cpu()).copy()
        train_x_full = torch.from_numpy(cebra_posdir_train_full).to(device).float()
        train_y_full = label_train_full.cuda().float()

        # reset weights to the initial state and optimizer
        self.load_state_dict(initial_state)
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=2e-4)

        for epoch in range(best_epoch):  # train only for the selected number of epochs
            self.train()
            optimizer.zero_grad()
            outputs = self(train_x_full)
            loss = criterion(outputs, train_y_full)
            loss.backward()
            optimizer.step()

        # -------- Final evaluation on untouched test --------
        cebra_posdir_test = model.transform(neural_test.cpu()).copy()
        test_x = torch.from_numpy(cebra_posdir_test).to(device).float()
        test_y = label_test.cuda().float()
        r2 = self.score(test_x, test_y, model=None)
        print(f'Final r\u00B2: {r2:.3f}')
        return self

    def score(self, x, label, model: Optional[CEBRA]):
        if model is None:
            embedding = x if isinstance(x, torch.Tensor) else torch.from_numpy(x).to(device).float()
        else:
            embedding = torch.from_numpy(model.transform(x.cpu())).to(device).float()
        self.eval()
        with torch.no_grad():
            predictions = self(embedding.float().to(device)).cpu().numpy()
            true_values = label.cpu().numpy()

        r2_scores = {}
        for i in range(len(predictions[0])):
            r2 = r2_score(true_values[:, i], predictions[:, i])
            r2_scores[f'Output_{i}'] = r2
        return sum(r2_scores.values()) / len(r2_scores)
