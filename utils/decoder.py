import copy
import os
import random
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch import optim
from tqdm import tqdm, trange

# -------------------- Utilities --------------------

def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TwoLayerMLP(nn.Module):
    """
    MLP decoder w/ LayerNorm, Dropout, Kaiming init, early stopping + retraining.
    Calling pattern:
        decoder = AdvancedDecoderMLP(...)
        decoder.fit(train_x, train_y)
        pr2, dr2, pmae = decoder.score(test_x, test_y)
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=2,
                 dropout_rate=0.4, lr=1e-3, weight_decay=2e-4,
                 max_iters=10000, min_epochs=1500, patience=500,
                 batch_size=2048, device="cuda", verbose=True, use_amp=True,
                 amp_dtype=None, allow_tf32=True):
        super().__init__()
        self.device = device
        self.verbose = verbose
        self.max_iters = max_iters
        self.min_epochs = min_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_amp = use_amp and torch.cuda.is_available() and str(device).startswith("cuda")
        self.amp_dtype = torch.bfloat16 if self.use_amp and amp_dtype is None else amp_dtype

        # Enable TF32 for faster matmuls on Ampere+ GPUs if requested
        if allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.scaler = torch.amp.GradScaler(device, enabled=self.use_amp and self.amp_dtype == torch.float16)

        # Architecture
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

        self.to(device)

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)

    def _autocast(self):
        if self.use_amp:
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()
    

    # ------------- Internal r2 utility -------------
    @staticmethod
    def _r2_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
        p = pred.detach().float().cpu().numpy()
        t = target.detach().float().cpu().numpy()
        return float(np.mean([r2_score(t[:, i], p[:, i]) for i in range(p.shape[1])]))

    # ------------- Time-aware train/val split -------------
    @staticmethod
    def _split_time_aware(x: torch.Tensor, y: torch.Tensor, frac=0.125):
        n = len(x)
        v = max(1, int(frac * n))
        return x[:-v], y[:-v], x[-v:], y[-v:]

    # ------------- Training with early stopping -------------
    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, seed: int = 42, adv_steps: int = 10, adv_eps : float= 0.01, adv: bool = False):
        adv_alpha = adv_eps * 2 / adv_steps
        if self.verbose:
            print(f"Fitting decoder with input shape {train_x.shape} and output shape {train_y.shape} "
                  f"using {self.device} with amp {self.use_amp} and dtype {self.amp_dtype}")
        setup_seed(seed)
        train_x, train_y = train_x.to(self.device), train_y.to(self.device)

        # Split internal validation from the end (time-aware)
        tr_x, tr_y, val_x, val_y = self._split_time_aware(train_x, train_y)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scaler = self.scaler

        best_epoch = 0
        best_r2 = -1e9
        bad = 0

        init_state = copy.deepcopy(self.state_dict())
        tmp_path = os.path.join(tempfile.gettempdir(), f"decoder_{random.randint(0, 1_000_000)}.pt")

        # ---- Phase 1: Early stopping training ----
        for epoch in range(self.max_iters):
            self.train()
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = criterion(self(tr_x), tr_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch % self.patience != 0: continue
            # Validation R2
            self.eval()
            with torch.no_grad():
                with self._autocast():
                    r2 = self._r2_mean(self(val_x), val_y)

            if r2 > best_r2:
                best_r2 = r2
                best_epoch = epoch + 1
                bad = 0
                torch.save(self.state_dict(), tmp_path)
            else:
                if epoch >= self.min_epochs - self.patience:
                    bad += 1
                if bad >= self.patience:
                    if self.verbose:
                        print(f"[EarlyStop] epoch={epoch + 1}, best_epoch={best_epoch}, best_r2={best_r2:.3f}")
                    break

        # ---- Phase 2: Retrain from scratch on full train for best_epoch ----
        self.load_state_dict(init_state)
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scaler = self.scaler

        for e in range(best_epoch):
            self.train()
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                loss = criterion(self(train_x), train_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # if adv:
            #     self.eval()
            #     optimizer.zero_grad()
            #     x_org = train_x.detach().clone()
            #     x_adv = x_org.clone().detach() + torch.empty_like(x_org).uniform_(-adv_eps,adv_eps)
            #     x_adv.requires_grad_(True)
            #     for _ in range(adv_steps):
            #         with self._autocast():
            #             loss_adv = criterion(self(x_adv), train_y)
            #             grad_x, = torch.autograd.grad(
            #                 loss_adv, x_adv,
            #                 retain_graph=False,
            #                 create_graph=False
            #             )
            #         with torch.no_grad():
            #             x_adv = x_adv + adv_alpha * grad_x.sign()
            #             x_adv = torch.max(torch.min(x_adv, x_org + adv_eps),
            #                               x_org - adv_eps)
            #             x_adv.requires_grad = True
                        
            #     with self._autocast():
            #         outputs = self(x_adv)
            #         loss_adv = criterion(outputs, train_y)
            #     scaler.scale(loss_adv).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            
            

        if self.verbose:
            print(f"[RetrainDone] epochs={best_epoch}, best_r2={best_r2:.3f}")
        return self

    # ------------- Final evaluation -------------
    def score(self, test_x, test_y, device) -> float:
        self.eval()
        with torch.no_grad():
            predictions = self(test_x.to(device)).cpu().numpy()
            true_values = test_y.cpu().numpy()

        r2_scores = {}
        for i in range(len(predictions[0])):
            r2 = r2_score(true_values[:, i], predictions[:, i])
            # print(r2)
            r2_scores[f'Output_{i}'] = r2
        return sum(r2_scores.values()) / len(r2_scores)
