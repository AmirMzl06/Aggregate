import torch
from torch import nn
from utils.decoder import TwoLayerMLP
from sklearn.metrics import r2_score
from torch import optim
from tqdm import tqdm, trange
import copy
import os
import random
import tempfile
from contextlib import nullcontext
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset


def setup_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LSTM_Dataset(Dataset):
    def __init__(self, x_s, y_s) -> None:
        super().__init__()
        self.x_s = x_s
        self.y_s = y_s

    def __getitem__(self, index):
        return self.x_s[index], self.y_s[index]
    
    def __len__(self):
        return len(self.x_s)
        
class LSTM_Dec(nn.Module):

    def __init__(self, feat_dim: int, hidden: int, num_outputs: int, nLayers: int = 2, bidir=True,
            device='cuda',
            verbose=True,
            use_amp=True,
            amp_dtype=None, 
            allow_tf32=True,
            batch_size=256,
            lr=1e-3, 
            weight_decay=2e-4,
            max_iters=10000, 
            min_epochs=1500, 
            patience=500,):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=nLayers,
            bidirectional=bidir,
            batch_first=True,
            dropout=0,
            
            )
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

        
        
        self.out = TwoLayerMLP((2 if bidir else 1) * hidden, 64, num_outputs) 
        
    def forward(self, x: torch.Tensor):
        if x.ndim < 3:
            x, _ = self.lstm(x.unsqueeze(0))
            x = x.squeeze(0)
        else: 
            x, _ = self.lstm(x) 
        x = self.out(x)  
        return x
    
    @staticmethod
    def get_loader(x, y, batch_size):
        temp_dataset = TensorDataset(x, y)
        temp_loader = DataLoader(temp_dataset, batch_size, shuffle=False, drop_last=True)
        x_s, y_s = [], []
        for x, y in temp_loader:
            x_s.append(x)
            y_s.append(y)
        dataset = LSTM_Dataset(x_s, y_s)
        return DataLoader(dataset, 4, True)
        
    
    def _autocast(self):
        if self.use_amp:
            return torch.autocast("cuda", dtype=self.amp_dtype)
        return nullcontext()
    

    
    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, seed: int = 42, adv_steps: int = 10, adv_eps : float= 0.01, adv: bool = False):
        adv_alpha = adv_eps * 2 / adv_steps
        if self.verbose:
            print(f"Fitting decoder with input shape {train_x.shape} and output shape {train_y.shape} "
                  f"using {self.device} with amp {self.use_amp} and dtype {self.amp_dtype}")
        setup_seed(seed)
        train_x, train_y = train_x.to(self.device), train_y.to(self.device)

        # Split internal validation from the end (time-aware)
        tr_x, tr_y, val_x, val_y = self._split_time_aware(train_x, train_y)
        tr_loader = self.get_loader(tr_x, tr_y, self.batch_size)
        train_loader = self.get_loader(train_x, train_y, self.batch_size)

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
            
            for x, y in tr_loader:
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss = criterion(self(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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
            for x, y in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    loss = criterion(self(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            
            

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


    @staticmethod
    def _split_time_aware(x: torch.Tensor, y: torch.Tensor, frac=0.125):
        n = len(x)
        v = max(1, int(frac * n))
        return x[:-v], y[:-v], x[-v:], y[-v:]
    
    @staticmethod
    def _r2_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
        p = pred.detach().float().cpu().numpy()
        t = target.detach().float().cpu().numpy()
        return float(np.mean([r2_score(t[:, i], p[:, i]) for i in range(p.shape[1])]))
