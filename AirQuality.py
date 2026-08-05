import os
import gc
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from utils.min_distance import min_l2_distance
from utils.constants import CEBRA_DIR, DATA_DIR

import sys
sys.path.insert(0, str(CEBRA_DIR))

import cebra
import cebra.attribution
from cebra import CEBRA

# ==========================================
# Configs
# ==========================================
AIR_FILE = os.path.join(DATA_DIR, "AirQualityUCI.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1024
MAX_ITER = 2500
OUTPUT_DIM = 16
ATTR_BATCH_SIZE = 128

DECODER_HIDDEN = 64
DECODER_EPOCHS = 10000
PATIENCE = 1000

OUT = "outputs/AirQuality"
IMG = "images/AirQuality"

os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# Decoder
# ==========================================
class Decoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, output_dim=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ==========================================
# Utils
# ==========================================
def clean(*items):
    for a in items:
        try:
            del a
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_air():
    df = pd.read_csv(AIR_FILE, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all")
    if "Date" in df.columns:
        df = df.drop(columns=["Date", "Time"])
    
    df = df.replace(-200, np.nan)
    df = df.interpolate(limit_direction="both")
    # Drop any remaining NaNs after interpolation
    df = df.dropna()
    
    X = df.values.astype(np.float32)
    print("DATA SHAPE:", X.shape)
    print("COLUMNS:", df.columns.tolist())
    return X

def time_label(n):
    return np.linspace(0, 1, n, dtype=np.float32).reshape(-1, 1)

def mean_r2(y, p):
    vals = []
    for i in range(y.shape[1]):
        vals.append(r2_score(y[:, i], p[:, i]))
    return float(np.mean(vals))

def embed(model, x):
    z = model.transform(torch.tensor(x, dtype=torch.float32))
    if torch.is_tensor(z):
        return z.detach().cpu().numpy()
    return np.asarray(z)

def build_model(adv, eps):
    return CEBRA(
        batch_size=BATCH_SIZE,
        temperature=0.4,
        model_architecture="offset10-model",
        time_offsets=10,
        max_iterations=MAX_ITER,
        output_dimension=OUTPUT_DIM,
        verbose=True,
        training_mode="adversarial" if adv else "clean",
        adv_alpha=eps/5 if adv else 0,
        adv_epsilon=eps if adv else 0,
        adv_steps=10 if adv else 0,
        attack_norm="linf",
        num_hidden_units=32,
        device="cuda_if_available"
    )

def reduce_attr(a):
    if torch.is_tensor(a):
        a = a.detach().cpu()
    a = torch.abs(a)
    if a.ndim == 3:
        a = a.mean(0)
    elif a.ndim == 1:
        a = a[None, :]
    return a.numpy().astype(np.float32)

def plot_attr(mat, path, title):
    plt.figure(figsize=(10, 5))
    plt.imshow(mat, aspect="auto", cmap="cividis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Latent")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ==========================================
# Core Functions
# ==========================================
def compute_jacobian(model, X_train, tag):
    net = model.solver_.model.to(DEVICE)
    net.eval()
    inp = torch.tensor(X_train, dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    # Init attribution
    method = cebra.attribution.init(
        name="jacobian-based-batched", 
        model=net, 
        input_data=inp, 
        output_dimension=OUTPUT_DIM
    )
    
    # Ensure batch_size isn't larger than dataset
    bs = min(ATTR_BATCH_SIZE, len(X_train))
    out = method.compute_attribution_map(batch_size=bs)
    
    jf = reduce_attr(out["jf"])
    jfi_key = "jf-inv-svd" if "jf-inv-svd" in out else "jf-inv"
    jfi = reduce_attr(out[jfi_key])
    
    jf_score = np.abs(jf).mean(0)
    jfi_score = np.abs(jfi).mean(0)
    
    # Sqrt(#features)
    k = int(np.sqrt(X_train.shape[1]))
    
    top_jf = np.argsort(jf_score)[::-1][:k]
    top_jfi = np.argsort(jfi_score)[::-1][:k]
    
    torch.save({"jf": torch.tensor(jf), "jf_inv": torch.tensor(jfi)}, os.path.join(OUT, f"{tag}_jacobians.pt"))
    np.savez(os.path.join(OUT, f"{tag}_attrs.npz"), jf=jf, jfinv=jfi, top_jf=top_jf, top_jfi=top_jfi)
    
    plot_attr(jf, os.path.join(IMG, f"{tag}_JC.png"), f"{tag} JC")
    plot_attr(jfi, os.path.join(IMG, f"{tag}_JC_INV.png"), f"{tag} JC inverse")
    
    print(f"[{tag}] Top JC ({k} features):", top_jf)
    print(f"[{tag}] Top JC INV ({k} features):", top_jfi)
    
    return {"top_jf": top_jf, "top_jfi": top_jfi}

def train_decoder_reconstruction(model, X_in_train, X_in_test, Y_tg_train, Y_tg_test, tag):
    # Split train into Train and Validation for Early Stopping (No Test Leakage)
    X_tr, X_val, Y_tr, Y_val = train_test_split(X_in_train, Y_tg_train, test_size=0.15, random_state=42, shuffle=False)
    
    z_train = torch.tensor(embed(model, X_tr)).float().to(DEVICE)
    z_val   = torch.tensor(embed(model, X_val)).float().to(DEVICE)
    z_test  = torch.tensor(embed(model, X_in_test)).float().to(DEVICE)
    
    y_train = torch.tensor(Y_tr).float().to(DEVICE)
    y_val   = torch.tensor(Y_val).float().to(DEVICE)
    y_test  = torch.tensor(Y_tg_test).float().to(DEVICE)
    
    target_dim = Y_tg_train.shape[1]
    dec = Decoder(input_dim=OUTPUT_DIM, hidden_dim=DECODER_HIDDEN, output_dim=target_dim).to(DEVICE)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    best_val_r2 = -999
    best_state = None
    patience = 0
    
    for epoch in range(DECODER_EPOCHS):
        dec.train()
        opt.zero_grad()
        pred = dec(z_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        
        dec.eval()
        with torch.no_grad():
            val_pred = dec(z_val).cpu().numpy()
            y_val_np = y_val.cpu().numpy()
            
        r = mean_r2(y_val_np, val_pred)
        
        if r > best_val_r2:
            best_val_r2 = r
            best_state = copy.deepcopy(dec.state_dict())
            patience = 0
        else:
            patience += 1
            
        if patience > PATIENCE:
            break
            
    # Load best state and Evaluate on TEST
    dec.load_state_dict(best_state)
    torch.save(dec.state_dict(), os.path.join(OUT, f"{tag}_decoder.pth"))
    
    dec.eval()
    with torch.no_grad():
        test_pred = dec(z_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()
        
    final_test_r2 = mean_r2(y_test_np, test_pred)
    print(f"[{tag}] Decoder Test R2: {final_test_r2:.4f}")
    
    clean(dec, z_train, z_val, z_test, y_train, y_val, y_test)
    return final_test_r2

def train_full(X_train, time_train, X_test, tag, adv):
    eps = float(min_l2_distance(torch.tensor(X_train))) / 2
    eps = max(eps, 1e-6)
    
    model = build_model(adv, eps)
    model.fit(X_train, time_train) # Training ONLY on Train Split
    model.save(os.path.join(OUT, f"{tag}.pth"))
    
    attr = compute_jacobian(model, X_train, tag)
    
    # Decoder uses full original features as target
    r2 = train_decoder_reconstruction(
        model=model, 
        X_in_train=X_train, X_in_test=X_test, 
        Y_tg_train=X_train, Y_tg_test=X_test, 
        tag=tag
    )
    
    clean(model)
    return {"top_jf": attr["top_jf"], "top_jfi": attr["top_jfi"], "r2": r2}

def train_reduced(X_train_full, X_test_full, time_train, source, attr_name, features, retrain_adv):
    tag = f"src_{source}_attr_{attr_name}_retrain_{'ACORN' if retrain_adv else 'CEBRA'}"
    
    # Subsetting features for input
    X_train_sub = X_train_full[:, features]
    X_test_sub = X_test_full[:, features]
    
    eps = float(min_l2_distance(torch.tensor(X_train_sub))) / 2
    eps = max(eps, 1e-6)
    
    model = build_model(retrain_adv, eps)
    model.fit(X_train_sub, time_train)
    
    # CRITICAL FIX: The model receives X_sub, but Decoder must predict X_full
    r2 = train_decoder_reconstruction(
        model=model, 
        X_in_train=X_train_sub, X_in_test=X_test_sub, 
        Y_tg_train=X_train_full, Y_tg_test=X_test_full, 
        tag=tag
    )
    
    print(f"REDUCED | Src: {source} | Attr: {attr_name} | Retrain: {'ACORN' if retrain_adv else 'CEBRA'} | R2: {r2:.4f}")
    clean(model)
    return r2

# ==========================================
# Main execution
# ==========================================
def main():
    X = load_air()
    time = time_label(len(X))
    
    # Splitting data FIRST
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    time_train = time[:split_idx]
    
    results = {}
    
    print("\n================ FULL CEBRA ================")
    results["CEBRA"] = train_full(X_train, time_train, X_test, "full_CEBRA", False)
    
    print("\n================ FULL ACORN ================")
    results["ACORN"] = train_full(X_train, time_train, X_test, "full_ACORN", True)
    
    rows = []
    print("\n================ 8 STATES ABLATION ================")
    for source in ["CEBRA", "ACORN"]:
        for attr in ["JC", "JC_INV"]:
            
            topk = results[source]["top_jf"] if attr == "JC" else results[source]["top_jfi"]
            
            for retrain_adv in [False, True]:
                retrain_name = "ACORN" if retrain_adv else "CEBRA"
                
                print("\n" + "-" * 40)
                print(f"Source: {source} | Attr: {attr} | Retrain: {retrain_name}")
                print(f"Features: {topk}")
                
                r2 = train_reduced(
                    X_train_full=X_train, 
                    X_test_full=X_test, 
                    time_train=time_train, 
                    source=source, 
                    attr_name=attr, 
                    features=topk, 
                    retrain_adv=retrain_adv
                )
                
                rows.append({
                    "source_model": source,
                    "attribute": attr,
                    "retrained_model": retrain_name,
                    "topk_features": str(topk.tolist()),
                    "k": len(topk),
                    "r2": r2
                })
                
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "AirQuality_topK_8states_summary.csv")
    df.to_csv(path, index=False)
    
    print("\n================ FINAL RESULT ================")
    print(df.to_string(index=False))
    print(f"\nSaved Results to: {path}")

if __name__ == "__main__":
    main()
