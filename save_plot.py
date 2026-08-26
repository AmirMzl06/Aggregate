import os
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "RRESULT"
IMG_DIR = os.path.join(RESULT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)
MODELS = ["CLEAN", "ADV"]

attrs = {}
for model in MODELS:
    path = os.path.join(RESULT_DIR, f"{model}_attr.npz")
    data = np.load(path)
    jf = np.abs(data["jf"])
    jfinv = np.abs(data["jfinv"])
    jf = jf / (jf.sum() + 1e-12)
    jfinv = jfinv / (jfinv.sum() + 1e-12)
    attrs[model] = {"jf": jf, "jfinv": jfinv}
    print(model, "JF shape:", jf.shape, "JFINV shape:", jfinv.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ims = []
for ax, model in zip(axes, MODELS):
    mat = attrs[model]["jf"]
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ims.append(im)
    ax.set_title(f"{model} - Jacobian Forward")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Latent dimension")
fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.8)
plt.tight_layout()
save_path = os.path.join(IMG_DIR, "CLEAN_vs_ADV_JF.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", save_path)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ims = []
for ax, model in zip(axes, MODELS):
    mat = attrs[model]["jfinv"]
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ims.append(im)
    ax.set_title(f"{model} - Jacobian Inverse")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Latent dimension")
fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.8)
plt.tight_layout()
save_path = os.path.join(IMG_DIR, "CLEAN_vs_ADV_JFINV.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", save_path)

np.savez(os.path.join(IMG_DIR, "normalized_attributions.npz"),
         CLEAN_JF=attrs["CLEAN"]["jf"],
         ADV_JF=attrs["ADV"]["jf"],
         CLEAN_JFINV=attrs["CLEAN"]["jfinv"],
         ADV_JFINV=attrs["ADV"]["jfinv"])
print("Done.")
