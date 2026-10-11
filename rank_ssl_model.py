"""Self-supervised nested-subset information ranking for neural populations: model and losses.

Idea (after Liu, van de Weijer, Bagdanov, "Exploiting Unlabeled Data in CNNs by Self-supervised
Learning to Rank", TPAMI 2019): nested subsets of neurons A_k1 ⊃ A_k2 ⊃ ... carry a known order of
information about a held-out reference population B, I(A_k1; B) >= I(A_k2; B) >= ...  The encoder is
trained to (1) keep as much information about B as possible and (2) respect that order, with no
labels, no contrastive pairs and no reconstruction.

Encoder f(neuron windows, neuron ids) -> z, for any number of neurons:
    per-neuron temporal Conv1D (weights shared by all neurons) -> one token per neuron
    + learnable neuron identity embedding
    -> self-attention across neurons (interactions / coordination between neurons)
    -> attention pooling with learned queries -> z in R^d_out

Losses:
    gaussian_mi(z, y)   Gaussian (log-det) mutual information between a batch of embeddings and the
                        reference population's activity (closed form on batch covariances)
    rank_loss(M)        pairwise hinge ranking over the MI of the nested levels (margin eps)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalStem(nn.Module):
    """(M, 1, W) activity window of one neuron -> (M, d_tok). The same weights for every neuron."""

    def __init__(self, window, channels=16, d_tok=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, 5, padding=2), nn.GELU(),
            nn.Conv1d(channels, 2 * channels, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(2 * channels, 2 * channels, 5, stride=2, padding=2), nn.GELU(),
        )
        with torch.no_grad():
            n_out = self.net(torch.zeros(1, 1, window)).numel()
        self.proj = nn.Linear(n_out, d_tok)  # keeps where in the window the activity happened

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class NeuronSetEncoder(nn.Module):
    def __init__(self, n_neurons, window=36, d_tok=64, conv_channels=16, n_layers=2, n_heads=4,
                 n_queries=4, d_out=32, dropout=0.0):
        super().__init__()
        self.window = window
        self.stem = TemporalStem(window, conv_channels, d_tok)
        self.neuron_id = nn.Embedding(n_neurons, d_tok)
        nn.init.normal_(self.neuron_id.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d_tok, n_heads, 2 * d_tok, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        try:
            self.attn = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        except TypeError:  # older torch
            self.attn = nn.TransformerEncoder(layer, n_layers)
        self.queries = nn.Parameter(torch.randn(n_queries, d_tok) * 0.02)
        self.pool = nn.MultiheadAttention(d_tok, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(n_queries * d_tok)
        self.out = nn.Linear(n_queries * d_tok, d_out)

    def forward(self, x, ids):
        """x: (S, n, W) activity windows of n neurons; ids: (S, n) neuron indices. Returns (S, d_out)."""
        s, n, w = x.shape
        tok = self.stem(x.reshape(s * n, 1, w)).reshape(s, n, -1) + self.neuron_id(ids)
        tok = self.attn(tok)
        pooled = self.pool(self.queries.expand(s, -1, -1), tok, tok, need_weights=False)[0]
        return self.out(self.norm(pooled.reshape(s, -1)))


def _logdet_pd(m):
    return 2.0 * torch.log(torch.diagonal(torch.linalg.cholesky(m), dim1=-2, dim2=-1)).sum(-1)


def gaussian_mi(z, y, eps=1e-3):
    """I(z; y) in nats under a joint-Gaussian model, estimated on the batch (rows = samples).

    I = 1/2 [log det C_zz + log det C_yy - log det C_[z,y]] on per-dimension standardized data,
    so it is invariant to any invertible linear map of z (scaling z cannot inflate it), it is 0 for
    a collapsed z, and it needs no negatives and no decoder. eps regularizes the correlation matrices.
    Computed in float64 with Cholesky factorizations.
    """
    z, y = z.double(), y.double()
    z = (z - z.mean(0)) / z.std(0).clamp_min(1e-5)
    y = (y - y.mean(0)) / y.std(0).clamp_min(1e-5)
    j = torch.cat([z, y], 1)
    c = j.T @ j / (j.shape[0] - 1) + eps * torch.eye(j.shape[1], dtype=j.dtype, device=j.device)
    dz = z.shape[1]
    return 0.5 * (_logdet_pd(c[:dz, :dz]) + _logdet_pd(c[dz:, dz:]) - _logdet_pd(c))


def rank_loss(m, margin, order=None):
    """Pairwise hinge ranking (Eq. 4 of Liu et al.) over level scores m (level 0 = most neurons).

    For every pair i < j it asks m[i] >= m[j] + margin. `order` permutes the levels first (shuffled-
    ranking control). Returns (mean hinge over pairs, fraction of pairs in the wrong order).
    """
    if order is not None:
        m = m[order]
    i, j = torch.triu_indices(len(m), len(m), offset=1, device=m.device)
    hinge = F.relu(m[j] - m[i] + margin)
    return hinge.mean(), (m[j] > m[i]).double().mean()


@torch.no_grad()
def effective_rank(z):
    """exp(entropy) of the eigenvalue spectrum of the correlation matrix of z (1 = collapsed to a line)."""
    z = z.double()
    z = (z - z.mean(0)) / z.std(0).clamp_min(1e-5)
    ev = torch.linalg.eigvalsh(z.T @ z / (len(z) - 1)).clamp_min(0)
    p = ev / ev.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
