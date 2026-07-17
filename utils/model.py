import torch.nn as nn
import torch
from utils.constants import CEBRA_DIR
import sys
sys.path.append(str(CEBRA_DIR))
from cebra.models import Offset36Dropoutv2, Offset10Model
import cebra.models.layers as cebra_layers
from utils.augmentation import GaussianSmoothing, Unfolder
from utils.mlp import TwoLayerMLP
import torch.nn.functional as F

class Encoder_Decoder(nn.Module):
    """
    Input:  (B, T, F)
    Output: logits for CTC of shape (T, B, C)
    """

    def __init__(self, neural_dim, cebra_out_dim, kernel, stride, num_classes, rnn_hidden, rnn_layers, rnn_dr = 0.4, rnn_bidir=True, cebra_unfolder=False, gru = False, smooth_width=2.0, gauss_in=True, no_rnn=False):
        super().__init__()
        def init_cebra(in_features):
            import sys
            sys.path.append('CEBRA-main')
            from cebra.models import Offset36Dropoutv2
            return Offset36Dropoutv2(in_features, 256, cebra_out_dim)
        
        current_dim = neural_dim
        self.cebra_unfolder = cebra_unfolder
        self.smoother = (GaussianSmoothing(neural_dim, 20, smooth_width, dim=1)) if gauss_in else (nn.Identity())

        if cebra_unfolder:
            self.cebra = init_cebra(current_dim)
            current_dim = cebra_out_dim

        self.unfolder = Unfolder(kernel, stride)
        current_dim *= kernel

        if not cebra_unfolder:
            self.cebra = init_cebra(current_dim)
            current_dim = cebra_out_dim

        if not no_rnn:
            if gru:
                self.rnn = nn.GRU(
                    current_dim, 
                    rnn_hidden,
                    rnn_layers,
                    batch_first=True, 
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                    )
            else:
                self.rnn = nn.LSTM(
                    current_dim,
                    rnn_hidden, 
                    rnn_layers,
                    batch_first=True,
                    bidirectional=rnn_bidir, 
                    dropout=rnn_dr
                )
            current_dim = rnn_hidden * (2 if rnn_bidir else 1)
        else:
            self.rnn = lambda x: (x, None)
        
        self.final_decoder = nn.Linear(current_dim, num_classes)
    
    def _apply_cebra(self, x, lengths):
        """Helper to permute, pad, forward CEBRA, and permute back."""
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = F.pad(x, (18, 17), mode='replicate')
        x = self.cebra(x).permute(0, 2, 1)  # (B, T, C)
        self.embeddings = x
        self.emb_lengths = lengths
        return x
    
    def get_cebra_embs(self):
        return self.embeddings, self.emb_lengths
    
    def forward(self, x, lengths):
        x = self.smoother(x)
        if self.cebra_unfolder:
            x = self._apply_cebra(x, lengths)
        x, lengths = self.unfolder(x, lengths)
        if not self.cebra_unfolder:
            x = self._apply_cebra(x, lengths)
        x, _ = self.rnn(x)
        x = self.final_decoder(x)
        return x, lengths

