"""Unstructured MSPT variant.

This module implements `MSPTUn`, an unstructured version of the Multi-Scale
Patch Transformer for point-cloud-like inputs where tokens are not mapped to a
fixed image grid.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import voxel_grid
from torch_geometric.utils import scatter
try:
    from timm.layers import trunc_normal_
except ImportError:
    from timm.models.layers import trunc_normal_
# keep your own timestep embedding import
from Embedding import timestep_embedding
import flash_attn
from flash_attn.flash_attn_interface import flash_attn_func
import logging
import torch.nn.utils.spectral_norm as spectral_norm
from torch.utils.checkpoint import checkpoint

### Experimental
ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU,
              'leaky_relu': nn.LeakyReLU(0.1), 'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Module):
    """Residual MLP used for preprocessing and feed-forward sublayers."""

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


##### RoPE utilities #####
class ChunkedGlobalPoolAttention(nn.Module):
    """Chunked self-attention with pooled global tokens.

    The input sequence is split into `V` chunks. Each chunk is summarized with
    `Q` pooled tokens, then all chunks attend to local + shared pooled tokens.
    """

    def __init__(self, dim, heads=8, V=16, Q=1, dropout=0.1, pool="mean"):
        """
        Args:
            dim:    feature dimension
            heads:  number of attention heads
            V:      number of chunks
            Q:      number of pooled tokens per chunk
            pool:   "mean" | "max" | "linear" (how to get Q reps)
            dropout: dropout prob inside attention
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.V = V
        self.Q = Q
        self.pool = pool

        if pool == "linear":
            self.pool_proj = nn.Linear(dim, Q * dim)

        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=heads,
                                          dropout=dropout,
                                          batch_first=True)

        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout)
        )

    def pad_to_multiple(self, x, multiple, dim=1):
        """Pad tensor x along `dim` to next multiple of `multiple`."""
        length = x.size(dim)
        pad_len = (multiple - (length % multiple)) % multiple
        if pad_len == 0:
            return x, 0
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad_tensor = x.new_zeros(pad_shape)
        x_padded = torch.cat([x, pad_tensor], dim=dim)
        return x_padded, pad_len

    def forward(self, features):
        """
        Args:
            features: [B, N, D]

        Returns:
            decoded:    [B, N, D]     (padding removed)
            supernodes: [B, V*Q, D]   (global set of pooled tokens after attention)
        """
        B, N, D = features.shape
        V, Q = self.V, self.Q

        # --- Step 1: pad to multiple of V ---
        x, pad_len = self.pad_to_multiple(features, V, dim=1)
        N_pad = x.size(1)

        # --- Step 2: split into V sequences ---
        seq_len = N_pad // V
        chunks = x.view(B, V, seq_len, D)  # [B,V,seq_len,D]

        # --- Step 3: pool Q tokens per chunk ---
        if self.pool == "mean":
            pooled = chunks.mean(dim=2, keepdim=True).expand(B, V, Q, D)
        elif self.pool == "max":
            pooled, _ = chunks.max(dim=2, keepdim=True)
            pooled = pooled.expand(B, V, Q, D)
        elif self.pool == "linear":
            pooled = self.pool_proj(chunks.mean(dim=2))  # [B,V,Q*D]
            pooled = pooled.view(B, V, Q, D)
        else:
            raise ValueError(f"Unknown pool method {self.pool}")

        # Flatten pooled across chunks → global pool set
        global_pooled = pooled.reshape(B, V*Q, D)  # [B, V*Q, D]

        # --- Step 4: append global pooled tokens to each chunk ---
        global_expand = global_pooled.unsqueeze(1).expand(B, V, -1, -1)  # [B,V,V*Q,D]
        chunks_with_pool = torch.cat([chunks, global_expand], dim=2)     # [B,V,seq_len+V*Q,D]

        # --- Step 5: attention per chunk ---
        seq = chunks_with_pool.view(B*V, seq_len + V*Q, D)  # [B*V, L, D]
        residual = seq
        seq = self.norm(seq)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)  # [B*V,L,D]
        seq = residual + attn_out
        seq = seq + self.ff(self.norm(seq))
        seq = seq.view(B, V, seq_len + V*Q, D)

        # --- Step 6: outputs ---
        # point features (ignore pooled tokens, only first seq_len positions)
        point_features = seq[:, :, :seq_len, :].reshape(B, N_pad, D)
        if pad_len > 0:
            point_features = point_features[:, :-pad_len, :]  # [B,N,D]

        # updated global supernodes (average across chunks to get unique set)
        supernodes = seq[:, :, -V*Q:, :]        # [B,V,V*Q,D]
        supernodes = supernodes.mean(dim=1)     # [B,V*Q,D]

        return point_features

class MSPTUnBlock(nn.Module):
    """Single MSPTUn block (attention + feed-forward + residuals)."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            V=16,
            Q=1,
            attn_pool="mean", # pool:   "mean" | "max" | "linear" (how to get Q reps)
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = ChunkedGlobalPoolAttention(dim=hidden_dim, heads=num_heads, dropout=dropout,
                                               pool=attn_pool, V=V, Q=Q)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, pos):
        """Run one block.

        Args:
            fx: Feature tensor `[B, N, D]`.
            pos: Position tensor `[B, N, 2]` (kept for API compatibility).
        """
        if self.training:
            fx = checkpoint(self.Attn, self.ln_1(fx), use_reentrant=True) + fx
        else:
            fx += self.Attn(self.ln_1(fx))
        if self.training:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class MSPTUn(nn.Module):
    """Unstructured MSPT model for point-set inputs."""

    def __init__(self,
                 pos_dim=2,
                 fx_dim=1,
                 out_dim=1,
                 num_blocks=5,
                 n_hidden=256,
                 dropout=0.1,
                 num_heads=8,
                 Time_Input=False,
                 act='gelu',
                 mlp_ratio=1,
                 ref=8,
                 unified_pos=False,
                 Q=1,
                 V=32,
                 use_rope=False,
                 attn_pool="mean"): # pool:   "mean" | "max" | "linear" (how to get Q reps)
        super().__init__()

        self.__name__ = 'MSPTUn_2D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.V = V
        self.Q = Q

        if self.unified_pos:
            self.preprocess = MLP(fx_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fx_dim + pos_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = pos_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([MSPTUnBlock(num_heads=num_heads, hidden_dim=n_hidden,
                                                    dropout=dropout, act=act, mlp_ratio=mlp_ratio,
                                                    last_layer=(_ == num_blocks - 1), out_dim=out_dim,
                                                    V=V, Q=Q, attn_pool=attn_pool)
                                     for _ in range(num_blocks)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x, batchsize=1):
        """Create reference-grid distance features for each input point."""
        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda().reshape(batchsize, self.ref * self.ref, 2)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def forward(self, pos, features, T=None):
        """Forward pass.

        Args:
            pos: Positions `[B, N, 2]`.
            features: Optional features `[B, N, fx_dim]`.
            T: Optional timestep tensor for conditional embeddings.
        """
        # Build the per-point token features consumed by attention blocks.
        if self.unified_pos:
            pos = self.get_grid(pos, pos.shape[0])
        if features is not None:
            features = torch.cat((pos, features), -1)
            features = self.preprocess(features)
        else:
            features = self.preprocess(pos)
        features = features + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, pos.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            features = features + Time_emb

        # Run blocks: each block expects features [B, N, d], and we also pass flattened pos + masks
        for block in self.blocks:
            features = block(features, pos)

        return features


if __name__ == "__main__":
    # quick smoke test shapes
    batch_size = 2
    H = 101
    W = 31
    N = H * W
    pos_dim = 2
    fx_dim = 1
    d_model = 256

    # create random grid positions in [0,1]
    pos = torch.rand(batch_size, N, pos_dim)
    fx = torch.randn(batch_size, N, fx_dim)

    model = MSPTUn(pos_dim=pos_dim,
                      fx_dim=fx_dim,
                      out_dim=1,
                      num_blocks=2,
                      n_hidden=d_model,
                      num_heads=4)

    out = model(pos, fx)
    print("output shape:", out.shape)  # should be [B, N, d_model]
    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
