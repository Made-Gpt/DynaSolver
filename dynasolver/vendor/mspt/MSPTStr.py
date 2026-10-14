"""Structured-grid MSPT variant.

This module implements `MSPTStr`, a structured 2D version of the Multi-Scale
Patch Transformer. The model assumes inputs can be reshaped to `[H, W]` grids
and uses chunked local/global attention with pooled supernodes.
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

class ChunkedGlobalPoolAttention(nn.Module):
    """Chunked self-attention with pooled global tokens.

    Tokens are split into `V` chunks. Each chunk contributes `Q` pooled tokens,
    which are shared as global context across all chunks during attention.
    """

    def __init__(self, dim, heads=8, V=16, Q=1, dropout=0.1, pool="mean", H=16, W=16):
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
        self.H, self.W = H, W
        self.pool = pool
        self.in_proj = nn.Conv2d(dim, self.dim, 1)

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
        feat_grid = features.view(B, self.H, self.W, D).permute(0, 3, 1, 2).contiguous()  # B, D, H, W
        features = self.in_proj(feat_grid).permute(0, 2, 3, 1).contiguous().view(B, N, self.dim)  # B, N, C
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

class MSPTStrBlock(nn.Module):
    """Single MSPTStr block (attention + feed-forward + residuals)."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            H=85,
            W=85,
            V=16,
            Q=1
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = ChunkedGlobalPoolAttention(dim=hidden_dim, heads=num_heads, dropout=dropout,
                                               pool="mean", V=V, Q=Q, H=H, W=W)
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


class MSPTStr(nn.Module):
    """Structured MSPT model for 2D fields."""

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
                 H=16,
                 W=16,
                 Q=1,
                 V=32,
                 use_rope=False):
        super().__init__()

        self.__name__ = 'MSPTStr_2D'
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        self.V = V
        self.Q = Q

        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fx_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fx_dim + pos_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = pos_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([MSPTStrBlock(num_heads=num_heads, hidden_dim=n_hidden,
                                                    dropout=dropout, act=act, mlp_ratio=mlp_ratio,
                                                    last_layer=(_ == num_blocks - 1), H=H, W=W, out_dim=out_dim,
                                                    V=V, Q=Q)
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

    def get_grid(self, batchsize=1):
        """Build a dense positional feature grid used by `unified_pos` mode."""
        # Infer device from model parameters so buffers follow model placement.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1)  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float, device=device)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float, device=device)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1)  # B ref ref 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, fx, T=None):
        """Forward pass.

        Args:
            x: Positions `[B, N, 2]`.
            fx: Optional input features `[B, N, fx_dim]`.
            T: Optional timestep tensor for conditional embeddings.
        """
        B, N, Dp = x.shape
        assert N == (self.H * self.W), "N must equal H*W for the current architecture"


        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1, 1).reshape(x.shape[0], self.H * self.W, self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx, x)

        return fx


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

    model = MSPTStr(pos_dim=pos_dim,
                      fx_dim=fx_dim,
                      out_dim=1,
                      num_blocks=2,
                      n_hidden=d_model,
                      num_heads=4,
                      H=H,
                      W=W)

    out = model(pos, fx)
    print("output shape:", out.shape)  # should be [B, N, d_model]
    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
