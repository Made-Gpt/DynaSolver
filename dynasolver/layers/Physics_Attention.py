import torch.nn as nn
import torch
from einops import rearrange, repeat
import numpy as np


class Physics_Attention_Irregular_Mesh(nn.Module):
    ## for irregular meshes in 1D, 2D or 3D space
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        slice_num=64,
        shapelist=None,
        slice_mode: str = "legacy",
        use_flash_attn: bool = False,
    ):
        super().__init__()
        if slice_mode not in ("legacy", "fast"):
            raise ValueError(f"slice_mode must be 'legacy' or 'fast', got {slice_mode!r}")
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = float(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.slice_mode = slice_mode
        self.slice_num = slice_num
        self.input_dim = dim
        self.use_flash_attn = bool(use_flash_attn)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def compute_slice_weights(self, x: torch.Tensor) -> torch.Tensor:
        """W branch only — identical for legacy and fast."""
        B, N, _ = x.shape
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()  # B H N C
        return self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G

    def slice_to_tokens(self, x: torch.Tensor, slice_weights: torch.Tensor) -> torch.Tensor:
        """Pool points → slice tokens. Fast: (W^T X) A; legacy: W^T (X A)."""
        if self.slice_mode == "fast":
            return self._fast_slice_to_tokens(x, slice_weights)
        return self._legacy_slice_to_tokens(x, slice_weights)

    def _legacy_slice_to_tokens(self, x: torch.Tensor, slice_weights: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
        fx_mid = fx_mid.permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        return slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

    def _fast_slice_to_tokens(self, x: torch.Tensor, slice_weights: torch.Tensor) -> torch.Tensor:
        # Pool raw X first, then apply in_project_fx on G tokens.
        # Bias uses s/(s+eps) so affine matches legacy under the shared +eps norm.
        slice_norm = slice_weights.sum(2)  # B H G
        denom = (slice_norm + 1e-5)[:, :, :, None]
        raw_slice = torch.einsum("bnc,bhng->bhgc", x, slice_weights) / denom  # B H G C

        weight = self.in_project_fx.weight  # [H*dh, C]
        w = weight.view(self.heads, self.dim_head, self.input_dim)  # H dh C
        slice_token = torch.einsum("bhgc,hdc->bhgd", raw_slice, w)
        if self.in_project_fx.bias is not None:
            bias = self.in_project_fx.bias.view(self.heads, self.dim_head)
            bias_scale = (slice_norm / (slice_norm + 1e-5))[:, :, :, None]
            slice_token = slice_token + bias[None, :, None, :] * bias_scale
        return slice_token

    def deslice_to_out(
        self,
        out_slice_token: torch.Tensor,
        slice_weights: torch.Tensor,
        *,
        use_bias: bool = True,
        use_dropout: bool = True,
    ) -> torch.Tensor:
        """Map slice tokens → point features via to_out. Fast: W (A S) + b."""
        if self.slice_mode == "fast":
            return self._fast_deslice_to_out(
                out_slice_token, slice_weights, use_bias=use_bias, use_dropout=use_dropout
            )
        return self._legacy_deslice_to_out(
            out_slice_token, slice_weights, use_bias=use_bias, use_dropout=use_dropout
        )

    def _legacy_deslice_to_out(
        self,
        out_slice_token: torch.Tensor,
        slice_weights: torch.Tensor,
        *,
        use_bias: bool = True,
        use_dropout: bool = True,
    ) -> torch.Tensor:
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        linear = self.to_out[0]
        bias = linear.bias if use_bias else None
        out_x = torch.nn.functional.linear(out_x, linear.weight, bias)
        if use_dropout:
            out_x = self.to_out[1](out_x)
        return out_x

    def _fast_deslice_to_out(
        self,
        out_slice_token: torch.Tensor,
        slice_weights: torch.Tensor,
        *,
        use_bias: bool = True,
        use_dropout: bool = True,
    ) -> torch.Tensor:
        # Apply to_out.weight per-head on tokens, deslice+sum heads, add bias once.
        linear = self.to_out[0]
        c_out = linear.weight.shape[0]
        w = linear.weight.view(c_out, self.heads, self.dim_head)  # C H dh
        projected = torch.einsum("bhgd,chd->bhgc", out_slice_token, w)  # B H G C
        out_x = torch.einsum("bhgc,bhng->bnc", projected, slice_weights)
        if use_bias and linear.bias is not None:
            out_x = out_x + linear.bias
        if use_dropout:
            out_x = self.to_out[1](out_x)
        return out_x

    def spatial_attend_tokens(self, slice_token: torch.Tensor) -> torch.Tensor:
        """Transolver attention among slice tokens only (no deslice).

        Args:
            slice_token: ``[B, H, G, D]`` head-split tokens from ``slice_to_tokens``.
        Returns:
            Attended tokens with the same shape ``[B, H, G, D]``.
        """
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        if self.use_flash_attn:
            from layers.flash_attn_utils import attention_qkv

            return attention_qkv(
                q_slice_token,
                k_slice_token,
                v_slice_token,
                dropout_p=self.dropout_p,
                causal=False,
                use_flash=True,
            )
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        return torch.matmul(attn, v_slice_token)  # B H G D

    def forward(self, x, return_feature=False, vis=False):
        # B N C — spatial-only path (slice → token attn → deslice); no temporal.
        slice_weights = self.compute_slice_weights(x)
        if vis:
            np.save("slice_weights.npy", slice_weights.detach().cpu().numpy())
        slice_token = self.slice_to_tokens(x, slice_weights)
        out_slice_token = self.spatial_attend_tokens(slice_token)
        out_x = self.deslice_to_out(out_slice_token, slice_weights)
        if return_feature:
            return out_x, slice_token
        return out_x


class Physics_Attention_Structured_Mesh_1D(nn.Module):
    ## for structured mesh in 1D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.length = shapelist[0]

        self.in_project_x = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv1d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_feature=False):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.length, C).contiguous().permute(0, 2, 1).contiguous()  # B C N

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        if return_feature:
            return self.to_out(out_x), slice_token
        else:
            return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, shapelist=None, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = shapelist[0]
        self.W = shapelist[1]

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    ## for structured mesh in 3D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=32, shapelist=None, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = shapelist[0]
        self.W = shapelist[1]
        self.D = shapelist[2]

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, self.D, C).contiguous().permute(0, 4, 1, 2, 3).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 4, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)
