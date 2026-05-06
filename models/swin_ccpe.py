"""
Swin Transformer with CCPE — Cross Contrast Patch Embedding backbone.
Standalone implementation (no mmdet dependency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial

from .ccpe_module import CrossContrastPatchEmbed


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)

        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H, W, attn_mask=None):
        def _inner(x):
            B, L, C = x.shape
            shortcut = x
            x = self.norm1(x)
            x = x.view(B, H, W, C)

            # Pad
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
            Hp, Wp = x.shape[1], x.shape[2]

            # Cyclic shift
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

            # Window partition
            x_windows = window_partition(x, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

            # Attention
            attn_windows = self.attn(x_windows, mask=attn_mask)

            # Merge
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
            x = window_reverse(attn_windows, self.window_size, Hp, Wp)

            # Reverse shift
            if self.shift_size > 0:
                x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            x = x[:, :H, :W, :].contiguous().view(B, H * W, C)
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        if self.use_checkpoint and x.requires_grad:
            return cp.checkpoint(_inner, x, use_reentrant=False)
        return _inner(x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class PatchMerging(nn.Module):
    """Downsample 2x by merging 2×2 patches."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        # Pad if odd
        if H % 2 == 1:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if W % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                use_checkpoint=use_checkpoint
            ) for i in range(depth)
        ])
        self.downsample = downsample

    def forward(self, x, H, W):
        # Compute attention mask for shifted windows
        attn_mask = self._compute_mask(H, W, x.device) if any(
            b.shift_size > 0 for b in self.blocks) else None

        for blk in self.blocks:
            mask = attn_mask if blk.shift_size > 0 else None
            x = blk(x, H, W, mask)

        x_out = x  # Features before downsampling
        H_out, W_out = H, W

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W, x_out, H_out, W_out

    def _compute_mask(self, H, W, device):
        ws = self.blocks[0].window_size
        shift = ws // 2
        Hp = int(((H - 1) // ws + 1) * ws)
        Wp = int(((W - 1) // ws + 1) * ws)

        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
        return attn_mask


class SwinTransformerCCPE(nn.Module):
    """
    Swin Transformer with Cross Contrast Patch Embedding.

    Architecture: CCPE → 4 Swin stages → multi-scale features
    Default: Swin-Tiny (depths=[2,2,6,2], embed_dim=96)
    """

    def __init__(self, in_channels=3, embed_dims=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, patch_size=4, out_indices=(1, 2, 3),
                 use_checkpoint=False, contrast_steps=None,
                 pretrained=None):
        super().__init__()
        self.out_indices = out_indices
        self.num_stages = len(depths)

        # CCPE patch embedding
        self.patch_embed = CrossContrastPatchEmbed(
            in_channels=in_channels, embed_dims=embed_dims,
            patch_size=patch_size, stride=patch_size,
            contrast_steps=contrast_steps
        )

        # Stochastic depth
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        # Build stages
        self.stages = nn.ModuleList()
        dim = embed_dims
        dp_idx = 0
        for i in range(self.num_stages):
            stage = SwinStage(
                dim=dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[dp_idx:dp_idx + depths[i]],
                downsample=PatchMerging(dim) if i < self.num_stages - 1 else None,
                use_checkpoint=use_checkpoint
            )
            self.stages.append(stage)
            dp_idx += depths[i]
            if i < self.num_stages - 1:
                dim *= 2

        # Norms for output features
        self.norms = nn.ModuleList()
        for i in out_indices:
            out_dim = embed_dims * (2 ** i)
            self.norms.append(nn.LayerNorm(out_dim))

        self.pretrained = pretrained
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def load_pretrained(self, checkpoint_path):
        """Load pretrained Swin weights (skip CCPE layers)."""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # Filter incompatible keys
        model_dict = self.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            # Map from official Swin keys to our keys
            new_k = self._map_key(k)
            if new_k and new_k in model_dict and model_dict[new_k].shape == v.shape:
                pretrained_dict[new_k] = v
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} pretrained params")
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

    def _map_key(self, key):
        """Map official Swin Transformer keys to our naming.

        Local SwinCCPE differs from the official Swin checkpoint in two
        ways: (1) we use CCPE in place of the standard patch_embed, so
        those source weights are skipped, and (2) the per-block MLP is
        an ``nn.Sequential`` with positional indices (``mlp.0``, ``mlp.3``)
        rather than the named ``mlp.fc1`` / ``mlp.fc2`` modules used by
        the official Swin / timm implementations. We also skip top-level
        ``norm.*`` and the ImageNet classifier ``head.*`` (none exist in
        our backbone), and ``attn_mask`` buffers (rebuilt at runtime).
        """
        # Skip CCPE-incompatible or absent modules
        if key.startswith('patch_embed') or key.startswith('head'):
            return None
        if key in ('norm.weight', 'norm.bias'):
            return None
        if key.endswith('.attn_mask'):
            return None
        # layers.X -> stages.X
        key = key.replace('layers.', 'stages.')
        # mlp.fc1.* -> mlp.0.* (first nn.Linear in nn.Sequential)
        # mlp.fc2.* -> mlp.3.* (Linear after GELU + Dropout)
        key = key.replace('.mlp.fc1.', '.mlp.0.')
        key = key.replace('.mlp.fc2.', '.mlp.3.')
        return key

    def forward(self, x):
        """
        Returns multi-scale features at out_indices stages.
        """
        x, hw_shape = self.patch_embed(x)
        H, W = hw_shape

        outs = []
        for i, stage in enumerate(self.stages):
            x, H, W, x_out, H_out, W_out = stage(x, H, W)
            if i in self.out_indices:
                norm_idx = self.out_indices.index(i) if isinstance(self.out_indices, (list, tuple)) else 0
                out = self.norms[norm_idx](x_out)
                B, _, C = out.shape
                out = out.view(B, H_out, W_out, C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)
