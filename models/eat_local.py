"""
Minimal local EAT backbone used when Transformers cannot load the model.
"""

import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x, x)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbedNew(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=16):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=to_2tuple(stride))

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class FixedPositionalEncoder(nn.Module):
    def __init__(self, pos_embed):
        super().__init__()
        self.positions = pos_embed

    def forward(self, x, padding_mask=None):
        return self.positions


class AltAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cosine_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cosine_attention = cosine_attention
        if cosine_attention:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

    def forward(self, x, padding_mask=None, alibi_bias=None):
        batch_size, seq_len, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, seq_len, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        dtype = q.dtype
        if self.cosine_attention:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            max_logit = torch.log(torch.tensor(1.0 / 0.01, device=x.device))
            attn = attn * torch.clamp(self.logit_scale, max=max_logit).exp()
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias
        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class AltBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        ffn_targets=False,
    ):
        super().__init__()
        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets
        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=mlp_drop)
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
            r = x
            x = self.mlp(self.norm2(x))
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
            r = self.norm1(x)
            x = self.mlp(r)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
        if not self.ffn_targets:
            t = x
        return x, t


class LocalEAT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_encoder = PatchEmbedNew(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            embed_dim=config["embed_dim"],
            stride=config["stride"],
        )
        self.extra_tokens = nn.Parameter(torch.zeros(1, 1, config["embed_dim"]))
        self.pos_drop = nn.Dropout(p=config["drop_rate"], inplace=True)
        nn.init.trunc_normal_(self.extra_tokens, std=0.02)
        self.fixed_positional_encoder = (
            FixedPositionalEncoder(self.build_sincos_pos_embed()) if config["fixed_positions"] else None
        )
        norm_layer = partial(nn.LayerNorm, eps=config["norm_eps"], elementwise_affine=config["norm_affine"])
        dpr = np.linspace(config["start_drop_path_rate"], config["end_drop_path_rate"], config["depth"])
        self.blocks = nn.ModuleList(
            [
                AltBlock(
                    config["embed_dim"],
                    config["num_heads"],
                    config["mlp_ratio"],
                    qkv_bias=config["qkv_bias"],
                    drop=config["drop_rate"],
                    attn_drop=config["attn_drop_rate"],
                    mlp_drop=config["activation_dropout"],
                    post_mlp_drop=config["post_mlp_drop"],
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_norm_first=config["layer_norm_first"],
                    ffn_targets=True,
                )
                for i in range(config["depth"])
            ]
        )
        self.pre_norm = norm_layer(config["embed_dim"])

    def build_sincos_pos_embed(self):
        width = self.config["mel_bins"] // self.config["patch_size"]
        max_length = self.config["max_length"]
        embed_dim = self.config["embed_dim"]
        pos_embed = nn.Parameter(torch.zeros(1, max_length * width, embed_dim), requires_grad=False)
        emb = get_2d_sincos_pos_embed_flexible(embed_dim, (max_length, width), cls_token=False)
        pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))
        return pos_embed

    def encode(self, x, layer=None):
        batch_size = x.shape[0]
        x = self.local_encoder(x)
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, None)[:, : x.size(1), :]
        x = torch.cat((self.extra_tokens.expand(batch_size, -1, -1), x), dim=1)
        x = self.pre_norm(x)
        x = self.pos_drop(x)
        if layer is not None:
            depth = len(self.blocks)
            if layer < 0:
                layer = depth + 1 + layer
            if layer < 0 or layer > depth:
                raise ValueError(f"EAT layer must be in [0, {depth}] or negative equivalent, got {layer}")
            if layer == 0:
                return x

        for block_idx, block in enumerate(self.blocks, start=1):
            x, _ = block(x)
            if layer is not None and block_idx == layer:
                return x
        return x

    def extract_features(self, x, layer=None):
        return self.encode(x, layer=layer)


def _resolve_model_files(model_name_or_path):
    if os.path.isfile(model_name_or_path) and model_name_or_path.endswith(".safetensors"):
        return os.path.join(os.path.dirname(model_name_or_path), "config.json"), model_name_or_path

    config_path = os.path.join(model_name_or_path, "config.json")
    weights_path = os.path.join(model_name_or_path, "model.safetensors")
    if os.path.isfile(config_path) and os.path.isfile(weights_path):
        return config_path, weights_path

    from huggingface_hub import hf_hub_download

    return (
        hf_hub_download(model_name_or_path, "config.json"),
        hf_hub_download(model_name_or_path, "model.safetensors"),
    )


def load_local_eat(model_name_or_path):
    from safetensors.torch import load_file

    config_path, weights_path = _resolve_model_files(model_name_or_path)

    with open(config_path, "r") as f:
        config = json.load(f)

    model = LocalEAT(config)
    state_dict = load_file(weights_path, device="cpu")
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            stripped[key[len("model.") :]] = value
        else:
            stripped[key] = value

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    ignored_unexpected = {"fc_norm.bias", "fc_norm.weight", "head.bias", "head.weight"}
    unexpected = [key for key in unexpected if key not in ignored_unexpected]
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load local EAT weights cleanly. "
            f"Missing keys: {missing[:10]}, unexpected keys: {unexpected[:10]}"
        )
    return model
