"""
AST model implementation.

This module adapts the official MIT AST implementation to the framework's
waveform-in/classification-out interface and loads local AudioSet checkpoints.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .base_model import BaseAudioModel
from .beats_model import BatchNorm1d, ArcMarginProduct

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "timm is required for the official AST implementation. "
        "Install it with: pip install timm"
    ) from exc

try:
    from timm.models.layers import to_2tuple, trunc_normal_
except ImportError:
    from timm.layers import to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    """
    Patch embedding layer from the official AST code.

    AST uses overlapping spectrogram patches, so this replacement keeps timm's
    ViT patch embedding from enforcing a fixed image size.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (
            img_size[0] // patch_size[0]
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class OfficialASTBackbone(nn.Module):
    """
    Official AST backbone with local AudioSet checkpoint loading.

    The implementation follows ast/src/models/ast_models.py but removes the
    fixed timm==0.4.5 assertion and any network download path.
    """

    def __init__(
        self,
        label_dim: int = 527,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        input_tdim: int = 1024,
        model_size: str = "base384",
        imagenet_pretrain: bool = False,
        audioset_checkpoint_path: str = None,
        verbose: bool = True,
    ):
        super().__init__()

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        if verbose:
            print("---------------AST Model Summary---------------")
            print(f"AudioSet checkpoint: {audioset_checkpoint_path}")

        if audioset_checkpoint_path is None:
            self._init_from_timm(
                label_dim=label_dim,
                fstride=fstride,
                tstride=tstride,
                input_fdim=input_fdim,
                input_tdim=input_tdim,
                model_size=model_size,
                imagenet_pretrain=imagenet_pretrain,
                verbose=verbose,
            )
        else:
            self._init_from_audioset_checkpoint(
                label_dim=label_dim,
                fstride=fstride,
                tstride=tstride,
                input_fdim=input_fdim,
                input_tdim=input_tdim,
                model_size=model_size,
                checkpoint_path=audioset_checkpoint_path,
                verbose=verbose,
            )

    def _create_timm_vit(self, model_size: str, pretrained: bool) -> nn.Module:
        model_names = {
            "tiny224": "vit_deit_tiny_distilled_patch16_224",
            "small224": "vit_deit_small_distilled_patch16_224",
            "base224": "vit_deit_base_distilled_patch16_224",
            "base384": "vit_deit_base_distilled_patch16_384",
        }
        if model_size not in model_names:
            raise ValueError(
                "model_size must be one of: tiny224, small224, base224, base384"
            )

        try:
            return timm.create_model(model_names[model_size], pretrained=pretrained)
        except Exception:
            # Newer timm releases often expose the same DeiT models without the
            # historical vit_ prefix.
            fallback_name = model_names[model_size].replace("vit_", "", 1)
            return timm.create_model(fallback_name, pretrained=pretrained)

    def _init_from_timm(
        self,
        label_dim: int,
        fstride: int,
        tstride: int,
        input_fdim: int,
        input_tdim: int,
        model_size: str,
        imagenet_pretrain: bool,
        verbose: bool,
    ) -> None:
        self.v = self._create_timm_vit(model_size, pretrained=imagenet_pretrain)
        self.original_num_patches = self.v.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches**0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim),
        )

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose:
            print(f"frequency stride={fstride}, time stride={tstride}")
            print(f"number of patches={num_patches}")

        new_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        if imagenet_pretrain:
            new_proj.weight = nn.Parameter(
                torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1)
            )
            new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        if imagenet_pretrain:
            new_pos_embed = (
                self.v.pos_embed[:, 2:, :]
                .detach()
                .reshape(1, self.original_num_patches, self.original_embedding_dim)
                .transpose(1, 2)
                .reshape(
                    1,
                    self.original_embedding_dim,
                    self.original_hw,
                    self.original_hw,
                )
            )
            if t_dim <= self.original_hw:
                start = int(self.original_hw / 2) - int(t_dim / 2)
                new_pos_embed = new_pos_embed[:, :, :, start : start + t_dim]
            else:
                new_pos_embed = F.interpolate(
                    new_pos_embed,
                    size=(self.original_hw, t_dim),
                    mode="bilinear",
                )
            if f_dim <= self.original_hw:
                start = int(self.original_hw / 2) - int(f_dim / 2)
                new_pos_embed = new_pos_embed[:, :, start : start + f_dim, :]
            else:
                new_pos_embed = F.interpolate(
                    new_pos_embed,
                    size=(f_dim, t_dim),
                    mode="bilinear",
                )
            new_pos_embed = new_pos_embed.reshape(
                1,
                self.original_embedding_dim,
                num_patches,
            ).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
            )
        else:
            self.v.pos_embed = nn.Parameter(
                torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim)
            )
            trunc_normal_(self.v.pos_embed, std=0.02)

    def _init_from_audioset_checkpoint(
        self,
        label_dim: int,
        fstride: int,
        tstride: int,
        input_fdim: int,
        input_tdim: int,
        model_size: str,
        checkpoint_path: str,
        verbose: bool,
    ) -> None:
        if model_size != "base384":
            raise ValueError("AudioSet pretrained AST checkpoints require model_size='base384'.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"AST AudioSet checkpoint not found: {checkpoint_path}")

        audio_model = OfficialASTBackbone(
            label_dim=527,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            model_size="base384",
            imagenet_pretrain=False,
            audioset_checkpoint_path=None,
            verbose=False,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        audio_model = nn.DataParallel(audio_model)
        missing, unexpected = audio_model.load_state_dict(state_dict, strict=False)
        if verbose and (missing or unexpected):
            print(
                "Loaded AST AudioSet checkpoint with "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys"
            )

        self.v = audio_model.module.v
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim),
        )

        if (fstride, tstride) != (10, 10):
            old_proj = self.v.patch_embed.proj
            new_proj = nn.Conv2d(
                1,
                self.original_embedding_dim,
                kernel_size=(16, 16),
                stride=(fstride, tstride),
            )
            new_proj.weight = old_proj.weight
            new_proj.bias = old_proj.bias
            self.v.patch_embed.proj = new_proj

        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches
        if verbose:
            print(f"frequency stride={fstride}, time stride={tstride}")
            print(f"number of patches={num_patches}")

        new_pos_embed = (
            self.v.pos_embed[:, 2:, :]
            .detach()
            .reshape(1, 1212, 768)
            .transpose(1, 2)
            .reshape(1, 768, 12, 101)
        )
        if t_dim < 101:
            start = 50 - int(t_dim / 2)
            new_pos_embed = new_pos_embed[:, :, :, start : start + t_dim]
        elif t_dim > 101:
            new_pos_embed = F.interpolate(new_pos_embed, size=(12, t_dim), mode="bilinear")

        if f_dim < 12:
            start = 6 - int(f_dim / 2)
            new_pos_embed = new_pos_embed[:, :, start : start + f_dim, :]
        elif f_dim > 12:
            new_pos_embed = F.interpolate(new_pos_embed, size=(f_dim, t_dim), mode="bilinear")

        new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(
            torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1)
        )

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(
            1,
            self.original_embedding_dim,
            kernel_size=(16, 16),
            stride=(fstride, tstride),
        )
        test_out = test_proj(test_input)
        return test_out.shape[2], test_out.shape[3]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract pooled AST features from fbank [B, T, 128].
        """
        x = x.unsqueeze(1).transpose(2, 3)

        batch_size = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(batch_size, -1, -1)
        dist_token = self.v.dist_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for block in self.v.blocks:
            x = block(x)
        x = self.v.norm(x)
        return (x[:, 0] + x[:, 1]) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_head(self.extract_features(x))


class ASTAudioModel(BaseAudioModel):
    """
    AST model wrapper for the training framework.

    It accepts waveform tensors [B, samples], converts them to official AST
    normalized Kaldi fbank features [B, target_length, 128], and adds the same
    embedding/classification head used by the other framework models.
    """

    def __init__(
        self,
        checkpoint_path: str = None,
        num_classes: int = None,
        model_name: str = None,
        feature_dim: int = 768,
        dropout: float = 0.1,
        sample_rate: int = 16000,
        target_length: int = 1024,
        norm_mean: float = -4.2677393,
        norm_std: float = 4.5689974,
        fstride: int = 10,
        tstride: int = 10,
        input_fdim: int = 128,
        model_size: str = "base384",
        imagenet_pretrain: bool = False,
        pooling: str = "cls_distill",
    ):
        super().__init__()

        checkpoint_path = checkpoint_path or model_name
        if checkpoint_path is None:
            raise ValueError(
                "AST requires model.checkpoint_path pointing to a local official .pth file."
            )
        if num_classes is None:
            raise ValueError("ASTAudioModel requires num_classes.")

        self.ast = OfficialASTBackbone(
            label_dim=527,
            fstride=fstride,
            tstride=tstride,
            input_fdim=input_fdim,
            input_tdim=target_length,
            model_size=model_size,
            imagenet_pretrain=imagenet_pretrain,
            audioset_checkpoint_path=checkpoint_path,
        )
        if self.ast.original_embedding_dim != feature_dim:
            raise ValueError(
                f"AST feature_dim mismatch: config has {feature_dim}, "
                f"backbone has {self.ast.original_embedding_dim}."
            )
        self.ast.mlp_head = nn.Identity()

        self.sample_rate = sample_rate
        self.target_length = target_length
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.pooling = pooling.lower()
        if self.pooling != "cls_distill":
            raise ValueError(
                f"Unsupported AST pooling: {pooling}. Expected 'cls_distill'."
            )

        self.bn = BatchNorm1d(input_size=feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.arc_margin = ArcMarginProduct(
            in_features=feature_dim,
            out_features=num_classes,
            s=30.0,
            m=0.5,
        )

        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def _waveform_to_fbank(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform batch [B, T] to official AST fbank [B, target_length, 128].
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        features = []
        for waveform in audio:
            waveform = waveform.float()
            waveform = waveform - waveform.mean()

            fbank = torchaudio.compliance.kaldi.fbank(
                waveform.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.sample_rate,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )

            n_frames = fbank.shape[0]
            if n_frames < self.target_length:
                pad = self.target_length - n_frames
                fbank = F.pad(fbank, (0, 0, 0, pad))
            else:
                fbank = fbank[: self.target_length, :]

            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            features.append(fbank)

        return torch.stack(features, dim=0)

    def _extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        fbank = self._waveform_to_fbank(audio)
        return self.ast.extract_features(fbank)

    def forward(
        self,
        audio: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        feature = self._extract_features(audio)
        feature = self.bn(feature)
        feature = self.dropout(feature)
        emb = self.fc(feature)
        return self.arc_margin(emb, labels)

    def extract_embedding(
        self,
        audio: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            feature = self._extract_features(audio)
            feature = self.bn(feature)
            emb = self.fc(feature)
        return emb

    def get_feature_dim(self) -> int:
        return self.feature_dim
