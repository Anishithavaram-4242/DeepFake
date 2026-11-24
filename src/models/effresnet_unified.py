"""
EFFResNet-Unified Architecture
==============================
Unified Deepfake Detection Network integrating:
1. Dual-Stream Pretrained Spatial Backbones (EfficientNet-B0 + ResNet-50)
2. Frequency-Domain 2D-FFT Magnitude Spectrum Branch
3. Steganalysis 5x5 SRM (Spatial Rich Model) Noise Residual Stream
4. Zero-Initialized Gated Cross-Attention Fusion
5. Locality-Restored Vision Transformer (ViT) Encoder
6. Dual Frame-Level & Temporal Video-Level Decision Fusion
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights


# ── 1. SRM HIGH-PASS FILTER BANK ─────────────────────────────────
def get_srm_filters() -> torch.Tensor:
    """Fixed 5x5 SRM (Spatial Rich Model) steganalysis high-pass kernels."""
    # Filter 1: 1st-order edge detection
    f1 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  2, -4,  2,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0

    # Filter 2: 2nd-order Laplacian-like edge detection
    f2 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0

    # Filter 3: 3x3 square edge residual embedded in 5x5
    f3 = np.array([
        [0,  0,  0,  0,  0],
        [0,  1, -2,  1,  0],
        [0, -2,  4, -2,  0],
        [0,  1, -2,  1,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0

    filters = np.stack([f1, f2, f3], axis=0)  # (3, 5, 5)
    filters = np.expand_dims(filters, axis=1)  # (3, 1, 5, 5)
    return torch.tensor(filters, dtype=torch.float32)


class SRMSteganalysisStream(nn.Module):
    """Extracts high-pass noise residuals and encodes them into 7x7 spatial tokens."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        srm_k = get_srm_filters()  # (3, 1, 5, 5)
        # Apply depthwise across RGB channels (3 filters x 3 channels = 9 channels, then 1x1 conv to 32)
        self.srm_conv = nn.Conv2d(3, 3, kernel_size=5, padding=2, groups=3, bias=False)
        self.srm_conv.weight.data = srm_k.repeat(1, 1, 1, 1)
        self.srm_conv.weight.requires_grad = False  # Keep SRM filters fixed

        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 14x14 -> 7x7
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = self.srm_conv(x)
        return self.encoder(residuals)  # (B, out_dim, 7, 7)


# ── 2. 2D-FFT FREQUENCY MAGNITUDE STREAM ─────────────────────────
class SpectralFrequencyStream(nn.Module):
    """Computes 2D FFT log-magnitude spectrum and encodes lattice artifacts into 7x7 tokens."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 14x14 -> 7x7
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2D Fast Fourier Transform on image tensor
        fft2d = torch.fft.fft2(x, norm="ortho")
        mag = torch.log(torch.abs(fft2d) + 1e-6)
        mag_shift = torch.fft.fftshift(mag, dim=(-2, -1))
        return self.encoder(mag_shift)  # (B, out_dim, 7, 7)


# ── 3. LOCALITY & CROSS-ATTENTION COMPONENTS ───────────────────────
class ConvLocalityBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.bn2(self.depthwise(self.act1(self.bn1(self.conv1(x))))))


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_val: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_val)
        attn_out, _ = self.mha(q, kv, kv)
        return self.dropout(attn_out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.mha(normed, normed, normed)
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


# ── 4. EFFRESNET-UNIFIED COMPREHENSIVE ARCHITECTURE ───────────────
class EFFResNetUnified(nn.Module):
    """
    Unified Architecture fusing Spatial CNNs, Spectral FFT, and SRM Steganalysis
    with Zero-Gated Cross-Attention and a Locality-Restored Vision Transformer.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_vit_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Spatial Dual-CNN Backbones
        eff_weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eff = models.efficientnet_b0(weights=eff_weights)
        self.eff_features = eff.features  # (B, 1280, 7, 7)

        res_weights = ResNet50_Weights.DEFAULT if pretrained else None
        res = models.resnet50(weights=res_weights)
        self.res_stem = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, res.layer2, res.layer3, res.layer4
        )  # (B, 2048, 7, 7)

        # Spatial fusion neck
        self.fusion_neck = ConvLocalityBlock(1280 + 2048, embed_dim)

        # 2. Auxiliary Forensic Streams
        self.freq_stream = SpectralFrequencyStream(embed_dim)
        self.srm_stream = SRMSteganalysisStream(embed_dim)

        # 3. Cross-Attention Adapters
        self.cross_attn_freq = CrossAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout)
        self.cross_attn_srm = CrossAttentionBlock(embed_dim, num_heads=num_heads, dropout=dropout)

        # 4. Zero-Initialized Learnable Gate Parameters
        # tanh(0) = 0 ensures model begins at 100% baseline capability with zero initial degradation
        self.alpha_freq = nn.Parameter(torch.zeros(1))
        self.alpha_srm = nn.Parameter(torch.zeros(1))

        # 5. Vision Transformer & Locality Restoration
        self.num_patches = 7 * 7
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_vit_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.post_attn_locality = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        self.feat_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # 1. Spatial Stream
        f_eff = self.eff_features(x)
        f_res = self.res_stem(x)
        if f_eff.shape[2:] != f_res.shape[2:]:
            f_eff = F.interpolate(f_eff, size=f_res.shape[2:], mode="bilinear", align_corners=False)
        f_spatial = self.fusion_neck(torch.cat([f_eff, f_res], dim=1))  # (B, 256, 7, 7)

        # 2. Forensic Streams (Frequency & Steganalysis)
        f_freq = self.freq_stream(x)  # (B, 256, 7, 7)
        f_srm = self.srm_stream(x)    # (B, 256, 7, 7)

        H, W = f_spatial.shape[2], f_spatial.shape[3]

        # 3. Token Preparation
        tokens_spatial = f_spatial.flatten(2).transpose(1, 2)  # (B, 49, 256)
        tokens_freq = f_freq.flatten(2).transpose(1, 2)        # (B, 49, 256)
        tokens_srm = f_srm.flatten(2).transpose(1, 2)          # (B, 49, 256)

        # 4. Zero-Gated Cross-Attention Fusion
        freq_context = self.cross_attn_freq(tokens_spatial, tokens_freq)
        srm_context = self.cross_attn_srm(tokens_spatial, tokens_srm)

        tokens_fused = (
            tokens_spatial
            + torch.tanh(self.alpha_freq) * freq_context
            + torch.tanh(self.alpha_srm) * srm_context
        )

        # 5. Transformer Processing
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens_fused], dim=1)
        tokens = self.pos_dropout(tokens + self.pos_embed[:, :tokens.size(1), :])

        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)

        # 6. Global & Local Representation Fusion
        cls_out = tokens[:, 0]  # (B, 256)
        spatial_tokens = tokens[:, 1:]
        spatial_grid = spatial_tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        spatial_recovered = self.post_attn_locality(spatial_grid)
        spatial_pool = F.adaptive_avg_pool2d(spatial_recovered, (1, 1)).flatten(1)

        # 7. Classification
        feat = self.feat_proj(torch.cat([cls_out, spatial_pool], dim=1))
        return self.classifier(feat)
