"""
EFFResNet-DSSFCA: Dual-Stream Spatial-Frequency Cross-Attention Model for Deepfake Detection.
Novelty 1: Augmenting EFFResNet-ViT with a parallel 2D-FFT Spectral Anomaly branch
coupled via bidirectional/spatial-frequency cross-attention.

Motivation:
Generative models (GANs, Autoencoders, Diffusion) rely on transpose convolutions and upsamplers
that leave periodic mathematical grid artifacts. In RGB pixel space, these artifacts are subtle
or masked by post-processing, but in the frequency domain (2D FFT Magnitude Spectrum), they
manifest as anomalous high-frequency energy distributions and spectral peaks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights


def compute_fft_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Computes normalized log-magnitude 2D Fast Fourier Transform spectrum.
    Args:
        x: (B, 3, H, W) normalized RGB tensor
    Returns:
        mag_norm: (B, 3, H, W) normalized log-magnitude spectral map
    """
    # 2D FFT with orthogonal normalization
    fft = torch.fft.fft2(x, norm="ortho")
    # Shift zero-frequency components to center
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    # Log magnitude spectrum to compress dynamic range
    mag = torch.log(torch.abs(fft_shifted) + 1e-6)
    
    # Instance-level normalization across spatial dimensions
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    std = mag.std(dim=(-2, -1), keepdim=True) + 1e-6
    return (mag - mean) / std


class SpectralResidualConv(nn.Module):
    """
    Lightweight Spectral CNN encoder extracting multi-scale frequency features
    from 2D FFT magnitude maps.
    """
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # -> 112x112
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                    # -> 56x56
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # -> 28x28
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # -> 14x14
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1, bias=False), # -> 7x7
            nn.BatchNorm2d(out_dim),
            nn.GELU()
        )

    def forward(self, mag: torch.Tensor) -> torch.Tensor:
        x = self.stem(mag)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x  # (B, out_dim, 7, 7)


class SpatialFrequencyCrossAttention(nn.Module):
    """
    Cross-Attention fusion module allowing Spatial tokens to query
    Frequency-domain spectral representations.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_spatial = nn.LayerNorm(embed_dim)
        self.norm_freq = nn.LayerNorm(embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        # Learnable gating factor initialized small to facilitate stable residual learning
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, spatial_tokens: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        """
        spatial_tokens: (B, N, D)
        freq_tokens: (B, N, D)
        """
        q = self.norm_spatial(spatial_tokens)
        k = self.norm_freq(freq_tokens)
        v = k
        
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        fused = spatial_tokens + self.gamma * self.dropout(attn_out)
        return fused


class ConvLocalityBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act2(self.bn2(self.depthwise(self.act1(self.bn1(self.conv1(x))))))


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
        x = x + self.dropout1(self.mha(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.mlp(self.norm2(x))
        return x


class EFFResNetDSSFCA(nn.Module):
    """
    EFFResNet-DSSFCA Architecture:
    Dual-Stream Spatial-Frequency Cross-Attention Network.
    
    Combines:
    1. Spatial Branch: EfficientNet-B0 + ResNet-50 + Locality Fusion
    2. Frequency Branch: 2D-FFT Log-Magnitude + Spectral CNN
    3. Spatial-Frequency Cross-Attention (DS-SFCA)
    4. Vision Transformer (ViT) Self-Attention Encoder
    5. Locality Recovery Block + Binary Classifier
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
        
        # --- STREAM 1: SPATIAL CNN BACKBONES ---
        eff_weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eff = models.efficientnet_b0(weights=eff_weights)
        self.eff_features = eff.features
        
        res_weights = ResNet50_Weights.DEFAULT if pretrained else None
        res = models.resnet50(weights=res_weights)
        self.res_stem = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, res.layer2, res.layer3, res.layer4
        )
        
        self.fusion_neck = ConvLocalityBlock(1280 + 2048, embed_dim)
        
        # --- STREAM 2: FREQUENCY SPECTRAL CNN BACKBONE ---
        self.spectral_encoder = SpectralResidualConv(out_dim=embed_dim)
        
        # --- INTER-STREAM CROSS-ATTENTION ---
        self.cross_attn = SpatialFrequencyCrossAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # --- TRANSFORMER ENCODER ---
        self.num_patches = 7 * 7
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_vit_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # --- LOCALITY RECOVERY & HEAD ---
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
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        
        # 1. Spatial Stream
        f_eff = self.eff_features(x)
        f_res = self.res_stem(x)
        if f_eff.shape[2:] != f_res.shape[2:]:
            f_eff = F.interpolate(f_eff, size=f_res.shape[2:], mode='bilinear', align_corners=False)
        f_spatial = self.fusion_neck(torch.cat([f_eff, f_res], dim=1))  # (B, embed_dim, 7, 7)
        
        # 2. Frequency Stream (2D FFT Spectrum -> Spectral CNN)
        mag_spectrum = compute_fft_magnitude(x)
        f_freq = self.spectral_encoder(mag_spectrum)                    # (B, embed_dim, 7, 7)
        
        H, W = f_spatial.shape[2], f_spatial.shape[3]
        
        # Flatten into tokens
        s_tokens = f_spatial.flatten(2).transpose(1, 2)                 # (B, 49, embed_dim)
        f_tokens = f_freq.flatten(2).transpose(1, 2)                    # (B, 49, embed_dim)
        
        # 3. Spatial-Frequency Cross-Attention
        enriched_tokens = self.cross_attn(s_tokens, f_tokens)           # (B, 49, embed_dim)
        
        # 4. ViT Self-Attention
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, enriched_tokens], dim=1)        # (B, 50, embed_dim)
        tokens = self.pos_dropout(tokens + self.pos_embed[:, :tokens.size(1), :])
        
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        
        cls_out = tokens[:, 0]
        spatial_tokens = tokens[:, 1:]
        spatial_grid = spatial_tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        spatial_recovered = self.post_attn_locality(spatial_grid)
        spatial_pool = F.adaptive_avg_pool2d(spatial_recovered, (1, 1)).flatten(1)
        
        fused_vec = torch.cat([cls_out, spatial_pool], dim=1)
        return self.feat_proj(fused_vec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)
