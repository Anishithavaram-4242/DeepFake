"""
EFFResNet-ViT: A Hybrid Convolutional and Vision Transformer Model for Deepfake Detection
Based on: "Deepfake Video Detection: A Comprehensive Deep Learning-Based Approach" (SMART-2025)

Architecture components:
1. EfficientNet-B0 backbone: extracts high-frequency, texture-rich local features
2. ResNet-50 backbone: extracts edge-based structural cues and contour features
3. Locality-preserving Fusion Neck: aligns and fuses dual CNN feature maps
4. Vision Transformer (ViT) Encoder: captures global long-range spatial correlations across facial regions
5. Temporal Sequence Modeling (optional, for multi-frame video sequences via LSTM / Temporal Attention)
6. Classification Head: binary classification (Real vs. Fake)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
)


class ConvLocalityBlock(nn.Module):
    """
    Locality-preserving convolution block to maintain spatial coherence
    as described in the EFFResNet-ViT specification.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        
        # Depthwise convolution for locality refinement
        self.depthwise = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.depthwise(out)))
        return out


class TransformerEncoderBlock(nn.Module):
    """
    Standard Vision Transformer Encoder Layer with Pre-LayerNorm,
    Multi-Head Self-Attention, and MLP Feedforward network.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
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
        # Pre-LN Self Attention with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.mha(normed, normed, normed)
        x = x + self.dropout1(attn_out)
        
        # Pre-LN MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class EFFResNetViT(nn.Module):
    """
    EFFResNet-ViT Hybrid Deepfake Detection Architecture.
    
    Inputs:
        - Image tensor: (B, 3, H, W)
        - Video sequence tensor: (B, T, 3, H, W) if sequence_mode=True
    Output:
        - Logits: (B, 1) raw logits for binary classification (0: Real, 1: Fake)
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_vit_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pretrained: bool = True,
        use_temporal_lstm: bool = False,
        lstm_hidden_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_temporal_lstm = use_temporal_lstm
        
        # 1. EfficientNet-B0 Backbone (extracts high-frequency local texture features)
        eff_weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        efficientnet = models.efficientnet_b0(weights=eff_weights)
        self.eff_features = efficientnet.features
        eff_channels = 1280  # EfficientNet-B0 output channels at 7x7 for 224x224 input
        
        # 2. ResNet-50 Backbone (extracts structural/edge cues)
        res_weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=res_weights)
        self.res_stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        res_channels = 2048  # ResNet-50 layer4 output channels at 7x7 for 224x224 input
        
        # 3. Hybrid Fusion Neck
        total_channels = eff_channels + res_channels  # 1280 + 2048 = 3328
        self.fusion_neck = ConvLocalityBlock(total_channels, embed_dim)
        
        # 4. ViT Tokenization & Transformer Encoder
        # Spatial resolution at 224x224 input is 7x7 = 49 patches
        self.num_patches = 7 * 7
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_vit_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Post-attention locality refinement (Conv2D recovery as described in paper)
        self.post_attn_locality = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # 5. Temporal Sequence Modeling (for multi-frame video inputs)
        self.feat_proj = nn.Linear(embed_dim * 2, embed_dim)
        
        if use_temporal_lstm:
            self.temporal_lstm = nn.LSTM(
                input_size=embed_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            classifier_in_dim = lstm_hidden_dim * 2
        else:
            self.temporal_lstm = None
            classifier_in_dim = embed_dim
            
        # 6. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization for ViT components
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize linear layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract fused CNN + ViT spatial representation for a batch of frames (B, 3, 224, 224).
        Returns a feature vector of shape (B, embed_dim * 2).
        """
        B = x.size(0)
        
        # 1. CNN Feature extraction
        f_eff = self.eff_features(x)       # (B, 1280, 7, 7)
        f_res = self.res_stem(x)           # (B, 2048, 7, 7)
        
        # Ensure identical spatial sizes
        if f_eff.shape[2:] != f_res.shape[2:]:
            f_eff = F.interpolate(f_eff, size=f_res.shape[2:], mode='bilinear', align_corners=False)
            
        # 2. Hybrid concatenation & locality-preserving fusion
        f_cat = torch.cat([f_eff, f_res], dim=1)  # (B, 3328, 7, 7)
        f_fused = self.fusion_neck(f_cat)         # (B, embed_dim, 7, 7)
        
        H, W = f_fused.shape[2], f_fused.shape[3]
        
        # 3. Flatten spatial dimensions to patch tokens: (B, H*W, embed_dim)
        tokens = f_fused.flatten(2).transpose(1, 2)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1 + H*W, embed_dim)
        
        # Add position embeddings
        tokens = self.pos_dropout(tokens + self.pos_embed[:, :tokens.size(1), :])
        
        # 4. ViT Self-Attention Layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)
            
        tokens = self.norm(tokens)
        
        # Split [CLS] token and spatial tokens
        cls_out = tokens[:, 0]  # (B, embed_dim)
        spatial_tokens = tokens[:, 1:]  # (B, H*W, embed_dim)
        
        # 5. Locality recovery post-attention via Conv2D
        spatial_grid = spatial_tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        spatial_recovered = self.post_attn_locality(spatial_grid)
        spatial_pool = F.adaptive_avg_pool2d(spatial_recovered, (1, 1)).flatten(1)  # (B, embed_dim)
        
        # Concatenate global [CLS] representation and spatial pool
        frame_repr = torch.cat([cls_out, spatial_pool], dim=1)  # (B, embed_dim * 2)
        return frame_repr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, 3, H, W) for single frames, or (B, T, 3, H, W) for video sequences.
        Returns:
            logits: (B, 1)
        """
        if x.dim() == 5:
            # Video sequence mode: (B, T, C, H, W)
            B, T, C, H, W = x.size()
            x_flat = x.view(B * T, C, H, W)
            raw_feats = self.extract_frame_features(x_flat)  # (B*T, embed_dim * 2)
            frame_feats = self.feat_proj(raw_feats)          # (B*T, embed_dim)
            frame_feats = frame_feats.view(B, T, -1)         # (B, T, embed_dim)
            
            if self.use_temporal_lstm and self.temporal_lstm is not None:
                lstm_out, _ = self.temporal_lstm(frame_feats)  # (B, T, lstm_hidden*2)
                video_feat = lstm_out[:, -1]                  # (B, lstm_hidden*2)
            else:
                video_feat = frame_feats.mean(dim=1)          # (B, embed_dim)
                
            logits = self.classifier(video_feat)
        elif x.dim() == 4:
            # Single frame mode: (B, C, H, W)
            raw_feats = self.extract_frame_features(x)       # (B, embed_dim * 2)
            frame_feat = self.feat_proj(raw_feats)           # (B, embed_dim)
            
            if self.use_temporal_lstm and self.temporal_lstm is not None:
                seq_in = frame_feat.unsqueeze(1)             # (B, 1, embed_dim)
                lstm_out, _ = self.temporal_lstm(seq_in)     # (B, 1, lstm_hidden*2)
                logits = self.classifier(lstm_out[:, -1])
            else:
                logits = self.classifier(frame_feat)
        else:
            raise ValueError(f"Expected 4D or 5D input tensor, got shape {x.shape}")
            
        return logits
