"""
EFFResNet-Unified — Video Deepfake Inference Script
====================================================
Usage:
    python predict.py --video path/to/video.mp4 --checkpoint path/to/best_effresnet_unified.pth

Output:
    - Console: REAL / FAKE verdict with confidence score
    - output/<video_name>_result.txt       : Text summary
    - output/<video_name>_frames/          : Annotated sample frames (optional, --save_frames)
    - output/<video_name>_report.json      : Full per-frame + video-level JSON report

Example:
    python predict.py --video myvideo.mp4 --checkpoint kernel_outputs_unified/output_unified/best_effresnet_unified.pth
    python predict.py --video myvideo.mp4 --checkpoint best_effresnet_unified.pth --save_frames
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (must match the trained checkpoint exactly)
# ─────────────────────────────────────────────────────────────────────────────

def get_srm_filters() -> torch.Tensor:
    f1 = np.array([
        [0,  0,  0,  0,  0],
        [0, -1,  2, -1,  0],
        [0,  2, -4,  2,  0],
        [0, -1,  2, -1,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0
    f2 = np.array([
        [-1,  2, -2,  2, -1],
        [ 2, -6,  8, -6,  2],
        [-2,  8,-12,  8, -2],
        [ 2, -6,  8, -6,  2],
        [-1,  2, -2,  2, -1]
    ], dtype=np.float32) / 12.0
    f3 = np.array([
        [0,  0,  0,  0,  0],
        [0,  1, -2,  1,  0],
        [0, -2,  4, -2,  0],
        [0,  1, -2,  1,  0],
        [0,  0,  0,  0,  0]
    ], dtype=np.float32) / 4.0
    filters = np.stack([f1, f2, f3], axis=0)
    return torch.tensor(np.expand_dims(filters, axis=1), dtype=torch.float32)


class SRMSteganalysisStream(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        srm_k = get_srm_filters()
        self.srm_conv = nn.Conv2d(3, 3, kernel_size=5, padding=2, groups=3, bias=False)
        self.srm_conv.weight.data = srm_k.repeat(1, 1, 1, 1)
        self.srm_conv.weight.requires_grad = False
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, out_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_dim), nn.GELU()
        )
    def forward(self, x): return self.encoder(self.srm_conv(x))


class SpectralFrequencyStream(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, out_dim, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_dim), nn.GELU()
        )
    def forward(self, x):
        fft2d = torch.fft.fft2(x, norm='ortho')
        mag = torch.log(torch.abs(fft2d) + 1e-6)
        mag_shift = torch.fft.fftshift(mag, dim=(-2, -1))
        return self.encoder(mag_shift)


class ConvLocalityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                                   groups=out_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()
    def forward(self, x):
        return self.act2(self.bn2(self.depthwise(self.act1(self.bn1(self.conv1(x))))))


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.mha     = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                             dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    def forward(self, q, kv):
        out, _ = self.mha(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv))
        return self.dropout(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1    = nn.LayerNorm(embed_dim)
        self.mha      = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                              dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2    = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        a, _ = self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout1(a)
        x = x + self.mlp(self.norm2(x))
        return x


class EFFResNetUnified(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_vit_layers=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        eff = models.efficientnet_b0(weights=None)
        self.eff_features = eff.features
        res = models.resnet50(weights=None)
        self.res_stem = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, res.layer2, res.layer3, res.layer4
        )
        self.fusion_neck      = ConvLocalityBlock(1280 + 2048, embed_dim)
        self.freq_stream      = SpectralFrequencyStream(embed_dim)
        self.srm_stream       = SRMSteganalysisStream(embed_dim)
        self.cross_attn_freq  = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.cross_attn_srm   = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.alpha_freq       = nn.Parameter(torch.zeros(1))
        self.alpha_srm        = nn.Parameter(torch.zeros(1))
        self.num_patches      = 7 * 7
        self.cls_token        = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed        = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_dropout      = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_vit_layers)
        ])
        self.norm                = nn.LayerNorm(embed_dim)
        self.post_attn_locality  = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim), nn.GELU()
        )
        self.feat_proj  = nn.Linear(embed_dim * 2, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        B = x.size(0)
        f_eff = self.eff_features(x)
        f_res = self.res_stem(x)
        if f_eff.shape[2:] != f_res.shape[2:]:
            f_eff = F.interpolate(f_eff, size=f_res.shape[2:], mode='bilinear', align_corners=False)
        f_spatial = self.fusion_neck(torch.cat([f_eff, f_res], dim=1))
        f_freq = self.freq_stream(x)
        f_srm  = self.srm_stream(x)
        H, W = f_spatial.shape[2], f_spatial.shape[3]
        t_spatial = f_spatial.flatten(2).transpose(1, 2)
        t_freq    = f_freq.flatten(2).transpose(1, 2)
        t_srm     = f_srm.flatten(2).transpose(1, 2)
        freq_ctx  = self.cross_attn_freq(t_spatial, t_freq)
        srm_ctx   = self.cross_attn_srm(t_spatial, t_srm)
        tokens = (t_spatial
                  + torch.tanh(self.alpha_freq) * freq_ctx
                  + torch.tanh(self.alpha_srm)  * srm_ctx)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_dropout(tokens + self.pos_embed[:, :tokens.size(1), :])
        for layer in self.transformer_layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]
        spatial_tokens = tokens[:, 1:]
        spatial_grid   = spatial_tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        spatial_pool   = F.adaptive_avg_pool2d(self.post_attn_locality(spatial_grid), (1, 1)).flatten(1)
        feat = self.feat_proj(torch.cat([cls_out, spatial_pool], dim=1))
        return self.classifier(feat)


# ─────────────────────────────────────────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = 32) -> list:
    """
    Evenly sample `num_frames` frames from the video.
    Returns list of PIL Images in RGB.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0

    if total_frames == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    # Evenly spaced frame indices
    sample_indices = set(
        int(i * total_frames / num_frames) for i in range(num_frames)
    )

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()

    print(f"  Video info   : {total_frames} total frames | {fps:.1f} FPS | {duration_sec:.1f}s")
    print(f"  Sampled      : {len(frames)} frames for inference")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def run_inference(model, frames: list, device, batch_size: int = 8) -> list:
    """
    Run model on list of PIL frames in batches.
    Returns list of fake probability scores (float, 0.0–1.0).
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch_pil = frames[i : i + batch_size]
            batch_tensor = torch.stack([TRANSFORM(f) for f in batch_pil]).to(device)
            logits = model(batch_tensor)
            probs  = torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist()
            all_probs.extend(probs)

    return all_probs


# ─────────────────────────────────────────────────────────────────────────────
# VOTING & DECISION
# ─────────────────────────────────────────────────────────────────────────────

def video_decision(probs: list, threshold: float = 0.5) -> dict:
    """
    Aggregate per-frame probabilities using three voting strategies.
    """
    arr = np.array(probs)

    mean_prob   = float(np.mean(arr))
    median_prob = float(np.median(arr))
    maj_vote    = float(np.mean(arr >= threshold) >= 0.5)   # majority hard vote

    # Use mean voting as the primary decision
    is_fake_mean   = mean_prob   >= threshold
    is_fake_median = median_prob >= threshold
    is_fake_maj    = maj_vote    >= threshold

    # Confidence = how far from the 0.5 threshold
    confidence = abs(mean_prob - 0.5) * 2.0 * 100  # 0–100%

    return {
        "mean_prob":        round(mean_prob, 4),
        "median_prob":      round(median_prob, 4),
        "majority_vote":    round(maj_vote, 4),
        "verdict_mean":     "FAKE" if is_fake_mean   else "REAL",
        "verdict_median":   "FAKE" if is_fake_median else "REAL",
        "verdict_majority": "FAKE" if is_fake_maj    else "REAL",
        "confidence_pct":   round(confidence, 1),
        "threshold":        threshold,
        "num_frames":       len(probs),
        "frames_flagged_fake": int(np.sum(arr >= threshold)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FRAME ANNOTATION (optional --save_frames)
# ─────────────────────────────────────────────────────────────────────────────

def save_annotated_frames(frames: list, probs: list, out_dir: Path, threshold: float = 0.5):
    """
    Save sample frames with REAL/FAKE label and confidence overlaid on each.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save up to 12 frames evenly spaced
    n = min(12, len(frames))
    indices = [int(i * len(frames) / n) for i in range(n)]

    for rank, idx in enumerate(indices):
        frame_np = np.array(frames[idx])
        prob     = probs[idx]
        label    = "FAKE" if prob >= threshold else "REAL"
        color    = (220, 50, 50) if label == "FAKE" else (50, 200, 50)  # RGB

        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (480, 480))

        # Draw banner
        cv2.rectangle(frame_bgr, (0, 0), (480, 60), (0, 0, 0), -1)
        cv2.putText(frame_bgr, f"{label}  {prob*100:.1f}%",
                    (12, 42), cv2.FONT_HERSHEY_DUPLEX, 1.3,
                    (color[2], color[1], color[0]), 2)  # BGR
        cv2.putText(frame_bgr, f"Frame {idx}",
                    (340, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        out_path = out_dir / f"frame_{rank+1:02d}_idx{idx}_{label}.jpg"
        cv2.imwrite(str(out_path), frame_bgr)

    print(f"  Annotated frames saved → {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────────────────────────────────────

def print_report(video_name: str, decision: dict, elapsed: float):
    verdict = decision["verdict_mean"]
    prob    = decision["mean_prob"]
    conf    = decision["confidence_pct"]
    flagged = decision["frames_flagged_fake"]
    total   = decision["num_frames"]

    color_on  = "\033[91m" if verdict == "FAKE" else "\033[92m"
    color_off = "\033[0m"
    bold      = "\033[1m"

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  EFFResNet-Unified — Deepfake Detection Result           ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Video   : {video_name:<46} ║")
    print(f"║  Verdict : {bold}{color_on}{verdict:<46}{color_off}  ║")
    print(f"║  Fake Probability (mean) : {prob*100:>6.2f}%                      ║")
    print(f"║  Confidence              : {conf:>6.1f}%                      ║")
    print(f"║  Frames flagged FAKE     : {flagged:>4} / {total:<4} frames          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Voting breakdown:                                       ║")
    print(f"║    Mean   voting  → {decision['verdict_mean']:<6} (p={decision['mean_prob']:.3f})                 ║")
    print(f"║    Median voting  → {decision['verdict_median']:<6} (p={decision['median_prob']:.3f})                 ║")
    print(f"║    Majority vote  → {decision['verdict_majority']:<6}                              ║")
    print(f"║  Inference time  : {elapsed:.1f}s                                   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EFFResNet-Unified — Deepfake Video Detector"
    )
    parser.add_argument("--video",      required=True,  help="Path to input video file (mp4, avi, mov, mkv)")
    parser.add_argument("--checkpoint", required=True,  help="Path to best_effresnet_unified.pth")
    parser.add_argument("--num_frames", type=int, default=32, help="Number of frames to sample (default: 32)")
    parser.add_argument("--threshold",  type=float, default=0.5, help="Decision threshold for FAKE (default: 0.5)")
    parser.add_argument("--save_frames", action="store_true", help="Save annotated sample frames to output/")
    parser.add_argument("--output_dir", default="output", help="Output directory for results (default: output/)")
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────
    video_path = Path(args.video)
    ckpt_path  = Path(args.checkpoint)

    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    video_name = video_path.stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup device ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  EFFResNet-Unified Deepfake Video Detector")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Video      : {video_path}")
    print(f"  Checkpoint : {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  Sampling   : {args.num_frames} frames")
    print(f"  Threshold  : {args.threshold}")
    print(f"{'='*60}")

    # ── Load model ────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    model = EFFResNetUnified(embed_dim=256, num_heads=8, num_vit_layers=4).to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    alpha_f = torch.tanh(model.alpha_freq).item()
    alpha_s = torch.tanh(model.alpha_srm).item()
    print(f"  Model loaded ✓  |  α_freq={alpha_f:.4f}  α_srm={alpha_s:.4f}")

    # ── Extract frames ────────────────────────────────────────────────
    print("\n[2/4] Extracting frames...")
    t0 = time.time()
    try:
        frames = extract_frames(str(video_path), num_frames=args.num_frames)
    except Exception as e:
        print(f"[ERROR] Frame extraction failed: {e}")
        sys.exit(1)

    # ── Run inference ─────────────────────────────────────────────────
    print("\n[3/4] Running inference...")
    probs   = run_inference(model, frames, device)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(frames)/elapsed:.1f} frames/sec)")

    # ── Compute decision ──────────────────────────────────────────────
    print("\n[4/4] Computing verdict...")
    decision = video_decision(probs, threshold=args.threshold)

    # ── Print report ──────────────────────────────────────────────────
    print_report(video_name, decision, elapsed)

    # ── Save annotated frames (optional) ─────────────────────────────
    if args.save_frames:
        frames_dir = output_dir / f"{video_name}_frames"
        save_annotated_frames(frames, probs, frames_dir, threshold=args.threshold)

    # ── Save JSON report ──────────────────────────────────────────────
    report = {
        "video":          str(video_path),
        "checkpoint":     str(ckpt_path),
        "num_frames_sampled": len(frames),
        "per_frame_probs":    [round(p, 4) for p in probs],
        "decision":           decision,
        "inference_time_sec": round(elapsed, 2),
        "device":             str(device),
    }
    json_path = output_dir / f"{video_name}_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Full report saved → {json_path}")

    # ── Save text summary ─────────────────────────────────────────────
    txt_path = output_dir / f"{video_name}_result.txt"
    with open(txt_path, "w") as f:
        f.write(f"Video    : {video_path}\n")
        f.write(f"Verdict  : {decision['verdict_mean']}\n")
        f.write(f"Fake Prob: {decision['mean_prob']*100:.2f}%\n")
        f.write(f"Confidence: {decision['confidence_pct']:.1f}%\n")
        f.write(f"Frames flagged FAKE: {decision['frames_flagged_fake']} / {decision['num_frames']}\n")
    print(f"  Text summary saved → {txt_path}")


if __name__ == "__main__":
    main()
