"""
Dataset module for FaceForensics++ extracted face frames on Kaggle.
Handles balanced train/val splitting by original video ID to avoid data leakage.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FaceForensicsFramesDataset(Dataset):
    """
    Dataset that loads pre-extracted face frames from FaceForensics++ directory.
    
    Structure supported:
        root/real/<video_id>/<frame_id>.png
        root/fake/<method>/<video_id_pair>/<frame_id>.png
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
        sequence_len: int = 1,
    ):
        self.samples = samples
        self.transform = transform
        self.sequence_len = sequence_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Fallback to black image in case of corrupted frame
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = transforms.ToTensor()(image)
            
        return tensor, torch.tensor(label, dtype=torch.float32)


def discover_dataset_root(candidate_roots: Optional[List[str]] = None) -> Path:
    """Find the valid FaceForensics++ dataset root path."""
    default_candidates = [
        "/kaggle/input/faceforencispp-extracted-frames",
        "/kaggle/input/datasets/adham7elmy/faceforencispp-extracted-frames",
        "/kaggle/input/faceforensics-extracted-frames",
        "./sample_dataset",
    ]
    if candidate_roots:
        default_candidates = candidate_roots + default_candidates
        
    for c in default_candidates:
        p = Path(c)
        if p.exists() and (p / "real").exists():
            return p
            
    # Check anywhere under /kaggle/input
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sub in kaggle_input.rglob("real"):
            if sub.is_dir() and (sub.parent / "fake").exists():
                return sub.parent
                
    raise FileNotFoundError("Could not locate FaceForensics++ dataset directory with 'real' and 'fake' folders.")


def create_manifest(
    dataset_root: Path,
    max_frames_per_video: int = 10,
    val_split: float = 0.2,
    seed: int = 42,
    balance_classes: bool = True
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Build balanced train and val splits grouped by video ID to prevent identity leakage.
    """
    random.seed(seed)
    real_dir = dataset_root / "real"
    fake_dir = dataset_root / "fake"
    
    # 1. Collect real frames by video ID
    real_videos: Dict[str, List[str]] = {}
    if real_dir.exists():
        for vid_folder in sorted(real_dir.iterdir()):
            if vid_folder.is_dir():
                frames = sorted([
                    str(p) for p in vid_folder.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ])
                if frames:
                    real_videos[vid_folder.name] = frames
                    
    # 2. Collect fake frames by video pair ID
    fake_videos: Dict[str, List[str]] = {}
    if fake_dir.exists():
        for method in sorted(fake_dir.iterdir()):
            if method.is_dir():
                for vid_folder in sorted(method.iterdir()):
                    if vid_folder.is_dir():
                        frames = sorted([
                            str(p) for p in vid_folder.iterdir()
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                        ])
                        if frames:
                            key = f"{method.name}_{vid_folder.name}"
                            fake_videos[key] = frames
                            
    # 3. Split real videos into train and val
    real_keys = list(real_videos.keys())
    random.shuffle(real_keys)
    n_real_val = max(1, int(len(real_keys) * val_split))
    val_real_keys = set(real_keys[:n_real_val])
    train_real_keys = set(real_keys[n_real_val:])
    
    # 4. Split fake videos into train and val
    fake_keys = list(fake_videos.keys())
    random.shuffle(fake_keys)
    n_fake_val = max(1, int(len(fake_keys) * val_split))
    val_fake_keys = set(fake_keys[:n_fake_val])
    train_fake_keys = set(fake_keys[n_fake_val:])
    
    def sample_frames(keys: set, source_dict: dict, label: int) -> List[Tuple[str, int]]:
        samples = []
        for k in keys:
            all_frames = source_dict[k]
            # Sample up to max_frames_per_video frames uniformly
            if len(all_frames) <= max_frames_per_video:
                selected = all_frames
            else:
                step = len(all_frames) / max_frames_per_video
                selected = [all_frames[int(i * step)] for i in range(max_frames_per_video)]
            for path in selected:
                samples.append((path, label))
        return samples

    train_real = sample_frames(train_real_keys, real_videos, label=0)
    val_real = sample_frames(val_real_keys, real_videos, label=0)
    train_fake = sample_frames(train_fake_keys, fake_videos, label=1)
    val_fake = sample_frames(val_fake_keys, fake_videos, label=1)
    
    if balance_classes:
        # Balance train set
        n_train_common = min(len(train_real), len(train_fake))
        train_samples = random.sample(train_real, n_train_common) + random.sample(train_fake, n_train_common)
        
        # Balance val set
        n_val_common = min(len(val_real), len(val_fake))
        val_samples = random.sample(val_real, n_val_common) + random.sample(val_fake, n_val_common)
    else:
        train_samples = train_real + train_fake
        val_samples = val_real + val_fake
        
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    
    return train_samples, val_samples


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation image augmentations."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform
