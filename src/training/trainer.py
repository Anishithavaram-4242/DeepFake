"""
Training and Validation engine for EFFResNet-ViT Deepfake Detection.
Includes Mixed-Precision (AMP), Cosine Annealing, and comprehensive metric tracking.
"""

import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.metrics import compute_metrics


class DeepfakeTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
        use_amp: bool = True,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.scheduler = scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.best_auc = 0.0
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": [], "val_auc": []
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)
        start_time = time.time()
        
        for step, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if (step + 1) % max(1, n_batches // 5) == 0 or (step + 1) == n_batches:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch}] Step [{step+1}/{n_batches}] - Loss: {loss.item():.4f} ({elapsed:.1f}s)")
                
        if self.scheduler is not None:
            self.scheduler.step()
            
        avg_loss = total_loss / max(1, n_batches)
        return avg_loss

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_probs = []
        
        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels_tensor = labels.to(self.device, non_blocking=True).unsqueeze(1)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels_tensor)
                probs = torch.sigmoid(logits)
                
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.squeeze(1).cpu().numpy().tolist())
            
        avg_loss = total_loss / max(1, len(self.val_loader))
        metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
        return avg_loss, metrics

    def fit(self, num_epochs: int = 5) -> Dict[str, List[float]]:
        print("=" * 65)
        print(f"Starting Training on device: {self.device} (AMP: {self.use_amp})")
        print(f"Train samples: {len(self.train_loader.dataset)} | Val samples: {len(self.val_loader.dataset)}")
        print("=" * 65)
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, metrics = self.evaluate()
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(metrics["accuracy"])
            self.history["val_prec"].append(metrics["precision"])
            self.history["val_rec"].append(metrics["recall"])
            self.history["val_f1"].append(metrics["f1_score"])
            self.history["val_auc"].append(metrics["auc_roc"])
            
            print(f"\n--- Epoch {epoch}/{num_epochs} Summary ---")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Acc: {metrics['accuracy']}% | Prec: {metrics['precision']}% | Rec: {metrics['recall']}%")
            print(f"Val F1: {metrics['f1_score']}% | Val AUC-ROC: {metrics['auc_roc']}%\n")
            
            # Save best checkpoint based on AUC-ROC
            if metrics["auc_roc"] > self.best_auc:
                self.best_auc = metrics["auc_roc"]
                ckpt_path = self.checkpoint_dir / "best_effresnet_vit.pth"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "metrics": metrics,
                }, ckpt_path)
                print(f"--> Saved new best checkpoint (AUC: {metrics['auc_roc']}%) to {ckpt_path}")
                
        # Save final model
        final_path = self.checkpoint_dir / "final_effresnet_vit.pth"
        torch.save(self.model.state_dict(), final_path)
        print("=" * 65)
        print(f"Training Complete. Best Validation AUC: {self.best_auc:.2f}%")
        print("=" * 65)
        return self.history
