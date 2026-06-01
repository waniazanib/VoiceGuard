"""
End-to-end training pipeline for VoiceGuard.
Streams the ASVspoof 2019 dataset from Kaggle via kagglehub, performs class-imbalance-weighted training with mixed precision,
computes dev-set Equal Error Rate (EER) targets, checkpoints the best performer, and automatically exports to ONNX FP16.
"""

import os
# os.environ["KAGGLEHUB_CACHE_DIR"] = "D:\cache\KaggleHub"
os.environ.setdefault("KAGGLEHUB_CACHE_DIR", "D:\\cache\\KaggleHub")
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import kagglehub
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

import config
import features
from model import SpoofDetector
import export_onnx


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the Equal Error Rate (EER) where the False Acceptance Rate (FAR)
    intersects with the False Rejection Rate (FRR). Uses ROC curve calculations.
    
    Args:
        scores: 1D array of spoof probability predictions (0.0 to 1.0)
        labels: 1D array of true binary labels (0 = bonafide, 1 = spoof)
        
    Returns:
        eer_value: Equal Error Rate as a ratio float (e.g. 0.052 for 5.2%)
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # Fallback under non-binary dataset slices
        
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    
    # Linear interpolation to find precise intersection point (EER)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr, fill_value="extrapolate")(x), 0.0, 1.0)
    except Exception:
        # Fallback approximation if root solver suffers optimization failures
        idx = np.nanargmin(np.absolute(fpr - fnr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
        
    return float(eer)


class ASVspoofDataset(Dataset):
    """
    ASVspoof2019 logical access dataset interface.
    Lazily loads and processes FLAC audio tracks on demand.
    """
    def __init__(self, dataset_root: str, split: str, augment: bool = False):
        """
        Loads protocol list and index file paths.
        
        Args:
            dataset_root: Cache path retrieved from kagglehub.
            split: Target subgroup ('train', 'dev', 'eval').
            augment: If True, applies SpecAugment masks to extracted features.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.augment = augment
        
        # Select respective protocol description file
        pt_map = {
            "train": "ASVspoof2019.LA.cm.train.trn.txt",
            "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
            "eval": "ASVspoof2019.LA.cm.eval.trl.txt"
        }
        
        if split not in pt_map:
            raise ValueError(f"Invalid split name choice '{split}'. Must choose train, dev, or eval.")
            
        protocol_file = os.path.join(
            dataset_root, "LA", "ASVspoof2019_LA_cm_protocols", pt_map[split]
        )
        
        if not os.path.exists(protocol_file):
            raise FileNotFoundError(f"ASVspoof protocol file missing inside cache: {protocol_file}")
            
        self.entries = []
        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id, file_id, _, attack_type, label_str = parts[:5]
                    # Format audio FLAC file source path
                    audio_path = os.path.join(
                        dataset_root, f"LA/ASVspoof2019_LA_{split}/flac/{file_id}.flac"
                    )
                    # Use numerical labels: 0=bonafide (real), 1=spoof (fake)
                    label = 0 if label_str == "bonafide" else 1
                    
                    # Store entry only if FLAC file exists physically
                    if os.path.exists(audio_path):
                        self.entries.append({
                            "path": audio_path,
                            "label": label,
                            "attack": attack_type
                        })
                        
        print(f"[i] Dataset loaded: split={split}, size={len(self.entries)} samples")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        entry = self.entries[idx]
        
        # Load audio wave and compute Constant-Q Transform
        waveform = features.load_audio(entry["path"])
        cqt_feat = features.extract_cqt(waveform)
        
        # Apply data augmentations if designated
        if self.augment:
            cqt_feat = features.apply_specaugment(cqt_feat)
            
        # Add single channel dimensional axis: (84, T) -> (1, 84, T)
        cqt_tensor = torch.tensor(cqt_feat[np.newaxis, :, :], dtype=torch.float32)
        label_tensor = entry["label"]
        
        return cqt_tensor, label_tensor


def main():
    print("================================================================")
    print("VoiceGuard — Acoustic Deepfake Detector Training Pipeline")
    print("================================================================")
    
    
    # Step 1: Dataset Streaming / Acquisition
    if config.LOCAL_DATASET_PATH is not None:
        dataset_root = config.LOCAL_DATASET_PATH
        print(f"[+] Using configured local dataset directory: {dataset_root}")
    else:
        print("[*] Contacting Kaggle to scan/stream ASVspoof2019 dataset...")
        try:
            dataset_root = kagglehub.dataset_download(config.KAGGLE_DATASET)
            print(f"[+] Dataset available locally inside cache folder: {dataset_root}")
        except Exception as e:
            print(f"[-] Critical Error: could not acquire Kaggle source - {e}")
            print("[!] Please configure Kaggle API tokens locally if download triggers permissions errors or specify a config.LOCAL_DATASET_PATH directory.")
            return

    # Step 2: Initialize Datasets and Loaders
    print("[*] Preparing PyTorch datasets (Train & Dev)...")
    train_dataset = ASVspoofDataset(dataset_root, "train", augment=True)
    dev_dataset = ASVspoofDataset(dataset_root, "dev", augment=False)
    
    # Cap parameters for quick debugging/run sanity checks if dataset proves massive
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Step 3: Compute class imbalance weights for CrossEntropyLoss
    print("[*] Analyzing training classes to establish dynamic weights...")
    labels_list = [entry["label"] for entry in train_dataset.entries]
    class_counts = np.bincount(labels_list)
    total_samples = len(labels_list)
    
    # Weights proportional to inverse frequency: total / (classes * class_count)
    weight_bonafide = total_samples / (2.0 * class_counts[0])
    weight_spoof = total_samples / (2.0 * class_counts[1])
    class_weights = torch.tensor([weight_bonafide, weight_spoof], dtype=torch.float32)
    print(f"[i] Class distribution: Bonafide={class_counts[0]}, Spoof={class_counts[1]}")
    print(f"[i] Calculated loss weights: Bonafide={weight_bonafide:.3f}, Spoof={weight_spoof:.3f}")

    # Set compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[i] Activating compute hardware: {device.type.upper()}")
    
    # Loss, model, optimization
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    model = SpoofDetector().to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    # Setup FP16 mixed precision handles
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # Track historical metric progress
    best_eer = 1.0  # lower is better
    history = []
    
    print("\nStarting optimization loops...")
    print("Epoch | Train Loss | Dev EER | Best EER")
    print("------+------------+---------+---------")
    
    for epoch in range(1, config.EPOCHS + 1):
        # 1. Training Pass
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass using Mixed Precision
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                
            # Backward and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss = epoch_loss / len(train_dataset)
        scheduler.step()
        
        # 2. Evaluation Pass
        model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in dev_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                
                # Compute spoof probability via softmax (spoof index is 1)
                probs = torch.softmax(logits, dim=1)
                spoof_probs = probs[:, 1].cpu().numpy()
                
                all_scores.extend(spoof_probs)
                all_labels.extend(batch_y.numpy())
                
        # Compute EER
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        dev_eer = compute_eer(all_scores, all_labels)
        
        # Track better runs
        saved_flag = ""
        if dev_eer < best_eer:
            best_eer = dev_eer
            torch.save(model.state_dict(), str(config.MODEL_PT_PATH))
            saved_flag = " (SAVED CHKPNT)"
            
        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "dev_eer": dev_eer,
            "best_eer": best_eer
        })
        
        # Output progression row
        print(f" {epoch:2d}   |   {epoch_loss:.3f}    |  {100*dev_eer:5.1f}%  |  {100*best_eer:5.1f}%{saved_flag}")
        
    print("\n[+] Training stages successfully completed.")
    print(f"[+] Best PyTorch model saved to: {config.MODEL_PT_PATH}")
    
    # Step 4: ONNX Deployment Exports
    try:
        print("\n[*] Initializing downstream ONNX conversion and quantization pipeline...")
        export_onnx.export(str(config.MODEL_PT_PATH))
    except Exception as e:
        print(f"[-] Execution Warning: automatic ONNX compilation failed - {e}")
        
    # Step 5: Save Metadata JSON File
    metadata = {
        "architecture": "ResNet-18",
        "feature_type": "CQT",
        "cqt_bins": config.CQT_BINS,
        "hop_length": config.HOP_LENGTH,
        "sample_rate": config.SAMPLE_RATE,
        "clip_duration": config.CLIP_DURATION,
        "best_dev_eer": float(best_eer),
        "trained_on": "ASVspoof2019 LA",
        "kaggle_dataset": config.KAGGLE_DATASET,
        "trained_epochs": config.EPOCHS,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[+] Saved model configuration metadata: {config.MODEL_METADATA_PATH}")
    print("\nEnjoy detecting deepfakes with VoiceGuard!")


if __name__ == "__main__":
    main()
