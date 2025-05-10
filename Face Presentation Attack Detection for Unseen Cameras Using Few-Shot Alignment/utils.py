import os
import sys
import json
import pickle
import random
import math
from typing import Tuple, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from scipy.io import savemat
from tqdm import tqdm

from predict import Predictor
from models import Net
from torchvision.transforms import Resize

# Constants
SUPPORTED_IMAGE_EXTS = [".jpg", ".JPG", ".png", ".PNG"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_split_data(root: str, val_rate: float = 0.2) -> Tuple[List[str], List[int], List[str], List[int]]:
    """
    Split dataset into train/val sets and save class indices
    
    Args:
        root: Path to dataset directory
        val_rate: Ratio of validation data
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels)
    """
    random.seed(0)  # For reproducibility
    assert os.path.exists(root), f"Dataset root: {root} does not exist."

    # Get sorted class names and create indices
    classes = sorted([cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))])
    class_indices = {cls: idx for idx, cls in enumerate(classes)}
    
    # Save class indices
    with open('class_indices.json', 'w') as f:
        json.dump({v: k for k, v in class_indices.items()}, f, indent=4)

    # Initialize data containers
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    class_counts = []

    for cls in classes:
        cls_path = os.path.join(root, cls)
        images = sorted([
            os.path.join(root, cls, img) 
            for img in os.listdir(cls_path) 
            if os.path.splitext(img)[-1] in SUPPORTED_IMAGE_EXTS
        ])
        
        class_counts.append(len(images))
        val_samples = random.sample(images, k=int(len(images) * val_rate))
        
        for img in images:
            if img in val_samples:
                val_paths.append(img)
                val_labels.append(class_indices[cls])
            else:
                train_paths.append(img)
                train_labels.append(class_indices[cls])

    # Print dataset stats
    print(f"{sum(class_counts)} images found in dataset")
    print(f"{len(train_paths)} images for training")
    print(f"{len(val_paths)} images for validation")
    
    assert len(train_paths) > 0, "No training images found"
    assert len(val_paths) > 0, "No validation images found"

    return train_paths, train_labels, val_paths, val_labels

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    batch_num: int,
    writer: SummaryWriter
) -> Tuple[float, float, int]:
    """Train model for one epoch"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss().to(device)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(data_loader, desc=f"Train Epoch {epoch}", file=sys.stdout) as pbar:
        for step, (images, _, _, labels, map_labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            map_labels = map_labels.to(device)
            
            # Forward pass
            pred, map_pred, _ = model(images)
            
            # Calculate losses
            spoof_loss = criterion(pred, labels)
            map_loss = mse_loss(map_pred, map_labels)
            loss = spoof_loss + map_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = pred.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Log to tensorboard
            batch_num += images.size(0)
            writer.add_scalar('Loss/train', loss.item(), batch_num)
            writer.add_scalar('Accuracy/train', correct / total, batch_num)
            
            pbar.set_postfix({
                'loss': running_loss / (step + 1),
                'acc': correct / total
            })
    
    return running_loss / len(data_loader), correct / total, batch_num


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int
) -> Tuple[float, float, float]:
    """Evaluate model on validation set"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    
    with tqdm(data_loader, desc=f"Val Epoch {epoch}", file=sys.stdout) as pbar:
        for step, (images, _, _, labels, _) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            pred, _, _ = model(images)
            loss = criterion(pred, labels)
            
            # Update metrics
            running_loss += loss.item()
            probs = F.softmax(pred, dim=1)
            _, predicted = pred.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Collect for ROC calculation
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (step + 1),
                'acc': correct / total
            })
    
    # Calculate ROC metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs, pos_label=1)
    val_err, val_threshold = get_err_threhold(fpr, tpr, thresholds)
    
    return running_loss / len(data_loader), correct / total, val_threshold





