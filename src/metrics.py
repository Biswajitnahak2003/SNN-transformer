"""
Metrics module: Dice Score, Hausdorff Distance, and Energy Tracking.
"""

import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


# ─── Dice Score ──────────────────────────────────────────────────────────────

def dice_score(pred, target, num_classes=4, smooth=1e-5):
    """
    Compute per-class and mean Dice coefficient.
    
    Args:
        pred: (B, H, W) predicted class indices
        target: (B, H, W) ground truth class indices
        num_classes: number of segmentation classes
        smooth: smoothing factor to avoid division by zero
    
    Returns:
        per_class_dice: dict mapping class_idx -> dice value
        mean_dice: float mean dice over non-background classes
    """
    per_class_dice = {}
    dice_values = []

    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2.0 * intersection + smooth) / (union + smooth)
        per_class_dice[c] = dice.item()

        if c > 0:  # Skip background for mean dice
            dice_values.append(dice.item())

    mean_dice = np.mean(dice_values) if dice_values else 0.0
    return per_class_dice, mean_dice


class DiceLoss(torch.nn.Module):
    """Differentiable Dice Loss for training."""

    def __init__(self, num_classes=4, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw model output
            targets: (B, H, W) ground truth class indices
        """
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(
            targets, self.num_classes
        ).permute(0, 3, 1, 2).float()

        loss = 0.0
        for c in range(1, self.num_classes):  # Skip background
            p = probs[:, c]
            t = targets_one_hot[:, c]
            intersection = (p * t).sum(dim=(-1, -2))
            union = p.sum(dim=(-1, -2)) + t.sum(dim=(-1, -2))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            loss += 1.0 - dice.mean()

        return loss / (self.num_classes - 1)


# ─── Hausdorff Distance ─────────────────────────────────────────────────────

def hausdorff_distance_95(pred, target, num_classes=4):
    """
    Compute 95th percentile Hausdorff distance per class.

    Args:
        pred: (B, H, W) predicted class indices (numpy or tensor)
        target: (B, H, W) ground truth class indices
    
    Returns:
        per_class_hd95: dict mapping class_idx -> HD95 value
        mean_hd95: float
    """
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()

    per_class_hd95 = {}
    hd_values = []

    for c in range(1, num_classes):  # Skip background
        pred_points = np.argwhere(pred == c)
        target_points = np.argwhere(target == c)

        if len(pred_points) == 0 and len(target_points) == 0:
            per_class_hd95[c] = 0.0
            hd_values.append(0.0)
            continue
        if len(pred_points) == 0 or len(target_points) == 0:
            per_class_hd95[c] = float('inf')
            continue

        # Compute directed Hausdorff distances
        d_forward = directed_hausdorff(pred_points, target_points)[0]
        d_backward = directed_hausdorff(target_points, pred_points)[0]

        # 95th percentile approximation via max of directed distances
        from scipy.spatial import cKDTree
        tree_target = cKDTree(target_points)
        tree_pred = cKDTree(pred_points)

        dists_pred_to_target, _ = tree_target.query(pred_points)
        dists_target_to_pred, _ = tree_pred.query(target_points)

        all_dists = np.concatenate([dists_pred_to_target, dists_target_to_pred])
        hd95 = np.percentile(all_dists, 95)

        per_class_hd95[c] = hd95
        hd_values.append(hd95)

    mean_hd95 = np.mean(hd_values) if hd_values else 0.0
    return per_class_hd95, mean_hd95


# ─── Running Metric Tracker ─────────────────────────────────────────────────

class MetricTracker:
    """Track running averages of metrics during training/validation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}
        self.counts = {}

    def update(self, name, value, count=1):
        if name not in self.values:
            self.values[name] = 0.0
            self.counts[name] = 0
        self.values[name] += value * count
        self.counts[name] += count

    def get(self, name):
        if name not in self.values or self.counts[name] == 0:
            return 0.0
        return self.values[name] / self.counts[name]

    def summary(self):
        return {k: self.get(k) for k in self.values}


# ─── Energy Tracker ──────────────────────────────────────────────────────────

class EnergyTracker:
    """Track computational savings from adaptive timesteps."""

    def __init__(self, T_max, num_patches=64):
        self.T_max = T_max
        self.num_patches = num_patches
        self.baseline_ops = 0    # Full T timesteps for all patches
        self.actual_ops = 0      # Actual ops with adaptive timesteps
        self.n_images = 0

    def update(self, timestep_map):
        """
        Update with a timestep map from the agent.
        
        Args:
            timestep_map: (B, num_patches) tensor of per-patch timestep assignments
        """
        if torch.is_tensor(timestep_map):
            timestep_map = timestep_map.cpu().numpy()
        
        batch_size = timestep_map.shape[0]
        for i in range(batch_size):
            self.baseline_ops += self.num_patches * self.T_max
            self.actual_ops += timestep_map[i].sum()
            self.n_images += 1

    def get_savings(self):
        if self.baseline_ops == 0:
            return 0.0
        return 1.0 - (self.actual_ops / self.baseline_ops)

    def summary(self):
        return {
            'n_images': self.n_images,
            'baseline_ops': self.baseline_ops,
            'actual_ops': self.actual_ops,
            'savings_pct': self.get_savings() * 100,
            'avg_timestep': self.actual_ops / max(1, self.n_images * self.total_pixels)
        }
