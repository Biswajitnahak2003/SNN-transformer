"""
Training loop — two-phase training with HuggingFace Accelerate.

Phase 1 (epochs 1–10): Train base SNN with fixed T timesteps (all pixels).
Phase 2 (epochs 11–20): Enable CNN agent → adaptive timesteps for each image.

Uses Accelerate for multi-GPU (2×T4) with mixed-precision (fp16).
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator

from metrics import DiceLoss, dice_score, hausdorff_distance_95, MetricTracker, EnergyTracker
from utils import (save_checkpoint, plot_loss_curves, plot_dice_curves,
                   plot_sample_predictions, plot_energy_savings, set_seed)
from snn_model import AdaptiveTimestepSNN
from agent import CNNTimestepAgent, AdaptiveTimestepPipeline
from dataset import get_dataloaders


# ─── Configuration ───────────────────────────────────────────────────────────

class TrainConfig:
    """All training hyperparameters in one place."""
    # Data
    data_dir: str = "/kaggle/input/datasets/luumsk/asnr-miccai-brats-2023-gli-challenge-training-data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    slice_size: tuple = (128, 128)
    batch_size: int = 2
    num_workers: int = 2

    # Model
    in_channels: int = 4
    num_classes: int = 4
    img_size: int = 128
    patch_size: int = 16
    embed_dim: int = 256
    depth: int = 6
    num_heads: int = 8
    T: int = 4            # max spiking timesteps
    beta: float = 0.5     # LIF decay rate
    num_patches: int = (img_size // patch_size) ** 2  # 64
    num_patches: int = (img_size // patch_size) ** 2  # 64

    # Training
    total_epochs: int = 20
    warmup_epochs: int = 10  # Phase 1 (no agent)
    lr: float = 1e-3
    weight_decay: float = 1e-4
    agent_lr: float = 5e-4
    agent_threshold: float = 0.3

    # Outputs
    save_dir: str = "outputs"
    seed: int = 42


# ─── Train / Val Loops ──────────────────────────────────────────────────────

def train_one_epoch(model, agent, dataloader, optimizer, criterion,
                    ce_loss_fn, accelerator, config, use_agent=False):
    """One training epoch."""
    model.train()
    if agent is not None:
        agent.train()

    tracker = MetricTracker()
    energy = EnergyTracker(config.T, config.num_patches)

    pbar = tqdm(dataloader, desc="Train", disable=not accelerator.is_main_process)
    for batch in pbar:
        images = batch['image']   # (B, 4, H, W)
        masks  = batch['mask']    # (B, H, W)

        optimizer.zero_grad()

        if use_agent and agent is not None:
            # Phase 2: adaptive timesteps
            with torch.no_grad():
                init_logits, _ = model.forward_single_timestep(images)
            t_map, _, soft_t = agent(images, init_logits,
                                     threshold=config.agent_threshold)
            logits = model(images, timestep_map=t_map)
            energy.update(t_map)
        else:
            # Phase 1: fixed T timesteps
            logits = model(images, timestep_map=None)

        # Combined loss: Dice + CE
        dice_l = criterion(logits, masks)
        ce_l   = ce_loss_fn(logits, masks)
        loss   = dice_l + ce_l

        accelerator.backward(loss)
        # Gradient clipping
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        preds = logits.argmax(dim=1)
        _, m_dice = dice_score(preds, masks, config.num_classes)

        tracker.update('loss', loss.item())
        tracker.update('dice', m_dice)

        pbar.set_postfix(loss=f"{tracker.get('loss'):.4f}",
                         dice=f"{tracker.get('dice'):.4f}")

    return tracker.summary(), energy


@torch.no_grad()
def validate(model, agent, dataloader, criterion, ce_loss_fn,
             accelerator, config, use_agent=False):
    """Validation loop."""
    model.eval()
    if agent is not None:
        agent.eval()

    tracker = MetricTracker()
    energy = EnergyTracker(config.T, config.num_patches)
    all_hd95 = []

    # Save a few samples for visualisation
    sample_images, sample_masks, sample_preds = [], [], []

    pbar = tqdm(dataloader, desc="Val  ", disable=not accelerator.is_main_process)
    for batch in pbar:
        images = batch['image']
        masks  = batch['mask']

        if use_agent and agent is not None:
            init_logits, _ = model.forward_single_timestep(images)
            t_map, _, _ = agent(images, init_logits,
                                threshold=config.agent_threshold)
            logits = model(images, timestep_map=t_map)
            energy.update(t_map)
        else:
            logits = model(images, timestep_map=None)

        dice_l = criterion(logits, masks)
        ce_l   = ce_loss_fn(logits, masks)
        loss   = dice_l + ce_l

        preds = logits.argmax(dim=1)
        _, m_dice = dice_score(preds, masks, config.num_classes)

        # Hausdorff distance (expensive — sample)
        try:
            _, hd95 = hausdorff_distance_95(preds, masks, config.num_classes)
            if hd95 < float('inf'):
                all_hd95.append(hd95)
        except Exception:
            pass

        tracker.update('loss', loss.item())
        tracker.update('dice', m_dice)

        # Collect samples
        if len(sample_images) < 4:
            sample_images.append(images[:1].cpu())
            sample_masks.append(masks[:1].cpu())
            sample_preds.append(preds[:1].cpu())

    tracker.update('hd95', np.mean(all_hd95) if all_hd95 else 0.0)

    samples = None
    if sample_images:
        samples = {
            'images': torch.cat(sample_images, dim=0),
            'masks':  torch.cat(sample_masks, dim=0),
            'preds':  torch.cat(sample_preds, dim=0),
        }

    return tracker.summary(), energy, samples


# ─── Main Training ───────────────────────────────────────────────────────────

def train(config=None):
    """Full two-phase training pipeline."""
    if config is None:
        config = TrainConfig()

    set_seed(config.seed)

    # Accelerate setup
    accelerator = Accelerator(mixed_precision='fp16')
    device = accelerator.device

    if accelerator.is_main_process:
        os.makedirs(config.save_dir, exist_ok=True)
        print(f"{'='*60}")
        print(f"  Adaptive Timestep SNN — Brain Tumor Segmentation")
        print(f"  Device: {device} | GPUs: {accelerator.num_processes}")
        print(f"  T={config.T} | Epochs={config.total_epochs} "
              f"| Warmup={config.warmup_epochs}")
        print(f"{'='*60}")

    # Data
    dataloaders = get_dataloaders(
        config.data_dir, batch_size=config.batch_size,
        slice_size=config.slice_size, num_workers=config.num_workers
    )

    # Model + Agent
    model = AdaptiveTimestepSNN(
        config.in_channels, config.num_classes, config.T,
        config.base_channels, config.beta
    )
    agent = CNNTimestepAgent(
        config.in_channels, config.num_classes, config.T
    )

    # Optimizers
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(agent.parameters()),
        lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_epochs, eta_min=1e-6
    )

    # Losses
    dice_loss_fn = DiceLoss(config.num_classes)
    ce_loss_fn   = nn.CrossEntropyLoss()

    # Accelerate wrapping
    model, agent, optimizer, scheduler = accelerator.prepare(
        model, agent, optimizer, scheduler
    )
    dataloaders['train'] = accelerator.prepare(dataloaders['train'])
    dataloaders['val']   = accelerator.prepare(dataloaders['val'])
    dice_loss_fn = dice_loss_fn.to(device)

    # History
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'val_hd95': [], 'energy_savings': []
    }
    best_val_dice = 0.0
    start_time = time.time()

    for epoch in range(1, config.total_epochs + 1):
        use_agent = epoch > config.warmup_epochs
        phase = "Phase 2 (Agent)" if use_agent else "Phase 1 (Warmup)"

        if accelerator.is_main_process:
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch}/{config.total_epochs} — {phase} "
                  f"| LR: {scheduler.get_last_lr()[0]:.2e}")

        # Train
        train_metrics, train_energy = train_one_epoch(
            model, agent, dataloaders['train'], optimizer,
            dice_loss_fn, ce_loss_fn, accelerator, config,
            use_agent=use_agent
        )

        # Validate
        val_metrics, val_energy, samples = validate(
            model, agent, dataloaders['val'],
            dice_loss_fn, ce_loss_fn, accelerator, config,
            use_agent=use_agent
        )

        scheduler.step()

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_hd95'].append(val_metrics.get('hd95', 0.0))

        savings = val_energy.get_savings() * 100 if use_agent else 0.0
        history['energy_savings'].append(savings)

        if accelerator.is_main_process:
            elapsed = (time.time() - start_time) / 60
            print(f"  Train — loss: {train_metrics['loss']:.4f} | "
                  f"dice: {train_metrics['dice']:.4f}")
            print(f"  Val   — loss: {val_metrics['loss']:.4f} | "
                  f"dice: {val_metrics['dice']:.4f} | "
                  f"HD95: {val_metrics.get('hd95', 0.0):.2f}")
            if use_agent:
                print(f"  Energy saved: {savings:.1f}%")
            print(f"  Elapsed: {elapsed:.1f} min")

            # Save best model
            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(
                    unwrapped, optimizer, epoch,
                    {'val_dice': best_val_dice},
                    os.path.join(config.save_dir, 'best_model.pt')
                )
                print(f"  ★ New best val dice: {best_val_dice:.4f}")

    # ── Final outputs ────────────────────────────────────────────────────
    if accelerator.is_main_process:
        total_time = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"  Training complete in {total_time:.1f} minutes")
        print(f"  Best val dice: {best_val_dice:.4f}")
        print(f"{'='*60}")

        # Generate plots
        plot_loss_curves(
            history['train_loss'], history['val_loss'],
            os.path.join(config.save_dir, 'loss_curves.png')
        )
        plot_dice_curves(
            history['train_dice'], history['val_dice'],
            os.path.join(config.save_dir, 'dice_curves.png')
        )

        if samples is not None:
            plot_sample_predictions(
                samples['images'], samples['masks'], samples['preds'],
                save_path=os.path.join(config.save_dir, 'sample_preds.png')
            )

        if any(s > 0 for s in history['energy_savings']):
            plot_energy_savings(
                val_energy,
                os.path.join(config.save_dir, 'energy_savings.png')
            )

        # Save history
        import json
        with open(os.path.join(config.save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    return history, model


if __name__ == '__main__':
    train()
