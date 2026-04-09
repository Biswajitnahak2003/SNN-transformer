"""
Utility functions: plotting, checkpoints, seeding, visualization.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('metrics', {})


# ─── Plotting Functions ─────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    ax.plot(epochs, val_losses, 'r-o', label='Val Loss', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_dice_curves(train_dices, val_dices, save_path=None):
    """Plot training and validation Dice curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_dices) + 1)
    ax.plot(epochs, train_dices, 'b-o', label='Train Dice', markersize=4)
    ax.plot(epochs, val_dices, 'g-o', label='Val Dice', markersize=4)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Training & Validation Dice Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(images, masks, preds, n_samples=4, save_path=None):
    """
    Visualize sample predictions vs ground truth.
    
    Args:
        images: (N, 4, H, W) input images
        masks: (N, H, W) ground truth masks
        preds: (N, H, W) predicted masks
        n_samples: number of samples to show
    """
    n_samples = min(n_samples, len(images))
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    class_names = ['Background', 'NCR', 'ED', 'ET']
    cmap = plt.cm.get_cmap('Set1', 4)

    for i in range(n_samples):
        # Show first modality (T1CE)
        ax = axes[i, 0]
        ax.imshow(images[i, 0].cpu().numpy(), cmap='gray')
        ax.set_title('T1CE Input' if i == 0 else '')
        ax.axis('off')

        # Show all 4 modalities as composite
        ax = axes[i, 1]
        composite = images[i].cpu().numpy().mean(axis=0)
        ax.imshow(composite, cmap='gray')
        ax.set_title('4-Modal Average' if i == 0 else '')
        ax.axis('off')

        # Ground truth mask
        ax = axes[i, 2]
        ax.imshow(masks[i].cpu().numpy(), cmap=cmap, vmin=0, vmax=3)
        ax.set_title('Ground Truth' if i == 0 else '')
        ax.axis('off')

        # Predicted mask
        ax = axes[i, 3]
        ax.imshow(preds[i].cpu().numpy(), cmap=cmap, vmin=0, vmax=3)
        ax.set_title('Prediction' if i == 0 else '')
        ax.axis('off')

    plt.suptitle('Sample Predictions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_energy_savings(energy_tracker, save_path=None):
    """Plot energy savings from adaptive timesteps."""
    summary = energy_tracker.summary()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: baseline vs actual ops
    ax = axes[0]
    bars = ax.bar(
        ['Baseline\n(Full T)', 'Adaptive\n(Agent)'],
        [summary['baseline_ops'], summary['actual_ops']],
        color=['#e74c3c', '#2ecc71'], width=0.5, edgecolor='black'
    )
    ax.set_ylabel('Total Operations (pixel×timesteps)', fontsize=11)
    ax.set_title('Compute: Baseline vs Adaptive', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Pie chart: saved vs used
    ax = axes[1]
    saved = summary['baseline_ops'] - summary['actual_ops']
    ax.pie(
        [summary['actual_ops'], max(0, saved)],
        labels=['Used', 'Saved'],
        colors=['#3498db', '#2ecc71'],
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12}
    )
    ax.set_title(f'Energy Savings: {summary["savings_pct"]:.1f}%',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*50}")
    print(f"Energy Summary:")
    print(f"  Images processed: {summary['n_images']}")
    print(f"  Baseline ops:     {summary['baseline_ops']:,.0f}")
    print(f"  Actual ops:       {summary['actual_ops']:,.0f}")
    print(f"  Savings:          {summary['savings_pct']:.1f}%")
    print(f"  Avg timestep/px:  {summary['avg_timestep']:.2f} / {4}")
    print(f"{'='*50}")


def plot_timestep_map(image, timestep_map, save_path=None):
    """Visualize the agent's timestep map overlaid on the image."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image (first modality)
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(timestep_map):
        timestep_map = timestep_map.cpu().numpy()

    axes[0].imshow(image[0] if image.ndim == 3 else image, cmap='gray')
    axes[0].set_title('Input (T1CE)', fontsize=12)
    axes[0].axis('off')

    # Timestep map
    im = axes[1].imshow(timestep_map, cmap='hot', vmin=1)
    axes[1].set_title('Timestep Map', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay
    axes[2].imshow(image[0] if image.ndim == 3 else image, cmap='gray')
    axes[2].imshow(timestep_map, cmap='hot', alpha=0.5, vmin=1)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Agent Timestep Assignment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
