#!/usr/bin/env python3
"""
Inference script for Adaptive Timestep SNN Brain Tumor Segmentation.

Usage:
    python inference.py --checkpoint path/to/model.pt --data_dir path/to/data --output_dir results/
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import nibabel as nib
from pathlib import Path

from src.snn_model import AdaptiveTimestepSNN
from src.agent import CNNTimestepAgent, AdaptiveTimestepPipeline
from src.dataset import BraTSDataset
from src.metrics import dice_score, hausdorff_distance_95
from src.utils import set_seed


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    # Initialize model
    model = AdaptiveTimestepSNN(
        in_channels=4,
        num_classes=4,
        img_size=128,
        patch_size=16,
        embed_dim=256,
        depth=6,
        num_heads=8,
        T=4,
        beta=0.5
    )

    # Initialize agent
    agent = CNNTimestepAgent(in_channels=8, hidden_dim=32)

    # Create pipeline
    pipeline = AdaptiveTimestepPipeline(model, agent)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pipeline.load_state_dict(checkpoint['model_state_dict'])
    pipeline.to(device)
    pipeline.eval()

    print(f"Loaded model from {checkpoint_path}")
    return pipeline


def predict_single_volume(model, volume_data, device, slice_thickness=1.0):
    """Predict segmentation for a single 3D volume."""
    model.eval()

    # Get volume dimensions
    depth, height, width, channels = volume_data.shape
    predictions = []

    with torch.no_grad():
        for slice_idx in tqdm(range(depth), desc="Processing slices"):
            # Extract 2D slice
            slice_data = volume_data[slice_idx]  # (H, W, 4)

            # Convert to tensor and add batch dimension
            slice_tensor = torch.from_numpy(slice_data).float().permute(2, 0, 1).unsqueeze(0)  # (1, 4, H, W)
            slice_tensor = slice_tensor.to(device)

            # Forward pass
            pred_logits = model(slice_tensor)  # (1, 4, H, W)

            # Convert to probabilities
            pred_probs = torch.softmax(pred_logits, dim=1)

            # Get predicted classes
            pred_mask = torch.argmax(pred_probs, dim=1).squeeze(0).cpu().numpy()  # (H, W)

            predictions.append(pred_mask)

    # Stack predictions into 3D volume
    pred_volume = np.stack(predictions, axis=0)  # (D, H, W)

    return pred_volume


def evaluate_predictions(gt_volume, pred_volume):
    """Compute evaluation metrics."""
    # Flatten volumes
    gt_flat = gt_volume.flatten()
    pred_flat = pred_volume.flatten()

    # Compute Dice coefficient
    dice = dice_score(gt_flat, pred_flat)

    # Compute Hausdorff distance (simplified 2D version per slice)
    hd95_scores = []
    for slice_idx in range(gt_volume.shape[0]):
        gt_slice = gt_volume[slice_idx]
        pred_slice = pred_volume[slice_idx]

        if np.sum(gt_slice) > 0:  # Only compute if there are ground truth labels
            hd95 = hausdorff_distance_95(gt_slice, pred_slice)
            hd95_scores.append(hd95)

    avg_hd95 = np.mean(hd95_scores) if hd95_scores else float('inf')

    return dice, avg_hd95


def main():
    parser = argparse.ArgumentParser(description="Adaptive SNN Brain Tumor Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to BraTS data directory")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                       help="Output directory for predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    # Get test data (you'll need to modify this based on your data structure)
    data_dir = Path(args.data_dir)

    # Example: Process first few cases
    case_dirs = list(data_dir.glob("BraTS-GLI-*"))
    case_dirs = case_dirs[:5]  # Process first 5 cases for demo

    results = []

    for case_dir in case_dirs:
        case_id = case_dir.name
        print(f"\nProcessing case: {case_id}")

        try:
            # Load MRI modalities (you'll need to implement proper loading)
            # This is a simplified example - adapt to your data format
            modalities = ['t1c', 't1n', 't2f', 't2w']

            # Load ground truth segmentation
            seg_path = case_dir / f"{case_id}-seg.nii"
            if seg_path.exists():
                seg_nii = nib.load(str(seg_path))
                gt_volume = seg_nii.get_fdata().astype(np.uint8)

                # Load MRI volumes (simplified - you'll need proper preprocessing)
                mri_volumes = []
                for mod in modalities:
                    mod_path = case_dir / f"{case_id}-{mod}.nii"
                    if mod_path.exists():
                        mod_nii = nib.load(str(mod_path))
                        mod_data = mod_nii.get_fdata()
                        mri_volumes.append(mod_data)

                if len(mri_volumes) == 4:
                    # Stack modalities
                    input_volume = np.stack(mri_volumes, axis=-1)  # (D, H, W, 4)

                    # Run prediction
                    pred_volume = predict_single_volume(model, input_volume, device)

                    # Evaluate
                    dice, hd95 = evaluate_predictions(gt_volume, pred_volume)

                    # Save prediction
                    pred_nii = nib.Nifti1Image(pred_volume.astype(np.uint8), seg_nii.affine)
                    pred_path = output_dir / f"{case_id}_pred.nii"
                    nib.save(pred_nii, str(pred_path))

                    results.append({
                        'case_id': case_id,
                        'dice': dice,
                        'hd95': hd95
                    })

                    print(".4f")
                else:
                    print(f"  Skipping {case_id}: Missing MRI modalities")
            else:
                print(f"  Skipping {case_id}: No segmentation ground truth")

        except Exception as e:
            print(f"  Error processing {case_id}: {str(e)}")
            continue

    # Save results summary
    if results:
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_path = output_dir / "inference_results.csv"
        results_df.to_csv(results_path, index=False)

        # Print summary
        print("
📊 Inference Summary:"        print(f"  Cases processed: {len(results)}")
        print(".4f")
        print(".2f")

        print(f"\nResults saved to: {results_path}")
    else:
        print("No cases were successfully processed.")


if __name__ == "__main__":
    main()