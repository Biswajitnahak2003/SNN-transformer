"""
BraTS 2023 Dataset - 2D Slice-wise Loading.

Loads 4 modality NIfTI volumes (t1c, t1n, t2f, t2w), extracts 2D axial slices,
applies augmentations, and provides train/val DataLoaders.
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─── Dataset ─────────────────────────────────────────────────────────────────

class BraTSDataset(Dataset):
    """BraTS 2023 GLI Dataset — 2D axial slice extraction."""

    MODALITY_SUFFIXES = ['t1c', 't1n', 't2f', 't2w']

    def __init__(self, data_dir, patient_ids=None, transform=None,
                 slice_size=(128, 128), min_brain_fraction=0.02,
                 cache_size=20):
        """
        Args:
            data_dir: Root directory containing patient folders.
            patient_ids: List of patient folder names (None = use all).
            transform: Optional augmentation callable(image, mask) -> (image, mask).
            slice_size: Resize slices to (H, W).
            min_brain_fraction: Minimum non-zero fraction to keep a slice.
            cache_size: Max patients to keep in memory cache.
        """
        self.data_dir = data_dir
        self.slice_size = slice_size
        self.transform = transform
        self.min_brain_fraction = min_brain_fraction
        self.cache_size = cache_size

        # Find patient directories
        if patient_ids is None:
            self.patient_dirs = sorted(
                glob.glob(os.path.join(data_dir, "BraTS-GLI-*"))
            )
        else:
            self.patient_dirs = [
                os.path.join(data_dir, pid) for pid in patient_ids
            ]

        # Pre-compute valid (patient_idx, slice_idx) pairs
        self.slices = []
        self._cache = {}
        self._cache_order = []
        self._build_slice_index()

    # ── File discovery ──

    def _find_modality_file(self, patient_dir, patient_name, suffix):
        """Locate a modality NIfTI file, handling multiple directory layouts."""
        # Layout 1: direct files  (Kaggle standard)
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(patient_dir, f"{patient_name}-{suffix}{ext}")
            if os.path.isfile(path):
                return path

        # Layout 2: directory named <patient>-<suffix>.nii containing real file
        dir_path = os.path.join(patient_dir, f"{patient_name}-{suffix}.nii")
        if os.path.isdir(dir_path):
            for f in sorted(os.listdir(dir_path)):
                if f.endswith(('.nii', '.nii.gz')):
                    return os.path.join(dir_path, f)

        raise FileNotFoundError(
            f"Cannot find modality '{suffix}' for {patient_name} in {patient_dir}"
        )

    def _find_seg_file(self, patient_dir, patient_name):
        """Locate the segmentation mask file."""
        for ext in ['.nii.gz', '.nii']:
            path = os.path.join(patient_dir, f"{patient_name}-seg{ext}")
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"Cannot find segmentation mask for {patient_name}"
        )

    # ── Data loading via Memory Mapped proxy ──

    def _get_slice(self, pdir, patient_name, s_idx):
        """Directly read only the necessary 2D slice from disk using memory-mapped dataobj."""
        img_ch = []
        for suffix in self.MODALITY_SUFFIXES:
            path = self._find_modality_file(pdir, patient_name, suffix)
            # .dataobj provides a memory-mapped array proxy
            # Slicing it reads only that chunk from disk, drastically faster than loading full 3D vol
            slice_data = nib.load(path).dataobj[:, :, s_idx]
            img_ch.append(np.array(slice_data, dtype=np.float32))
            
        seg_path = self._find_seg_file(pdir, patient_name)
        seg_data = nib.load(seg_path).dataobj[:, :, s_idx]
        seg = np.array(seg_data, dtype=np.int64)
        
        return np.stack(img_ch, axis=0), seg

    # ── Slice index ──

    def _build_slice_index(self):
        """Scan patients to find valid (non-empty) axial slices, with caching."""
        import json
        
        # Try to load pre-computed index
        index_cache_path = os.path.join(self.data_dir, "slice_index_cache.json")
        try:
            if os.path.exists(index_cache_path):
                with open(index_cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Verify cache is for our specific patients list
                if len(cached_data.get('patient_dirs', [])) == len(self.patient_dirs):
                    print(f"Loading cached slice index from {index_cache_path}...")
                    self.slices = cached_data['slices']
                    return
        except Exception as e:
            print(f"Could not load slice cache: {e}")

        print(f"Building slice index for {len(self.patient_dirs)} patients... (this may take a few minutes)")
        
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(self.patient_dirs), total=len(self.patient_dirs))
        except ImportError:
            iterator = enumerate(self.patient_dirs)
            
        for p_idx, pdir in iterator:
            patient_name = os.path.basename(pdir)
            
            # OPTIMIZATION: Only load the segmentation mask to find valid slices
            # Much faster than loading all 5 high-res modalities
            try:
                seg_path = self._find_seg_file(pdir, patient_name)
                # mmap=True is critical for fast reading without loading full file to RAM
                seg_img = nib.load(seg_path)
                seg = seg_img.get_fdata(dtype=np.float32)
                
                n_slices = seg.shape[-1]
                
                for s in range(n_slices):
                    # For segmentation mask, any non-zero pixel means brain/tumor is present
                    brain_frac = np.mean(seg[:, :, s] != 0)
                    if brain_frac >= self.min_brain_fraction:
                        self.slices.append((p_idx, s))
            except Exception as e:
                print(f"  Warning: Could not process {patient_name}: {e}")

        print(f"  Found {len(self.slices)} valid slices.")
        
        # Save cache for next time
        try:
            # We can only save if we have write access to the data dir (often not true on Kaggle)
            # So we'll try to write to current working directory instead if data_dir is read-only
            if not os.access(self.data_dir, os.W_OK):
                index_cache_path = "slice_index_cache.json"
                
            with open(index_cache_path, 'w') as f:
                json.dump({
                    'patient_dirs': self.patient_dirs,
                    'slices': self.slices
                }, f)
            print(f"  Saved slice index cache to {index_cache_path}")
        except Exception as e:
            print(f"  Could not save cache (read-only filesystem?): {e}")

        # Clear memory cache
        self._cache.clear()
        self._cache_order.clear()

    # ── __getitem__ ──

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        p_idx, s_idx = self.slices[idx]
        pdir = self.patient_dirs[p_idx]
        patient_name = os.path.basename(pdir)
        
        # Read exact 2D slice from disk memory-map
        img, msk = self._get_slice(pdir, patient_name, s_idx)

        # Per-modality z-score normalisation (non-zero voxels only)
        for c in range(4):
            ch = img[c]
            nz = ch[ch != 0]
            if len(nz) > 0:
                img[c] = np.where(ch != 0, (ch - nz.mean()) / (nz.std() + 1e-8), 0)

        # To tensors
        img = torch.from_numpy(img)          # (4, H, W)
        msk = torch.from_numpy(msk).long()   # (H, W)

        # Remap BraTS labels: 0,1,2,4 -> 0,1,2,3
        msk[msk == 4] = 3

        # Resize
        img = F.interpolate(
            img.unsqueeze(0), size=self.slice_size,
            mode='bilinear', align_corners=False
        ).squeeze(0)
        msk = F.interpolate(
            msk.float().unsqueeze(0).unsqueeze(0), size=self.slice_size,
            mode='nearest'
        ).squeeze(0).squeeze(0).long()

        # Augmentations
        if self.transform is not None:
            img, msk = self.transform(img, msk)

        return {'image': img, 'mask': msk}


# ─── Augmentations ───────────────────────────────────────────────────────────

class BraTSAugmentation:
    """Basic augmentations: flip, rotate, intensity jitter."""

    def __init__(self, flip_prob=0.5, max_angle=15, intensity_jitter=0.1):
        self.flip_prob = flip_prob
        self.max_angle = max_angle
        self.intensity_jitter = intensity_jitter

    def __call__(self, image, mask):
        # Random horizontal flip
        if torch.rand(1).item() < self.flip_prob:
            image = torch.flip(image, [-1])
            mask = torch.flip(mask, [-1])

        # Random vertical flip
        if torch.rand(1).item() < self.flip_prob * 0.5:
            image = torch.flip(image, [-2])
            mask = torch.flip(mask, [-2])

        # Random rotation via affine grid
        if self.max_angle > 0 and torch.rand(1).item() < 0.5:
            angle = (torch.rand(1).item() * 2 - 1) * self.max_angle
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            theta = torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                dtype=torch.float32
            ).unsqueeze(0)
            grid = F.affine_grid(theta, image.unsqueeze(0).size(), align_corners=False)
            image = F.grid_sample(
                image.unsqueeze(0), grid, mode='bilinear', align_corners=False
            ).squeeze(0)
            mask = F.grid_sample(
                mask.float().unsqueeze(0).unsqueeze(0), grid,
                mode='nearest', align_corners=False
            ).squeeze(0).squeeze(0).long()

        # Random intensity scaling
        if torch.rand(1).item() < 0.3:
            scale = 1.0 + (torch.rand(1).item() * 2 - 1) * self.intensity_jitter
            image = image * scale

        return image, mask


# ─── DataLoader Factory ─────────────────────────────────────────────────────

def get_dataloaders(data_dir, batch_size=2, val_split=0.2,
                    slice_size=(128, 128), num_workers=2, seed=42):
    """
    Create train / val DataLoaders with patient-level split.

    Returns:
        dict with keys 'train' and 'val', each a DataLoader.
    """
    patient_dirs = sorted(glob.glob(os.path.join(data_dir, "BraTS-GLI-*")))
    patient_ids = [os.path.basename(d) for d in patient_dirs]

    np.random.seed(seed)
    np.random.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_split))
    val_ids = patient_ids[:n_val]
    train_ids = patient_ids[n_val:]

    print(f"Train patients: {len(train_ids)}, Val patients: {len(val_ids)}")

    aug = BraTSAugmentation()
    train_ds = BraTSDataset(data_dir, train_ids, transform=aug,
                            slice_size=slice_size)
    val_ds = BraTSDataset(data_dir, val_ids, transform=None,
                          slice_size=slice_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    return {'train': train_dl, 'val': val_dl}
