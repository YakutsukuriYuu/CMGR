"""ScanObjectNN dataset for CMGR.

ScanObjectNN is a real-world 3D object recognition dataset
derived from ScanNet. Used as the test set for cross-domain
M2O and S2O experiments.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ScanObjectNNDataset(Dataset):
    """ScanObjectNN dataset for 3D FSCIL.

    ScanObjectNN has 15 categories in its hardest variant (OBJ_BG),
    11 in the standard variant, and 40 in the full variant.

    Args:
        root: Root directory containing .h5 files.
        split: 'train' or 'test'.
        num_points: Number of points (2048).
        variant: 'OBJ_BG' (15 cls), 'OBJ_ONLY' (15 cls), or 'PB_T50_RS' (15 cls).
        classes: List of class indices to include.
        transform: Optional transform.
        seed: Random seed.
    """

    # ScanObjectNN class names (15 classes in OBJ_BG)
    CLASS_NAMES = [
        'bag', 'bin', 'box', 'cabinet', 'chair',
        'desk', 'display', 'door', 'shelf', 'table',
        'bed', 'pillow', 'sink', 'sofa', 'toilet',
    ]

    def __init__(self, root, split='test', num_points=2048,
                 variant='OBJ_BG', classes=None, transform=None, seed=42):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.variant = variant
        self.transform = transform
        self.seed = seed

        if classes is not None:
            self.classes = classes
        else:
            self.classes = list(range(len(self.CLASS_NAMES)))

        self.class_to_idx = {i: i for i in self.classes}

        self.data = None
        self.labels = None

        # Try to load HDF5 data
        h5_path = os.path.join(root, f'{variant}_{split}.h5')
        if os.path.exists(h5_path):
            self._load_h5(h5_path)
        else:
            print(f"[ScanObjectNNDataset] Warning: {h5_path} not found.")
            print("[ScanObjectNNDataset] Using synthetic data for testing.")

    def _load_h5(self, h5_path):
        """Load data from HDF5 file."""
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                self.data = f['data'][:]  # [N, num_points, 3]
                self.labels = f['label'][:]  # [N]

            # Filter by selected classes
            if len(self.classes) < len(self.CLASS_NAMES):
                mask = np.isin(self.labels, self.classes)
                self.data = self.data[mask]
                self.labels = self.labels[mask]

                # Remap labels
                label_map = {old: new for new, old in enumerate(self.classes)}
                self.labels = np.array([label_map[l] for l in self.labels])

            print(f"[ScanObjectNNDataset] Loaded {len(self.data)} samples from {h5_path}")
        except ImportError:
            print("[ScanObjectNNDataset] h5py not installed. Using synthetic data.")

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        return 100  # Default for synthetic data

    def __getitem__(self, idx):
        """Get a sample.

        Returns:
            point_cloud: [2048, 3] tensor.
            label: Class index.
        """
        if self.data is None:
            return self._get_synthetic_sample(idx)

        points = self.data[idx].astype(np.float32)
        label = int(self.labels[idx])

        # Resample if needed
        points = self._resample(points, self.num_points)

        point_cloud = torch.from_numpy(points).float()
        return point_cloud, label

    def _get_synthetic_sample(self, idx):
        """Generate synthetic data for testing."""
        rng = np.random.RandomState(self.seed + idx)
        points = rng.randn(self.num_points, 3).astype(np.float32)
        label = idx % len(self.classes)
        return torch.from_numpy(points), label

    def _resample(self, points, n):
        """Resample to n points."""
        if points.shape[0] == n:
            return points
        elif points.shape[0] > n:
            indices = np.random.choice(points.shape[0], n, replace=False)
            return points[indices]
        else:
            indices = np.random.choice(points.shape[0], n, replace=True)
            return points[indices]

    def get_class_name(self, class_idx):
        """Get class name from index."""
        if 0 <= class_idx < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_idx]
        return f"class_{class_idx}"

    def get_num_classes(self):
        return len(self.classes)
