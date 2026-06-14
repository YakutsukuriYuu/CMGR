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

        self.classes = self._resolve_classes(classes)
        self.class_names = [self.CLASS_NAMES[i] for i in self.classes]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.original_label_to_idx = {
            original: idx for idx, original in enumerate(self.classes)
        }

        self.data = None
        self.labels = None
        self.samples = []

        # Try to load HDF5 data
        h5_path = self._find_h5_path()
        if os.path.exists(h5_path):
            self._load_h5(h5_path)
        else:
            print(f"[ScanObjectNNDataset] Warning: no H5 file found under {root} "
                  f"for variant={variant}, split={split}.")
            print("[ScanObjectNNDataset] Using synthetic data for testing.")
            self.samples = [
                (i, i % len(self.classes)) for i in range(100)
            ]

    def _resolve_classes(self, classes):
        """Resolve class names or original ScanObjectNN IDs to original IDs."""
        if classes is None:
            return list(range(len(self.CLASS_NAMES)))

        resolved = []
        for cls in classes:
            if isinstance(cls, str):
                if cls not in self.CLASS_NAMES:
                    raise ValueError(f"Unknown ScanObjectNN class name: {cls}")
                cls_id = self.CLASS_NAMES.index(cls)
            else:
                cls_id = int(cls)
                if cls_id < 0 or cls_id >= len(self.CLASS_NAMES):
                    raise ValueError(f"Unknown ScanObjectNN class id: {cls_id}")
            if cls_id not in resolved:
                resolved.append(cls_id)
        return resolved

    def _find_h5_path(self):
        """Find ScanObjectNN H5 file using CMGR and common upstream names."""
        split_alias = 'training' if self.split in ('train', 'training') else self.split
        variant = self.variant.lower()

        candidate_names = [
            f'{self.variant}_{self.split}.h5',
            f'{self.variant}_{split_alias}.h5',
        ]

        if variant in ('obj_bg', 'objectbg', 'scanobjectnn'):
            candidate_names.append(f'{split_alias}_objectdataset.h5')
        if variant in ('obj_only', 'objectonly'):
            candidate_names.append(f'{split_alias}_objectdataset.h5')
        if variant in ('pb_t50_rs', 'hardest', 'scanobjectnn_hardest'):
            candidate_names.append(f'{split_alias}_objectdataset_augmentedrot_scale75.h5')

        seen = set()
        for name in candidate_names:
            for base in (self.root, os.path.join(self.root, 'main_split')):
                path = os.path.join(base, name)
                if path in seen:
                    continue
                seen.add(path)
                if os.path.exists(path):
                    return path

        return os.path.join(self.root, candidate_names[0])

    def _load_h5(self, h5_path):
        """Load data from HDF5 file."""
        try:
            import h5py
            with h5py.File(h5_path, 'r') as f:
                self.data = f['data'][:]  # [N, num_points, 3]
                self.labels = f['label'][:].reshape(-1).astype(np.int64)  # [N]

            # Filter by selected classes
            if len(self.classes) < len(self.CLASS_NAMES):
                mask = np.isin(self.labels, self.classes)
                self.data = self.data[mask]
                self.labels = self.labels[mask]

                # Remap labels
                label_map = {old: new for new, old in enumerate(self.classes)}
                self.labels = np.array(
                    [label_map[int(l)] for l in self.labels], dtype=np.int64
                )

            self.samples = [
                (i, int(label)) for i, label in enumerate(self.labels)
            ]

            print(f"[ScanObjectNNDataset] Loaded {len(self.data)} samples from {h5_path}")
        except ImportError:
            print("[ScanObjectNNDataset] h5py not installed. Using synthetic data.")
            self.samples = [
                (i, i % len(self.classes)) for i in range(100)
            ]

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
        points = self._normalize(points)

        if self.transform is not None:
            points = self.transform(points)

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

    @staticmethod
    def _normalize(points):
        """Center point cloud and scale it to the unit sphere."""
        centroid = np.mean(points, axis=0, keepdims=True)
        points = points - centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale
        return points.astype(np.float32)

    def get_class_name(self, class_idx):
        """Get class name from index."""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return f"class_{class_idx}"

    def get_num_classes(self):
        return len(self.classes)
