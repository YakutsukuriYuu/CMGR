"""CO3D dataset for CMGR.

CO3D (Common Objects in 3D) is a real-world 3D object dataset
captured with handheld cameras. Used as the test set for
cross-domain S2C experiments.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class CO3DDataset(Dataset):
    """CO3D dataset for 3D Few-Shot Class-Incremental Learning.

    Args:
        root: Root directory of CO3D data.
        split: 'train' or 'test'.
        num_points: Number of points to sample (2048).
        classes: List of class names to include.
        transform: Optional transform.
        seed: Random seed.
    """

    # CO3D has 50+ object categories
    CATEGORIES = [
        'apple', 'backpack', 'banana', 'baseballbat', 'baseballglove',
        'bench', 'bicycle', 'bottle', 'bowl', 'broccoli',
        'cake', 'car', 'carrot', 'cellphone', 'chair',
        'cup', 'donut', 'hairdryer', 'handbag', 'hydrant',
        'keyboard', 'kite', 'laptop', 'microwave', 'motorcycle',
        'mouse', 'orange', 'parkingmeter', 'pizza', 'plant',
        'remote', 'sandwich', 'scissors', 'skateboard', 'stopsign',
        'suitcase', 'surfboard', 'teddybear', 'toaster', 'toilet',
        'toybus', 'toyplane', 'toytrain', 'toytruck', 'umbrella',
        'vase', 'wineglass', 'zebra',
    ]

    def __init__(self, root, split='test', num_points=2048, classes=None,
                 transform=None, seed=42):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.seed = seed

        if classes is not None:
            self.categories = classes
        else:
            self.categories = self.CATEGORIES

        self.class_to_idx = {name: idx for idx, name in enumerate(self.categories)}

        self.samples = []
        if os.path.exists(root):
            self._scan_directory()
        else:
            print(f"[CO3DDataset] Warning: Root directory {root} not found.")
            print("[CO3DDataset] Using synthetic data for testing.")

    def _scan_directory(self):
        """Scan CO3D directory structure.

        Expected structure:
            root/
            ├── category_1/
        │   ├── frame_00001.pointcloud.npy
        │   └── ...
        └── ...
        """
        for cat_name in self.categories:
            cat_dir = os.path.join(self.root, cat_name)
            if not os.path.exists(cat_dir):
                continue

            label = self.class_to_idx[cat_name]

            # CO3D organizes by sequences
            for seq_dir in sorted(os.listdir(cat_dir)):
                seq_path = os.path.join(cat_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue

                for fname in sorted(os.listdir(seq_path)):
                    if fname.endswith('.pointcloud.npy') or fname.endswith('.npy'):
                        self.samples.append((os.path.join(seq_path, fname), label))
                        break  # One point cloud per sequence

    def __len__(self):
        return max(len(self.samples), 1)

    def __getitem__(self, idx):
        """Get a sample.

        Returns:
            point_cloud: [2048, 3] tensor.
            label: Class index.
        """
        if len(self.samples) == 0:
            return self._get_synthetic_sample(idx)

        file_path, label = self.samples[idx]
        points = np.load(file_path).astype(np.float32)

        # Resample
        points = self._resample(points, self.num_points)

        point_cloud = torch.from_numpy(points).float()
        return point_cloud, label

    def _get_synthetic_sample(self, idx):
        """Generate synthetic data for testing."""
        rng = np.random.RandomState(self.seed + idx)
        points = rng.randn(self.num_points, 3).astype(np.float32)
        label = idx % len(self.categories)
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
        if 0 <= class_idx < len(self.categories):
            return self.categories[class_idx]
        return f"class_{class_idx}"

    def get_num_classes(self):
        return len(self.categories)
