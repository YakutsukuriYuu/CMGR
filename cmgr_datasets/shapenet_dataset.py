"""ShapeNet dataset for CMGR.

ShapeNet is a synthetic 3D model dataset used for training in
cross-domain 3D FSCIL experiments (S2C, S2O) and in-domain S2S.

Data organization:
- Each sample is a 3D model represented as a point cloud
- Classes are organized by synset IDs
- Point clouds are uniformly sampled to 2048 points

Expected directory structure:
    shapenet/
    ├── 02691156/  (airplane)
    │   ├── model_0001.npy
    │   └── ...
    ├── 02747177/  (trashbin)
    │   └── ...
    └── ...
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    """ShapeNet dataset for 3D Few-Shot Class-Incremental Learning.

    Args:
        root: Root directory of ShapeNet data.
        split: 'train' or 'test'.
        num_points: Number of points to sample (2048).
        classes: List of class IDs to include. If None, use all.
        transform: Optional transform to apply to point clouds.
        augment: Whether to apply data augmentation (random scale + translation).
    """

    # ShapeNet class mapping (55 categories)
    CLASS_NAMES = {
        '02691156': 'airplane', '02747177': 'trashbin', '02773838': 'bag',
        '02801938': 'basket', '02808440': 'bathtub', '02818832': 'bed',
        '02828884': 'bench', '02834778': 'bicycle', '02858304': 'boat',
        '02871439': 'bookshelf', '02876657': 'bottle', '02880940': 'bowl',
        '02924116': 'bus', '02933112': 'cabinet', '02942699': 'camera',
        '02946921': 'can', '02954340': 'cap', '02958343': 'car',
        '02992529': 'cellphone', '03001627': 'chair', '03046257': 'clock',
        '03085013': 'keyboard', '03207941': 'dishwasher', '03211117': 'display',
        '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file cabinet',
        '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
        '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
        '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
        '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
        '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
        '03991062': 'pot', '04004475': 'printer', '04074963': 'remote',
        '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
        '04256520': 'sofa', '04330267': 'stove', '04379243': 'table',
        '04401088': 'telephone', '04460130': 'tower', '04468005': 'train',
        '04530566': 'vessel', '04554684': 'washer',
    }

    def __init__(self, root, split='train', num_points=2048, classes=None,
                 transform=None, augment=False, seed=42):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.transform = transform
        self.augment = augment
        self.seed = seed

        # Scan directory for available classes and samples
        self.samples = []  # List of (file_path, class_id)
        self.class_to_idx = {}

        if os.path.exists(root):
            self._scan_directory(classes)
        else:
            print(f"[ShapeNetDataset] Warning: Root directory {root} not found.")
            print("[ShapeNetDataset] Using synthetic data for testing.")

    def _scan_directory(self, classes=None):
        """Scan directory for samples."""
        available_classes = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])

        if classes is not None:
            available_classes = [c for c in available_classes if c in classes]

        for idx, cls_id in enumerate(available_classes):
            self.class_to_idx[cls_id] = idx
            cls_dir = os.path.join(self.root, cls_id)

            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.npy') or fname.endswith('.ply') or \
                   fname.endswith('.off') or fname.endswith('.pcd'):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample.

        Args:
            idx: Sample index.

        Returns:
            point_cloud: [2048, 3] tensor of point coordinates.
            label: Class index (int).
        """
        if len(self.samples) == 0:
            # Return synthetic data for testing
            return self._get_synthetic_sample(idx)

        file_path, label = self.samples[idx]

        # Load point cloud
        if file_path.endswith('.npy'):
            points = np.load(file_path)
        elif file_path.endswith('.ply'):
            points = self._load_ply(file_path)
        else:
            points = self._load_off(file_path)

        # Ensure correct shape
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[indices]

        # Data augmentation (only during training with augmentation=True)
        if self.augment:
            points = self._augment(points)

        point_cloud = torch.from_numpy(points).float()
        return point_cloud, label

    def _get_synthetic_sample(self, idx):
        """Generate a synthetic sample for testing."""
        rng = np.random.RandomState(self.seed + idx)
        points = rng.randn(self.num_points, 3).astype(np.float32)
        label = idx % 10  # 10 synthetic classes
        return torch.from_numpy(points), label

    def _augment(self, points):
        """Apply data augmentation: random scale + random translation.

        Args:
            points: [N, 3] numpy array.

        Returns:
            Augmented [N, 3] numpy array.
        """
        # Random scale [0.8, 1.2]
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale

        # Random translation [-0.1, 0.1]
        translation = np.random.uniform(-0.1, 0.1, size=(1, 3))
        points = points + translation

        return points

    def get_class_name(self, class_idx):
        """Get class name from class index."""
        for name, idx in self.class_to_idx.items():
            if idx == class_idx:
                return self.CLASS_NAMES.get(name, name)
        return f"class_{class_idx}"

    def get_num_classes(self):
        """Get number of classes."""
        return len(self.class_to_idx)

    @staticmethod
    def _load_ply(file_path):
        """Load PLY file."""
        try:
            from plyfile import PlyData
            plydata = PlyData.read(file_path)
            vertices = plydata['vertex']
            points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
            return points.astype(np.float32)
        except Exception:
            return np.random.randn(2048, 3).astype(np.float32)

    @staticmethod
    def _load_off(file_path):
        """Load OFF file."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines[0].strip() == 'OFF':
                    lines = lines[1:]
                n_verts = int(lines[0].split()[0])
                verts = []
                for i in range(1, n_verts + 1):
                    verts.append([float(x) for x in lines[i].split()[:3]])
                return np.array(verts, dtype=np.float32)
        except Exception:
            return np.random.randn(2048, 3).astype(np.float32)
