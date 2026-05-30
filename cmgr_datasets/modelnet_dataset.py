"""ModelNet dataset for CMGR.

ModelNet is a synthetic 3D model dataset used for training in
cross-domain M2O and in-domain M2M experiments.

Supports both ModelNet-10 and ModelNet-40 variants.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

try:
    import open3d as o3d
except ImportError:
    o3d = None


class ModelNetDataset(Dataset):
    """ModelNet dataset for 3D Few-Shot Class-Incremental Learning.

    Args:
        root: Root directory of ModelNet data.
        split: 'train' or 'test'.
        num_points: Number of points to sample (2048).
        variant: 'modelnet10' or 'modelnet40'.
        classes: List of class names to include. If None, use all.
        transform: Optional transform to apply to point clouds.
        augment: Whether to apply data augmentation.
        seed: Random seed.
    """

    # ModelNet-40 class names
    MODELNET40_CLASSES = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
        'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk',
        'door', 'dresser', 'flower_pot', 'glass_box', 'guitar',
        'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
        'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet',
        'tv_stand', 'vase', 'wardrobe', 'xbox',
    ]

    # ModelNet-10 class names
    MODELNET10_CLASSES = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser',
        'monitor', 'night_stand', 'sofa', 'table', 'toilet',
    ]

    def __init__(self, root, split='train', num_points=2048, variant='modelnet40',
                 classes=None, transform=None, augment=False, seed=42):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.variant = variant
        self.transform = transform
        self.augment = augment
        self.seed = seed

        # Select class list
        if variant == 'modelnet10':
            self.all_classes = self.MODELNET10_CLASSES
        else:
            self.all_classes = self.MODELNET40_CLASSES

        if classes is not None:
            self.all_classes = [c for c in self.all_classes if c in classes]

        # Build class mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.all_classes)}

        # Scan for samples
        self.samples = []
        if os.path.exists(root):
            self._scan_directory()
        else:
            print(f"[ModelNetDataset] Warning: Root directory {root} not found.")
            print("[ModelNetDataset] Using synthetic data for testing.")

    def _scan_directory(self):
        """Scan directory for model files."""
        for cls_name in self.all_classes:
            cls_dir = os.path.join(self.root, cls_name, self.split)
            if not os.path.exists(cls_dir):
                cls_dir = os.path.join(self.root, cls_name)

            if not os.path.exists(cls_dir):
                continue

            label = self.class_to_idx[cls_name]
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.off') or fname.endswith('.ply') or \
                   fname.endswith('.npy'):
                    self.samples.append((os.path.join(cls_dir, fname), label))

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
        points = self._load_points(file_path)

        # Resample to num_points
        points = self._resample(points, self.num_points)
        points = self._normalize(points)

        # Augmentation
        if self.augment:
            points = self._augment(points)

        point_cloud = torch.from_numpy(points).float()
        return point_cloud, label

    def _get_synthetic_sample(self, idx):
        """Generate synthetic data for testing."""
        rng = np.random.RandomState(self.seed + idx)
        points = rng.randn(self.num_points, 3).astype(np.float32)
        label = idx % len(self.all_classes)
        return torch.from_numpy(points), label

    def _load_points(self, file_path):
        """Load point cloud from file."""
        if file_path.endswith('.npy'):
            return np.load(file_path).astype(np.float32)
        elif file_path.endswith('.off'):
            return self._load_off_uniform(file_path, self.num_points)
        elif file_path.endswith('.ply'):
            return self._load_ply(file_path)
        return np.random.randn(self.num_points, 3).astype(np.float32)

    def _resample(self, points, n):
        """Resample point cloud to exactly n points."""
        if points.shape[0] == n:
            return points
        elif points.shape[0] > n:
            indices = np.random.choice(points.shape[0], n, replace=False)
            return points[indices]
        else:
            indices = np.random.choice(points.shape[0], n, replace=True)
            return points[indices]

    def _augment(self, points):
        """Random scale + translation augmentation."""
        scale = np.random.uniform(0.8, 1.2)
        points = points * scale
        translation = np.random.uniform(-0.1, 0.1, size=(1, 3))
        return points + translation

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
        if 0 <= class_idx < len(self.all_classes):
            return self.all_classes[class_idx]
        return f"class_{class_idx}"

    def get_num_classes(self):
        """Get number of classes."""
        return len(self.all_classes)

    @staticmethod
    def _load_off_uniform(file_path, num_points):
        """Load OFF mesh and uniformly sample surface points."""
        if o3d is not None:
            try:
                mesh = o3d.io.read_triangle_mesh(file_path)
                if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
                    point_cloud = mesh.sample_points_uniformly(num_points)
                    points = np.asarray(point_cloud.points, dtype=np.float32)
                    if points.shape[0] > 0:
                        return points
            except Exception:
                pass
        return ModelNetDataset._load_off(file_path)

    @staticmethod
    def _load_off(file_path):
        """Load OFF file."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if lines[0].strip() == 'OFF':
                    lines = lines[1:]
                parts = lines[0].split()
                n_verts = int(parts[0])
                verts = []
                for i in range(1, n_verts + 1):
                    verts.append([float(x) for x in lines[i].split()[:3]])
                return np.array(verts, dtype=np.float32)
        except Exception:
            return np.random.randn(2048, 3).astype(np.float32)

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
