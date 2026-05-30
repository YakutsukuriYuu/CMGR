"""CMGR dataset modules.

Provides dataset classes for 3D Few-Shot Class-Incremental Learning:
- ShapeNetDataset: ShapeNet synthetic 3D models
- ModelNetDataset: ModelNet synthetic 3D models
- CO3DDataset: CO3D real-world 3D scans
- ScanObjectNNDataset: ScanObjectNN real-world 3D scans
- FewShotSampler: Creates N-way K-shot episodes for few-shot learning
"""

from .shapenet_dataset import ShapeNetDataset
from .modelnet_dataset import ModelNetDataset
from .co3d_dataset import CO3DDataset
from .scanobjectnn_dataset import ScanObjectNNDataset
from .few_shot_sampler import FewShotSampler
