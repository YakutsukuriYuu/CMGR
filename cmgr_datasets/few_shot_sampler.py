"""Few-shot sampler for CMGR.

Creates N-way K-shot episodes for few-shot class-incremental learning.
Each episode contains K support samples per class and query samples
for evaluation.
"""

import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class FewShotSampler(Sampler):
    """Creates N-way K-shot episodes for few-shot learning.

    Each episode samples N classes, then K samples per class for support,
    and optionally Q samples per class for query.

    Args:
        dataset: Dataset with (point_cloud, label) pairs.
        n_way: Number of classes per episode (N).
        k_shot: Number of support samples per class (K).
        q_query: Number of query samples per class (Q).
        num_episodes: Total number of episodes to generate.
        seed: Random seed for reproducibility.
    """

    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15,
                 num_episodes=100, seed=42):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.num_episodes = num_episodes
        self.seed = seed

        # Group samples by class
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())

    def __len__(self):
        return self.num_episodes

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        for _ in range(self.num_episodes):
            # Select N classes
            selected_classes = rng.choice(
                self.classes, size=self.n_way, replace=False
            )

            episode_indices = []
            for cls in selected_classes:
                cls_indices = self.class_to_indices[cls]
                # Sample K + Q samples
                total_needed = self.k_shot + self.q_query
                if len(cls_indices) >= total_needed:
                    selected = rng.choice(cls_indices, size=total_needed, replace=False)
                else:
                    # If not enough samples, sample with replacement
                    selected = rng.choice(cls_indices, size=total_needed, replace=True)
                episode_indices.extend(selected)

            yield episode_indices

    def get_episode_dataset(self, dataset, episode_indices):
        """Create a subset dataset from episode indices.

        Args:
            dataset: Full dataset.
            episode_indices: List of indices for the episode.

        Returns:
            torch.utils.data.Subset
        """
        from torch.utils.data import Subset
        return Subset(dataset, episode_indices)


class IncrementalTaskSampler:
    """Samples data for incremental tasks.

    Splits classes into base and incremental tasks,
    providing appropriate data loaders for each.

    Args:
        dataset: Full dataset.
        base_classes: List of base class IDs.
        inc_classes_per_task: List of lists, each containing class IDs for that task.
        exemplars_per_class: Number of exemplars to keep per base class.
        batch_size: Batch size for data loading.
        num_workers: Number of data loading workers.
    """

    def __init__(self, dataset, base_classes, inc_classes_per_task,
                 exemplars_per_class=1, batch_size=8, num_workers=4):
        self.dataset = dataset
        self.base_classes = base_classes
        self.inc_classes_per_task = inc_classes_per_task
        self.exemplars_per_class = exemplars_per_class
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Group by class
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            self.class_to_indices[label].append(idx)

    def get_base_indices(self):
        """Get indices for base class training data."""
        indices = []
        for cls in self.base_classes:
            indices.extend(self.class_to_indices[cls])
        return indices

    def get_task_indices(self, task_id):
        """Get indices for an incremental task.

        Args:
            task_id: Task index (0-based, where 0 is the first incremental task).

        Returns:
            List of indices.
        """
        if task_id < len(self.inc_classes_per_task):
            classes = self.inc_classes_per_task[task_id]
            indices = []
            for cls in classes:
                indices.extend(self.class_to_indices[cls])
            return indices
        return []

    def get_all_seen_classes(self, up_to_task):
        """Get all classes seen up to a given task.

        Args:
            up_to_task: Task index (inclusive).

        Returns:
            List of class IDs.
        """
        all_classes = list(self.base_classes)
        for t in range(up_to_task + 1):
            if t < len(self.inc_classes_per_task):
                all_classes.extend(self.inc_classes_per_task[t])
        return all_classes

    def get_test_indices(self, up_to_task):
        """Get test indices for all seen classes up to a given task.

        Args:
            up_to_task: Task index (inclusive).

        Returns:
            List of indices.
        """
        all_classes = self.get_all_seen_classes(up_to_task)
        indices = []
        for cls in all_classes:
            indices.extend(self.class_to_indices[cls])
        return indices
