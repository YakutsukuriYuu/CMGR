"""Exemplar sampling utilities for CMGR.

Provides random and herding-based exemplar selection for storing
a small memory buffer of base class samples during incremental learning.
"""

import torch
import numpy as np
from collections import defaultdict


class ExemplarSampler:
    """Manages exemplar selection and storage for class-incremental learning.

    Supports two selection strategies:
    - 'random': randomly pick exemplars per class
    - 'herding': pick exemplars closest to the class mean feature (requires features)
    """

    def __init__(self, exemplars_per_class=1, strategy='random', seed=42):
        self.exemplars_per_class = exemplars_per_class
        self.strategy = strategy
        self.seed = seed
        self.exemplars = {}  # class_id -> list of (index, point_cloud, label)

    def select_random(self, dataset, class_id, num_exemplars, samples=None):
        """Randomly select exemplars for a given class.

        Args:
            dataset: Dataset object with samples.
            class_id: Target class ID.
            num_exemplars: Number of exemplars to select.
            samples: Optional pre-built list of (path, label) for fast lookup.

        Returns:
            List of selected sample indices.
        """
        if samples is None:
            samples = list(dataset)
        class_indices = [i for i, (_, label) in enumerate(samples) if label == class_id]

        rng = np.random.RandomState(self.seed + class_id)
        selected = rng.choice(class_indices, size=min(num_exemplars, len(class_indices)),
                              replace=False)
        return selected.tolist()

    def select_herding(self, dataset, class_id, num_exemplars, features, samples=None):
        """Select exemplars using herding (closest to class mean).

        Args:
            dataset: Dataset object with samples.
            class_id: Target class ID.
            num_exemplars: Number of exemplars to select.
            features: Feature matrix [N, D] for all samples.
            samples: Optional pre-built list of (path, label) for fast lookup.

        Returns:
            List of selected sample indices.
        """
        if samples is None:
            samples = list(dataset)
        class_indices = [i for i, (_, label) in enumerate(samples) if label == class_id]

        class_features = features[class_indices]
        class_mean = class_features.mean(dim=0, keepdim=True)

        # Greedy selection: iteratively pick the sample that brings the
        # exemplar mean closest to the class mean
        selected = []
        remaining = set(range(len(class_indices)))

        for _ in range(min(num_exemplars, len(class_indices))):
            best_idx = None
            best_dist = float('inf')

            for idx in remaining:
                trial = selected + [idx]
                trial_features = class_features[trial]
                trial_mean = trial_features.mean(dim=0, keepdim=True)
                dist = torch.norm(trial_mean - class_mean).item()
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        return [class_indices[i] for i in selected]

    def update_exemplars(self, dataset, class_ids, features=None, samples=None):
        """Update exemplar set for given classes.

        Args:
            dataset: Dataset object.
            class_ids: List of class IDs to store exemplars for.
            features: Optional feature matrix for herding selection.
            samples: Optional pre-built list of (path, label) for fast lookup.
                     If dataset has .samples attribute, uses that automatically.
        """
        if samples is None and hasattr(dataset, 'samples'):
            samples = dataset.samples
        for class_id in class_ids:
            if self.strategy == 'herding' and features is not None:
                indices = self.select_herding(dataset, class_id,
                                              self.exemplars_per_class, features,
                                              samples=samples)
            else:
                indices = self.select_random(dataset, class_id,
                                             self.exemplars_per_class,
                                             samples=samples)
            self.exemplars[class_id] = indices

    def get_exemplar_indices(self):
        """Return all stored exemplar indices as a flat list."""
        all_indices = []
        for indices in self.exemplars.values():
            all_indices.extend(indices)
        return all_indices

    def get_exemplar_dataset(self, dataset):
        """Return a Subset dataset containing only exemplars.

        Args:
            dataset: Full dataset.

        Returns:
            torch.utils.data.Subset of exemplar samples.
        """
        from torch.utils.data import Subset
        indices = self.get_exemplar_indices()
        return Subset(dataset, indices)

    def has_class(self, class_id):
        """Check if exemplars exist for a given class."""
        return class_id in self.exemplars

    def get_class_exemplar_indices(self, class_id):
        """Return exemplar indices for a specific class."""
        return self.exemplars.get(class_id, [])
