"""Evaluation metrics for 3D Few-Shot Class-Incremental Learning.

Implements:
- Micro-accuracy: standard classification accuracy on all seen classes
- AA (Average Accuracy): mean accuracy across all incremental tasks
- Delta A (Averaged Relative Performance Degradation): relative forgetting
- AF (Average Forgetting): absolute forgetting measure
"""

import torch
import numpy as np
from collections import defaultdict


def compute_accuracy(predictions, targets):
    """Compute micro-accuracy (top-1).

    Args:
        predictions: Tensor [N] of predicted class indices.
        targets: Tensor [N] of ground truth class indices.

    Returns:
        float: accuracy in [0, 1].
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    correct = (predictions == targets).sum()
    total = len(targets)
    return float(correct) / total if total > 0 else 0.0


def compute_per_class_accuracy(predictions, targets, num_classes):
    """Compute per-class accuracy.

    Args:
        predictions: Tensor [N] of predicted class indices.
        targets: Tensor [N] of ground truth class indices.
        num_classes: Total number of classes.

    Returns:
        dict: class_id -> accuracy for each class that appears in targets.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    per_class = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            per_class[c] = float((predictions[mask] == c).sum()) / mask.sum()
    return per_class


def compute_AA(accuracies):
    """Compute Average Accuracy across all tasks.

    Args:
        accuracies: List of accuracy values, one per task [Acc_0, Acc_1, ..., Acc_{T-1}].

    Returns:
        float: AA = (1/T) * sum(Acc_t)
    """
    if len(accuracies) == 0:
        return 0.0
    return float(np.mean(accuracies))


def compute_delta_A(accuracies):
    """Compute Averaged Relative Performance Degradation (Delta A).

    Delta A = (1/(T-1)) * sum_{t=0}^{T-2} |Acc_t - Acc_{t+1}| / Acc_t

    Lower is better (less forgetting).

    Args:
        accuracies: List of accuracy values [Acc_0, Acc_1, ..., Acc_{T-1}].

    Returns:
        float: Delta A value.
    """
    if len(accuracies) < 2:
        return 0.0
    T = len(accuracies)
    delta_sum = 0.0
    for t in range(T - 1):
        if accuracies[t] > 0:
            delta_sum += abs(accuracies[t] - accuracies[t + 1]) / accuracies[t]
    return delta_sum / (T - 1)


def compute_AF(task_accuracies):
    """Compute Average Forgetting.

    AF = (1/(T-1)) * sum_{t=1}^{T-1} (Acc_t_best - Acc_t)

    Where Acc_t_best is the best accuracy ever achieved for task t's classes.

    Args:
        task_accuracies: Dict mapping task_id -> list of accuracies over time.
            task_accuracies[t] = [acc_at_task_t, acc_at_task_t+1, ...]
            where acc_at_task_t is accuracy on task t's test set evaluated
            after training task t.

    Returns:
        float: AF value.
    """
    if len(task_accuracies) < 2:
        return 0.0

    forgetting_sum = 0.0
    count = 0

    for task_id in sorted(task_accuracies.keys()):
        if task_id == 0:
            continue  # Skip base task
        accs = task_accuracies[task_id]
        if len(accs) < 2:
            continue
        best_acc = max(accs[:-1])  # Best before current evaluation
        current_acc = accs[-1]
        forgetting_sum += (best_acc - current_acc)
        count += 1

    return forgetting_sum / max(count, 1)


class MetricsTracker:
    """Tracks metrics across incremental tasks for CMGR evaluation."""

    def __init__(self):
        self.task_accuracies = []  # List of (task_id, accuracy)
        self.per_task_class_acc = defaultdict(list)  # task_id -> [acc over time]
        self.task_predictions = {}  # task_id -> (predictions, targets)

    def record_task(self, task_id, accuracy, predictions=None, targets=None):
        """Record accuracy for a completed task.

        Args:
            task_id: Task identifier (0 for base, 1+ for incremental).
            accuracy: Micro-accuracy on all seen classes.
            predictions: Optional prediction tensor.
            targets: Optional target tensor.
        """
        self.task_accuracies.append((task_id, accuracy))
        self.per_task_class_acc[task_id].append(accuracy)
        if predictions is not None and targets is not None:
            self.task_predictions[task_id] = (predictions, targets)

    def get_accuracy_curve(self):
        """Return list of accuracies in task order."""
        return [acc for _, acc in self.task_accuracies]

    def compute_AA(self):
        """Compute Average Accuracy."""
        accs = self.get_accuracy_curve()
        return compute_AA(accs)

    def compute_delta_A(self):
        """Compute Delta A."""
        accs = self.get_accuracy_curve()
        return compute_delta_A(accs)

    def compute_AF(self):
        """Compute Average Forgetting."""
        return compute_AF(self.per_task_class_acc)

    def summary(self):
        """Return a summary dict of all metrics."""
        accs = self.get_accuracy_curve()
        return {
            'accuracies': accs,
            'final_accuracy': accs[-1] if accs else 0.0,
            'AA': self.compute_AA(),
            'delta_A': self.compute_delta_A(),
            'AF': self.compute_AF(),
        }
