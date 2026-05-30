"""Standard cross-entropy classification loss for CMGR."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Cross-entropy classification loss.

    Used for training the classification head on top of the fused features.
    Supports optional label smoothing.
    """

    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        """Compute cross-entropy loss.

        Args:
            logits: [B, C] classification logits.
            targets: [B] ground truth class indices.

        Returns:
            Scalar loss tensor.
        """
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
