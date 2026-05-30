"""Mask consistency loss L_mc for SAGR module.

Computes the consistency between masked and unmasked feature similarity matrices
to regularize the self-masking operation in SAGR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskConsistencyLoss(nn.Module):
    """Mask consistency loss L_mc.

    L_mc = ||sim(F_U) - sim(F_MU)||^2 / B^2

    where:
    - F_U: features after self-masking (only top attention weights kept)
    - F_MU: features without masking (all attention weights)
    - sim(): cosine similarity matrix
    - B: batch size

    This loss encourages the masked features to preserve the same
    inter-sample similarity structure as the unmasked features.
    """

    def __init__(self):
        super().__init__()

    def cosine_similarity_matrix(self, features):
        """Compute pairwise cosine similarity matrix.

        Args:
            features: [B, D] feature tensor.

        Returns:
            [B, B] cosine similarity matrix.
        """
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=-1)
        # Compute similarity matrix
        sim_matrix = torch.mm(features_norm, features_norm.t())
        return sim_matrix

    def forward(self, F_U, F_MU):
        """Compute mask consistency loss.

        Args:
            F_U: Masked features [B, D].
            F_MU: Unmasked features [B, D].

        Returns:
            Scalar loss tensor.
        """
        B = F_U.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=F_U.device, dtype=F_U.dtype)

        sim_U = self.cosine_similarity_matrix(F_U)
        sim_MU = self.cosine_similarity_matrix(F_MU)

        # L_mc = ||sim(F_U) - sim(F_MU)||^2 / B^2
        loss = torch.norm(sim_U - sim_MU, p='fro') ** 2 / (B * B)
        return loss
