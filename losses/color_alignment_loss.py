"""Color alignment loss L_c for TAM module.

Encourages the enhanced image features (with learned background color)
to align with the text features of the corresponding class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorAlignmentLoss(nn.Module):
    """Color alignment loss L_c.

    L_c = mean((1 - cosine_similarity(F_E, F_T)) / 2)

    where:
    - F_E: enhanced image features [V, D]
    - F_T: text features [D] (broadcast to all views)

    This loss encourages the enhanced depth-rendered images
    (with TAM's learned background color) to have features
    that are semantically aligned with the class text description.
    """

    def __init__(self):
        super().__init__()

    def forward(self, F_E, F_T):
        """Compute color alignment loss.

        Args:
            F_E: Enhanced image features [V, D] from CLIP.
            F_T: Text features [D] from CLIP text encoder.

        Returns:
            Scalar loss tensor.
        """
        # Ensure F_T is [D]
        if F_T.dim() > 1:
            F_T = F_T.squeeze()

        # Cosine similarity between each enhanced view feature and text feature
        # [V, D] x [D] -> [V]
        cos_sim = F.cosine_similarity(F_E, F_T.unsqueeze(0).expand_as(F_E), dim=-1)

        # L_c = mean((1 - cos_sim) / 2)
        loss = ((1.0 - cos_sim) / 2.0).mean()
        return loss
