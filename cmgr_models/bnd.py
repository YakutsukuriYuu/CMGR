"""BND: Base-Novel Discriminator for CMGR.

BND is a simple binary classifier that determines whether a sample
belongs to a base class or a novel class. This enables routing
between the base network (NetB) and the current incremental network.

Architecture (from the paper):
- fc1: feat_dim -> 256, ReLU
- fc2: 256 -> 1
- Trained with BCEWithLogitsLoss
- Labels: base classes = 1, novel classes = 0
- Inference: if logit > threshold (0.1) -> base class -> use NetB
             else -> novel class -> use current network

Training configuration:
- Optimizer: Adam, lr=1e-3
- Loss: BCEWithLogitsLoss
- Epochs: 10
- Input: avg-pooled point cloud features F_P
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BND(nn.Module):
    """Base-Novel Discriminator.

    A simple 2-layer FC classifier that discriminates between base and
    novel classes using point cloud features.

    Args:
        feat_dim: Input feature dimension (256 for ReCon features).
        hidden_dim: Hidden layer dimension (256).
    """

    def __init__(self, feat_dim=1536, hidden_dim=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # 2-layer FC classifier
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        """Forward pass.

        Args:
            features: [B, feat_dim] point cloud features (avg-pooled).

        Returns:
            logits: [B, 1] raw logits (before sigmoid).
                   Apply sigmoid to get probabilities.
        """
        h = F.relu(self.fc1(features))
        logit = self.fc2(h)
        return logit

    def predict(self, features, threshold=0.1):
        """Predict base/novel class membership.

        Args:
            features: [B, feat_dim] point cloud features.
            threshold: Raw-logit decision threshold (default: 0.1).

        Returns:
            is_base: [B] boolean tensor (True = base class, False = novel class).
            logits: [B, 1] raw logits.
        """
        logits = self.forward(features)
        is_base = logits > threshold  # [B, 1]
        return is_base.squeeze(-1), logits

    def classify_with_routing(self, features, netB_logits, net_logits, threshold=0.1):
        """Route classification through base or novel network.

        If BND predicts base class (logit > threshold), use NetB's prediction.
        Otherwise, use the current network's prediction.

        Args:
            features: [B, feat_dim] point cloud features for BND.
            netB_logits: [B, C] classification logits from base network NetB.
            net_logits: [B, C] classification logits from current network.
            threshold: BND decision threshold.

        Returns:
            routed_logits: [B, C] routed classification logits.
            is_base: [B] boolean mask of base class predictions.
        """
        is_base, _ = self.predict(features, threshold)

        # Route: base -> NetB, novel -> current net
        routed_logits = torch.where(
            is_base.unsqueeze(-1).expand_as(netB_logits),
            netB_logits,
            net_logits,
        )

        return routed_logits, is_base


class BNDTrainer:
    """Trainer for the BND module.

    Handles the training loop for BND, which is trained before
    each incremental task to discriminate base and novel classes.

    Args:
        bnd_model: BND model instance.
        lr: Learning rate (default: 1e-3).
        epochs: Training epochs (default: 10).
        device: Training device.
    """

    def __init__(self, bnd_model, lr=1e-3, epochs=10, device='cuda',
                 threshold=0.1):
        self.bnd_model = bnd_model
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss()
        self.last_stats = {}

    def _balanced_training_set(self, base_features, novel_features):
        """Build a balanced BND batch by oversampling the smaller side."""
        n_base = base_features.shape[0]
        n_novel = novel_features.shape[0]
        if n_base == 0 or n_novel == 0:
            raise ValueError(
                f"BND needs both base and novel features, got "
                f"base={n_base}, novel={n_novel}"
            )

        target = max(n_base, n_novel)
        if n_base == target:
            base_balanced = base_features
        else:
            base_idx = torch.randint(n_base, (target,), device=self.device)
            base_balanced = base_features[base_idx]

        if n_novel == target:
            novel_balanced = novel_features
        else:
            novel_idx = torch.randint(n_novel, (target,), device=self.device)
            novel_balanced = novel_features[novel_idx]

        features = torch.cat([base_balanced, novel_balanced], dim=0)
        labels = torch.cat([
            torch.ones(target, 1, device=self.device),
            torch.zeros(target, 1, device=self.device),
        ], dim=0)

        perm = torch.randperm(features.shape[0], device=self.device)
        return features[perm], labels[perm]

    def train(self, base_features, novel_features):
        """Train BND to discriminate base and novel classes.

        Args:
            base_features: [N_base, D] features from base class exemplars.
            novel_features: [N_novel, D] features from novel class samples.

        Returns:
            avg_loss: Average training loss.
        """
        self.bnd_model.train()
        self.bnd_model.to(self.device)

        # Prepare data
        base_features = base_features.to(self.device)
        novel_features = novel_features.to(self.device)

        features, labels = self._balanced_training_set(base_features, novel_features)

        # Optimizer
        optimizer = torch.optim.Adam(self.bnd_model.parameters(), lr=self.lr)

        # Training loop
        total_loss = 0.0
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = self.bnd_model(features)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / self.epochs
        self.bnd_model.eval()
        with torch.no_grad():
            base_logits = self.bnd_model(base_features)
            novel_logits = self.bnd_model(novel_features)
            base_pred = base_logits > self.threshold
            novel_pred = novel_logits > self.threshold
            self.last_stats = {
                'base_logit_mean': base_logits.mean().item(),
                'novel_logit_mean': novel_logits.mean().item(),
                'base_route_rate': base_pred.float().mean().item(),
                'novel_route_rate': novel_pred.float().mean().item(),
                'base_count': int(base_features.shape[0]),
                'novel_count': int(novel_features.shape[0]),
            }
        return avg_loss
