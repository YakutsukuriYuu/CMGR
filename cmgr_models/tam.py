"""TAM: Texture Amplification Module for CMGR.

TAM enhances depth-rendered images by:
1. Generating an adaptive background color from point cloud features
2. Detecting pure background regions in the depth maps
3. Filling the learned color into background regions
4. Computing color alignment loss between enhanced images and text features

Architecture (from the paper):
- ColorGenerator: 2-layer MLP, input=avg-pooled F_P (256-dim), output=3 RGB channels
  - W1: 256->256, ReLU, W2: 256->3, tanh, then (raw+1)/2 for [0,1] range
- Background detection: 9x9 convolution to detect pure white regions
- Color alignment loss: L_c = mean((1 - cosine_similarity(F_E, F_T)) / 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAM(nn.Module):
    """Texture Amplification Module.

    Enhances depth-rendered images by filling background regions with
    a learned adaptive color that is semantically meaningful.

    Args:
        feat_dim: Input feature dimension (256 for ReCon features).
        white_threshold: Threshold for detecting white background pixels.
        kernel_size: Size of the convolution kernel for background detection.
    """

    def __init__(self, feat_dim=384, white_threshold=0.9, kernel_size=9):
        super().__init__()
        self.feat_dim = feat_dim
        self.white_threshold = white_threshold
        self.kernel_size = kernel_size

        # Color generator: 2-layer MLP
        self.color_generator = ColorGenerator(feat_dim=feat_dim)

        # Background detection kernel (9x9 average filter)
        # Fixed kernel, not trainable
        self.register_buffer(
            'bg_kernel',
            torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        )
        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3),
        )
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3),
        )

    def detect_background(self, depth_maps):
        """Detect background regions in depth maps.

        Background is defined as:
        - White pixels (depth value >= white_threshold)
        - Pure background regions (all pixels in kernel are white)

        Args:
            depth_maps: [B, V, 1, H, W] depth maps.
                        Background is white (high values), foreground is dark.

        Returns:
            bg_mask: [B, V, 1, H, W] binary mask (1=background, 0=foreground).
        """
        B, V, C, H, W = depth_maps.shape

        # Reshape for batch processing: [B*V, C, H, W]
        depth_flat = depth_maps.reshape(B * V, C, H, W)

        # Detect white pixels: average across channels then threshold
        # Works for both 1-channel and 3-channel depth maps
        depth_gray = depth_flat.mean(dim=1, keepdim=True)  # [B*V, 1, H, W]
        M_w = (depth_gray >= self.white_threshold).float()

        # Use convolution to detect pure background regions
        conv_result = F.conv2d(M_w, self.bg_kernel, padding=self.kernel_size // 2)

        # Pure background: all neighbors are white AND the pixel itself is white
        M_b = ((conv_result >= 0.99) & (M_w >= 0.99)).float()

        # Reshape back: [B, V, 1, H, W]
        bg_mask = M_b.reshape(B, V, 1, H, W)

        return bg_mask

    def enhance_depth_maps(self, depth_maps, point_features):
        """Enhance depth maps with learned background color.

        Args:
            depth_maps: [B, V, C, H, W] depth maps (C=1 or C=3).
            point_features: [B, D] point cloud features (avg-pooled).

        Returns:
            enhanced_maps: [B, V, 3, H, W] enhanced RGB maps.
            color: [B, 3] learned background color.
        """
        B, V, C, H, W = depth_maps.shape

        # Generate adaptive color from point cloud features
        color = self.color_generator(point_features)  # raw RGB in [0, 1], [B, 3]
        color_normalized = (color - self.clip_mean) / self.clip_std

        # Detect background regions
        bg_mask = self.detect_background(depth_maps)  # [B, V, 1, H, W]

        # Convert depth maps to 3-channel if needed
        if C == 1:
            depth_rgb = depth_maps.expand(-1, -1, 3, -1, -1)  # [B, V, 3, H, W]
        else:
            depth_rgb = depth_maps  # Already 3-channel

        # Fill background with learned color
        # color: [B, 3] -> [B, 1, 3, 1, 1] for broadcasting
        color_broadcast = color_normalized.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        color_map = color_broadcast.expand(B, V, 3, H, W)

        # Apply: where background, use learned color; otherwise keep depth
        enhanced_maps = depth_rgb * (1 - bg_mask) + color_map * bg_mask

        return enhanced_maps, color

    def forward(self, depth_maps, point_features):
        """Forward pass.

        Args:
            depth_maps: [B, V, 1, H, W] depth maps from renderer.
            point_features: [B, D] avg-pooled point cloud features.

        Returns:
            enhanced_maps: [B, V, 3, H, W] enhanced RGB maps.
            color: [B, 3] learned background color.
        """
        return self.enhance_depth_maps(depth_maps, point_features)


class ColorGenerator(nn.Module):
    """Adaptive color generator for TAM.

    Generates a 3-channel RGB color from point cloud features.
    Uses a 2-layer MLP with tanh activation, scaled to [0, 1].

    Formula (Equation 6):
        hidden = ReLU(W1 * F_P_avgpooled + b1)
        raw = tanh(W2 * hidden + b2)
        color = (raw + 1) / 2  # Scale to [0, 1]

    Args:
        feat_dim: Input feature dimension (256).
        hidden_dim: Hidden layer dimension (256).
    """

    def __init__(self, feat_dim=256, hidden_dim=256):
        super().__init__()
        self.W1 = nn.Linear(feat_dim, hidden_dim)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Linear(hidden_dim, 3)
        self.b2 = nn.Parameter(torch.zeros(3))

    def forward(self, F_P_avgpooled):
        """Generate RGB color from point cloud features.

        Args:
            F_P_avgpooled: [B, feat_dim] avg-pooled point cloud features.

        Returns:
            color: [B, 3] RGB color in [0, 1] range.
        """
        hidden = F.relu(self.W1(F_P_avgpooled) + self.b1)
        raw = torch.tanh(self.W2(hidden) + self.b2)  # [-1, 1]
        color = (raw + 1.0) / 2.0  # [0, 1]
        return color
