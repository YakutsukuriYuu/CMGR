"""CLIP2Point depth encoder wrapper for CMGR.

Wraps CLIP2Point's rendering pipeline + CLIP ViT-B/32 visual encoder.
- Renders depth maps from point clouds (needed for TAM)
- Extracts intermediate ViT features at layers {0, 4, 8} (768-dim) for SAGR
- Produces final 512-dim features (after CLIP's projection)
"""

import sys
import os
import importlib
import torch
import torch.nn as nn

# CLIP2Point root path
CLIP2POINT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'deps', 'CLIP2Point')


def _import_clip2point_module(module_name):
    """Import a module from CLIP2Point without permanently polluting sys.path."""
    if CLIP2POINT_ROOT not in sys.path:
        sys.path.insert(0, CLIP2POINT_ROOT)
    try:
        mod = importlib.import_module(module_name)
        return mod
    finally:
        if CLIP2POINT_ROOT in sys.path:
            sys.path.remove(CLIP2POINT_ROOT)


class DepthEncoder(nn.Module):
    """Trainable depth encoder wrapping CLIP2Point.

    Uses CLIP2Point's Selector + Renderer for depth map rendering,
    and CLIP ViT-B/32 visual encoder for feature extraction.

    Args:
        num_views: Number of rendering views (12 for V100 16GB).
        sagr_layers: Layer indices for intermediate feature extraction.
        ckpt_path: Path to CLIP2Point pretrained checkpoint.
    """

    def __init__(self, num_views=10, sagr_layers=None, ckpt_path=None):
        super().__init__()
        self.num_views = num_views
        self.sagr_layers = sagr_layers if sagr_layers is not None else [0, 4, 8]

        # Import CLIP2Point rendering components
        render_mod = _import_clip2point_module('render')
        Selector = render_mod.Selector
        Renderer = render_mod.Renderer

        # View selector and depth renderer
        # shape_features_size=0 uses ViewSelector with fixed angles (no learned selection)
        self.selector = Selector(num_views, shape_features_size=0)
        self.renderer = Renderer(points_radius=0.02)

        # CLIP ViT-B/32 visual encoder (trainable, like in CLIP2Point)
        import clip
        clip_model, _ = clip.load("ViT-B/32", device='cpu')
        self.visual = clip_model.visual

        # Load CLIP2Point pretrained weights if provided
        if ckpt_path and os.path.exists(ckpt_path):
            self._load_pretrained(ckpt_path)

        # Register hooks for intermediate feature extraction
        self._intermediate_features = {}
        self._register_hooks()

    def _load_pretrained(self, ckpt_path):
        """Load pretrained CLIP2Point weights into the visual encoder."""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'pre_model' in checkpoint:
                    state_dict = checkpoint['pre_model']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # Filter for visual encoder keys
            visual_dict = {}
            for k, v in state_dict.items():
                if k.startswith('point_model.'):
                    new_key = k[len('point_model.'):]
                    visual_dict[new_key] = v
                elif k.startswith('visual.'):
                    visual_dict[k] = v
                elif not any(k.startswith(p) for p in
                        ('image_model.', 'selector.', 'renderer.', 'criterion.', 'weights')):
                    visual_dict[k] = v

            model_dict = self.visual.state_dict()
            compatible = {k: v for k, v in visual_dict.items()
                          if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compatible)
            self.visual.load_state_dict(model_dict)
            print(f"[DepthEncoder] Loaded {len(compatible)}/{len(model_dict)} "
                  f"parameters from {ckpt_path}")
        except Exception as e:
            print(f"[DepthEncoder] Warning: Could not load pretrained weights: {e}")

    def _register_hooks(self):
        """Register forward hooks on CLIP ViT transformer blocks."""
        for layer_idx in self.sagr_layers:
            self.visual.transformer.resblocks[layer_idx].register_forward_hook(
                self._make_hook(f'layer_{layer_idx}')
            )

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # output: [seq_len, B*V, 768] (CLIP uses seq-first)
            self._intermediate_features[name] = output
        return hook_fn

    def render_depth_maps(self, point_clouds):
        """Render multi-view depth maps from point clouds.

        Args:
            point_clouds: [B, N, 3] point cloud coordinates.

        Returns:
            depth_maps: [B, V, C, H, W] rendered depth maps (224x224).
        """
        with torch.no_grad():
            # Selector expects a points tensor, not an integer
            azim, elev, dist = self.selector(point_clouds)
            azim = azim.to(point_clouds.device)
            elev = elev.to(point_clouds.device)
            dist = dist.to(point_clouds.device)
            depth_maps = self.renderer(
                points=point_clouds, azim=azim, elev=elev,
                dist=dist, view=self.num_views
            )
        return depth_maps

    def forward(self, point_clouds):
        """Extract depth features from point clouds.

        Args:
            point_clouds: [B, N, 3] point cloud coordinates.

        Returns:
            depth_maps: [B, V, C, H, W] rendered depth maps (for TAM).
            final_features: [B*V, 512] per-view features (after CLIP projection).
            intermediate_features: list of [seq_len, B*V, 768] tensors at SAGR layers.
        """
        self._intermediate_features = {}

        # Render depth maps
        depth_maps = self.render_depth_maps(point_clouds)  # [B, V, C, H, W]
        B, V, C, H, W = depth_maps.shape

        # Process through CLIP visual encoder
        imgs = depth_maps.reshape(B * V, C, H, W)

        # CLIP visual forward
        x = imgs.type(self.visual.conv1.weight.dtype)
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([
            self.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # [seq_len, B*V, width]

        # Transformer blocks (hooks capture intermediates)
        for block in self.visual.transformer.resblocks:
            x = block(x)

        x = x.permute(1, 0, 2)  # [B*V, seq_len, width]
        x = self.visual.ln_post(x[:, 0, :])  # CLS token → [B*V, 768]

        if self.visual.proj is not None:
            x = x @ self.visual.proj  # [B*V, 512]

        final_features = x

        intermediate_features = []
        for layer_idx in self.sagr_layers:
            key = f'layer_{layer_idx}'
            if key in self._intermediate_features:
                intermediate_features.append(self._intermediate_features[key])

        return depth_maps, final_features, intermediate_features

    def get_trainable_params(self):
        """Return trainable parameters (visual encoder)."""
        return self.visual.parameters()
