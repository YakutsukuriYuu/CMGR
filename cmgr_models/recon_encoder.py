"""ReCon 3D encoder wrapper for CMGR.

Wraps the actual ReCon PointTransformer model as a frozen backbone.
"""

import sys
import os
import importlib
import importlib.util
import torch
import torch.nn as nn
from easydict import EasyDict

# ReCon root path
RECON_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'deps', 'ReCon'))

# Add ReCon's parent dirs to sys.path for its internal imports (utils, etc.)
if RECON_ROOT not in sys.path:
    sys.path.insert(0, RECON_ROOT)


def _load_recon_transformer():
    """Load ReCon's transformer module without triggering models/__init__.py.

    ReCon's models/__init__.py imports ReCon.py which needs the chamfer CUDA extension.
    We only need transformer.py, so we load it directly.
    """
    models_dir = os.path.join(RECON_ROOT, 'models')
    init_path = os.path.join(models_dir, '__init__.py')

    # Temporarily replace ReCon's models/__init__.py with an empty one
    backup_path = init_path + '.bak'
    has_backup = False
    if os.path.exists(init_path):
        os.rename(init_path, backup_path)
        has_backup = True

    try:
        # Write a minimal __init__.py that only imports what we need
        with open(init_path, 'w') as f:
            f.write("# Minimal init for CMGR import\n")

        # Now import transformer
        import models.transformer as mod
        return mod
    finally:
        # Restore original __init__.py
        if has_backup:
            os.rename(backup_path, init_path)
        # Clean up cached module
        for key in list(sys.modules.keys()):
            if key.startswith('models'):
                del sys.modules[key]


_recon_mod = None


def _get_PointTransformer():
    global _recon_mod
    if _recon_mod is None:
        _recon_mod = _load_recon_transformer()
    return _recon_mod.PointTransformer


class ReConEncoder(nn.Module):
    """Frozen ReCon 3D encoder for CMGR.

    Args:
        trans_dim: Transformer hidden dim (384 for ReCon).
        depth: Number of transformer layers (12).
        num_heads: Number of attention heads (6).
        group_size: Points per group (32).
        num_group: Number of groups via FPS (64).
        encoder_dims: Group encoder output dim (384).
        sagr_layers: Layer indices to extract intermediate features.
        ckpt_path: Path to pretrained ReCon checkpoint.
        freeze: Whether to freeze all parameters.
    """

    def __init__(self, trans_dim=384, depth=12, num_heads=6,
                 group_size=32, num_group=64, encoder_dims=384,
                 sagr_layers=None, ckpt_path=None, freeze=True):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.sagr_layers = sagr_layers if sagr_layers is not None else [0, 4, 8]

        PointTransformer = _get_PointTransformer()

        config = EasyDict({
            'type': 'linear',
            'trans_dim': trans_dim,
            'depth': depth,
            'drop_path_rate': 0.1,
            'cls_dim': 512,
            'num_heads': num_heads,
            'group_size': group_size,
            'num_group': num_group,
            'encoder_dims': encoder_dims,
        })

        self.backbone = PointTransformer(config)

        if ckpt_path and os.path.exists(ckpt_path):
            self._load_pretrained(ckpt_path)

        self._intermediate_features = {}
        self._register_hooks()

        if freeze:
            self._freeze()

    def _load_pretrained(self, ckpt_path):
        self.backbone.load_model_from_ckpt(ckpt_path, log=True)

    def _register_hooks(self):
        for layer_idx in self.sagr_layers:
            self.backbone.blocks.blocks[layer_idx].register_forward_hook(
                self._make_hook(f'layer_{layer_idx}')
            )

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            self._intermediate_features[name] = output
        return hook_fn

    def _freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        print("[ReConEncoder] All parameters frozen.")

    def train(self, mode=True):
        return super().train(False)

    def forward(self, point_clouds):
        """Extract features from point clouds.

        Returns:
            final_features: [B, 1536] (cls + img + text + gap, each 384-dim).
            intermediate_features: list of [B, 67, 384] tensors at SAGR layers.
        """
        self._intermediate_features = {}

        neighborhood, center = self.backbone.group_divider(point_clouds)
        group_input_tokens = self.backbone.encoder(neighborhood)

        B = point_clouds.shape[0]
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        img_tokens = self.backbone.img_token.expand(B, -1, -1)
        text_tokens = self.backbone.text_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, img_tokens, text_tokens, group_input_tokens), dim=1)

        pos_group = self.backbone.pos_embed(center)
        pos = torch.cat([
            self.backbone.cls_pos.expand(B, -1, -1),
            self.backbone.img_pos.expand(B, -1, -1),
            self.backbone.text_pos.expand(B, -1, -1),
            pos_group
        ], dim=1)

        for block in self.backbone.blocks.blocks:
            x = block(x + pos)

        x = self.backbone.norm(x)

        cls_token = x[:, 0]
        img_token = x[:, 1]
        text_token = x[:, 2]
        point_tokens = x[:, 3:]
        gap = point_tokens.mean(dim=1)

        final_features = torch.cat([cls_token, img_token, text_token, gap], dim=-1)

        intermediate_features = []
        for layer_idx in self.sagr_layers:
            key = f'layer_{layer_idx}'
            if key in self._intermediate_features:
                intermediate_features.append(self._intermediate_features[key])

        return final_features, intermediate_features
