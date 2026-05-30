"""CMGR: Cross-Modal Geometric Rectification framework.

Classification: cosine_similarity(F_hat, F_T) * 100 + CLIP(I_E, d)
Loss: L = L_cls + alpha * L_mc + beta * L_c + gamma * L_kd
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmgr_models.recon_encoder import ReConEncoder
from cmgr_models.depth_encoder import DepthEncoder
from cmgr_models.clip_wrapper import CLIPWrapper
from cmgr_models.sagr import SAGR
from cmgr_models.tam import TAM
from cmgr_models.bnd import BND


class CMGR(nn.Module):

    def __init__(self, config, device='cuda'):
        super().__init__()
        self.config = config
        self.device = device
        self._incremental_mode = False

        self.recon_token_dim = 384
        self.recon_feat_dim = 1536
        self.depth_feat_dim = 512
        self.clip_intermediate_dim = 768

        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 1.0)
        self.gamma = config.get('gamma', 10.0)
        self.use_clip_logits_during_training = config.get(
            'use_clip_logits_during_training', False
        )
        self.clip_logit_weight = config.get('clip_logit_weight', 1.0)

        # Teacher features for knowledge distillation (incremental training)
        self.register_buffer('_teacher_features', None, persistent=False)
        self._num_base_classes = 0

        # Text feature cache (avoids re-encoding same class names every batch)
        self._text_cache_key = None
        self._text_cache_features = None

        # 1. Frozen 3D encoder (ReCon)
        self.recon_encoder = ReConEncoder(
            trans_dim=self.recon_token_dim,
            depth=config.get('recon_num_layers', 12),
            num_heads=6,
            group_size=32,
            num_group=64,
            encoder_dims=384,
            sagr_layers=config.get('sagr_layers', [0, 4, 8]),
            ckpt_path=config.get('recon_ckpt_path', None),
            freeze=True,
        )

        # 2. Depth encoder (CLIP2Point)
        self.depth_encoder = DepthEncoder(
            num_views=config.get('num_views', 10),
            sagr_layers=config.get('sagr_layers', [0, 4, 8]),
            ckpt_path=config.get('depth_ckpt_path', None),
        )

        # 3. Frozen CLIP ViT-B/32
        self.clip_wrapper = CLIPWrapper(
            model_name='ViT-B/32',
            device=device,
        )

        # 4. SAGR
        self.sagr = SAGR(
            feat_dim_3d=self.recon_token_dim,
            feat_dim_2d=self.clip_intermediate_dim,
            output_dim=self.depth_feat_dim,
            sagr_layers=config.get('sagr_layers', [0, 4, 8]),
            mask_ratio=config.get('mask_ratio', 0.9),
            num_sa_layers=config.get('num_sa_layers', 2),
        )

        # 5. TAM
        self.tam = TAM(
            feat_dim=self.recon_token_dim,
            white_threshold=0.9,
            kernel_size=9,
        )

        # 6. BND
        self.bnd = BND(
            feat_dim=self.recon_feat_dim,
            hidden_dim=256,
        )

        self.to(device)

    def set_incremental_mode(self):
        """Incremental mode: freeze depth encoder, only train SAGR + TAM.

        NOTE: Paper says '部分训练' for depth encoder, but unfreezing visual.proj
        (393K params) causes gradient interference with SAGR (3M params).
        Keeping depth encoder fully frozen for stability.
        """
        self._incremental_mode = True
        for param in self.depth_encoder.parameters():
            param.requires_grad = False
        self.depth_encoder.eval()
        print("[CMGR] Incremental mode: depth encoder frozen, only SAGR + TAM trainable.")

    @torch.no_grad()
    def store_teacher_features(self, dataloader, class_names, num_base_classes):
        """Store teacher F_hat_pooled features per base class for KD.

        Computes prototype (class-mean) F_hat_pooled for each base class
        using the current frozen model. These prototypes serve as distillation
        targets during incremental training to prevent forgetting.
        """
        self._num_base_classes = num_base_classes
        self.eval()

        # Accumulate F_hat_pooled per base class
        sum_features = torch.zeros(num_base_classes, self.depth_feat_dim, device=self.device)
        counts = torch.zeros(num_base_classes, device=self.device)

        for point_clouds, labels in dataloader:
            point_clouds = point_clouds.to(self.device)
            labels = labels.to(self.device)

            recon_final, recon_intermediates = self.recon_encoder(point_clouds)

            if recon_intermediates:
                point_tokens = recon_intermediates[-1][:, 3:, :]
                recon_pooled = point_tokens.mean(dim=1)
            else:
                recon_pooled = recon_final[:, :384]

            depth_maps, depth_final, depth_intermediates = self.depth_encoder(point_clouds)
            B, V, C, H, W = depth_maps.shape

            enhanced_maps, _ = self.tam(depth_maps, recon_pooled)
            sagr_features, _ = self.sagr(
                recon_intermediates, depth_intermediates,
                recon_final, depth_final,
            )
            F_hat = self.sagr.aggregate_views(
                F_P=recon_final,
                F_U=sagr_features,
                F_D=depth_final,
            )
            F_hat_pooled = F_hat.reshape(B, V, -1).mean(dim=1)

            for c in range(num_base_classes):
                mask = (labels == c)
                if mask.any():
                    sum_features[c] += F_hat_pooled[mask].sum(dim=0)
                    counts[c] += mask.sum()

        # Compute class-mean prototypes
        prototypes = torch.zeros(num_base_classes, self.depth_feat_dim, device=self.device)
        for c in range(num_base_classes):
            if counts[c] > 0:
                prototypes[c] = sum_features[c] / counts[c]

        self._teacher_features = prototypes  # [num_base, 512]
        self.train()
        print(f"[CMGR] Stored teacher features for {num_base_classes} base classes "
              f"(KD weight gamma={self.gamma})")

    def get_trainable_params(self):
        """Get trainable parameters.

        Base training: SAGR + TAM + full depth encoder
        Incremental training: SAGR + TAM + depth encoder proj layer
        """
        params = []
        params.extend(self.sagr.parameters())
        params.extend(self.tam.parameters())
        if not self._incremental_mode:
            params.extend(self.depth_encoder.get_trainable_params())
        return params

    def get_bnd_params(self):
        return list(self.bnd.parameters())

    def forward(self, point_clouds, class_names=None, labels=None):
        losses = {}

        # Step 1: 3D encoding (frozen)
        recon_final, recon_intermediates = self.recon_encoder(point_clouds)

        # Point token avg-pool for TAM
        if recon_intermediates:
            point_tokens = recon_intermediates[-1][:, 3:, :]
            recon_pooled = point_tokens.mean(dim=1)
        else:
            recon_pooled = recon_final[:, :384]

        # Step 2: Depth encoding + rendering
        depth_maps, depth_final, depth_intermediates = self.depth_encoder(point_clouds)
        B, V, C, H, W = depth_maps.shape

        # Step 3: TAM enhancement
        enhanced_maps, learned_color = self.tam(depth_maps, recon_pooled)

        # Step 4: SAGR rectification
        sagr_features, mc_loss = self.sagr(
            recon_intermediates, depth_intermediates,
            recon_final, depth_final,
        )

        if self.training:
            losses['mc_loss'] = mc_loss

        # Step 5: Cross-view aggregation
        F_hat = self.sagr.aggregate_views(
            F_P=recon_final,
            F_U=sagr_features,
            F_D=depth_final,
        )
        F_hat_pooled = F_hat.reshape(B, V, -1).mean(dim=1)  # [B, 512]
        self._last_F_hat_pooled = F_hat_pooled  # for KD loss in compute_loss

        # Step 6: Text features (cached: skip re-encoding if class_names unchanged)
        text_features = None
        if class_names is not None:
            cache_key = tuple(class_names)
            if self._text_cache_key != cache_key:
                self._text_cache_features = self.clip_wrapper.encode_text(class_names)
                self._text_cache_key = cache_key
            text_features = self._text_cache_features

        clip_img_features_train = None
        if self.training and text_features is not None and labels is not None:
            enhanced_flat = enhanced_maps.reshape(B * V, 3, H, W)  # [B*V, 3, H, W]
            clip_img_features_train = self.clip_wrapper.encode_image_with_grad(
                enhanced_flat
            )  # [B*V, 512]

        # Step 7: Classification via cosine similarity (Eq.7)
        # logits = F_hat @ F_T^T * 100 + CLIP(I_E, d)
        if text_features is not None:
            geo_sim = F.cosine_similarity(
                F_hat_pooled.unsqueeze(1),
                text_features.unsqueeze(0), dim=-1,
            ) * 100.0  # [B, C]

            if self.training:
                if (self.use_clip_logits_during_training and
                        clip_img_features_train is not None):
                    clip_sim = self.clip_wrapper.compute_similarity(
                        clip_img_features_train, text_features
                    )
                    clip_sim_pooled = clip_sim.reshape(B, V, -1).mean(dim=1)
                    logits = geo_sim + self.clip_logit_weight * clip_sim_pooled
                else:
                    logits = geo_sim
            else:
                enhanced_flat = enhanced_maps.reshape(B * V, 3, H, W)
                with torch.no_grad():
                    clip_img_features = self.clip_wrapper.encode_image(enhanced_flat)
                    clip_sim = self.clip_wrapper.compute_similarity(
                        clip_img_features, text_features
                    )
                    clip_sim_pooled = clip_sim.reshape(B, V, -1).mean(dim=1)
                logits = geo_sim + clip_sim_pooled
        else:
            logits = F_hat_pooled

        # Step 8: Color alignment loss (batched: all B*V images in one forward pass)
        if (self.training and text_features is not None and labels is not None and
                clip_img_features_train is not None):
            gt_text_features = text_features[labels]  # [B, 512]
            feat_all = clip_img_features_train.reshape(B, V, -1)  # [B, V, 512]
            # cosine similarity per view: [B, V]
            cos_sim = F.cosine_similarity(
                feat_all, gt_text_features.unsqueeze(1).expand_as(feat_all), dim=-1
            )
            losses['color_loss'] = ((1.0 - cos_sim) / 2.0).mean()

        return logits, losses

    def compute_loss(self, logits, labels, losses):
        cls_loss = F.cross_entropy(logits, labels)
        mc_loss = losses.get('mc_loss', torch.tensor(0.0, device=logits.device))
        color_loss = losses.get('color_loss', torch.tensor(0.0, device=logits.device))

        # Knowledge distillation: cosine distance penalty for base-class F_hat drift
        kd_loss = torch.tensor(0.0, device=logits.device)
        if self._incremental_mode and self._teacher_features is not None and self._num_base_classes > 0:
            base_mask = labels < self._num_base_classes
            if base_mask.any():
                current_feats = self._last_F_hat_pooled[base_mask]  # [n_base, 512]
                teacher_feats = self._teacher_features[labels[base_mask]]  # [n_base, 512]
                cos_sim = F.cosine_similarity(current_feats, teacher_feats, dim=-1)
                kd_loss = ((1.0 - cos_sim) / 2.0).mean()  # bounded in [0, 1]

        total_loss = cls_loss + self.alpha * mc_loss + self.beta * color_loss + self.gamma * kd_loss

        return total_loss, {
            'total_loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'mc_loss': mc_loss.item() if torch.is_tensor(mc_loss) else mc_loss,
            'color_loss': color_loss.item() if torch.is_tensor(color_loss) else color_loss,
            'kd_loss': kd_loss.item() if torch.is_tensor(kd_loss) else kd_loss,
        }

    def store_base_netB(self):
        """Store a snapshot of current SAGR+TAM state as the frozen base NetB.

        Call this BEFORE starting incremental training. During inference,
        BND routes base-class samples through this frozen NetB.
        """
        self._base_sagr_sd = self._clone_state_dict(self.sagr.state_dict())
        self._base_tam_sd = self._clone_state_dict(self.tam.state_dict())
        self._base_depth_sd = self._clone_state_dict(self.depth_encoder.state_dict())
        print("[CMGR] Stored base NetB snapshot for BND routing.")

    @staticmethod
    def _clone_state_dict(state_dict):
        """Clone a state dict so later load_state_dict calls cannot mutate it."""
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }

    def _forward_with_weights(self, point_clouds, class_names,
                               sagr_sd, tam_sd, depth_sd):
        """Run forward pass with specific SAGR+TAM+depth weights (no grad)."""
        cur_sagr = self._clone_state_dict(self.sagr.state_dict())
        cur_tam = self._clone_state_dict(self.tam.state_dict())
        cur_depth = self._clone_state_dict(self.depth_encoder.state_dict())

        try:
            self.sagr.load_state_dict(sagr_sd)
            self.tam.load_state_dict(tam_sd)
            self.depth_encoder.load_state_dict(depth_sd)

            logits, _ = self.forward(point_clouds, class_names)
            return logits
        finally:
            self.sagr.load_state_dict(cur_sagr)
            self.tam.load_state_dict(cur_tam)
            self.depth_encoder.load_state_dict(cur_depth)

    @torch.no_grad()
    def inference(self, point_clouds, class_names, base_class_names, threshold=0.1):
        """Inference with BND routing: base → NetB, novel → incremental network.

        Args:
            point_clouds: [B, N, 3] input point clouds.
            class_names: List of ALL seen class names (base + novel).
            base_class_names: List of base class names only.
            threshold: BND decision threshold (default: 0.1).

        Returns:
            predictions: [B] predicted class indices (into class_names).
            is_base: [B] boolean mask of BND base predictions.
        """
        recon_final, _ = self.recon_encoder(point_clouds)
        is_base, _ = self.bnd.predict(recon_final, threshold)

        predictions = torch.zeros(point_clouds.shape[0], dtype=torch.long,
                                  device=point_clouds.device)

        if is_base.any() and hasattr(self, '_base_sagr_sd'):
            # Base samples → frozen NetB (only knows base classes)
            base_idx = is_base.nonzero(as_tuple=True)[0]
            base_logits = self._forward_with_weights(
                point_clouds[base_idx], base_class_names,
                self._base_sagr_sd, self._base_tam_sd, self._base_depth_sd,
            )
            base_pred_local = base_logits.argmax(dim=-1)
            base_to_full = torch.tensor(
                [class_names.index(name) for name in base_class_names],
                dtype=torch.long,
                device=point_clouds.device,
            )
            predictions[base_idx] = base_to_full[base_pred_local]

        if (~is_base).any():
            # Novel samples → incremental network (all seen classes)
            novel_idx = (~is_base).nonzero(as_tuple=True)[0]
            novel_logits, _ = self.forward(point_clouds[novel_idx], class_names)
            predictions[novel_idx] = novel_logits.argmax(dim=-1)

        return predictions, is_base

    def save_netB(self, path):
        checkpoint = {
            'sagr': self.sagr.state_dict(),
            'tam': self.tam.state_dict(),
            'depth_encoder': self.depth_encoder.state_dict(),
            'bnd': self.bnd.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
        print(f"[CMGR] Saved NetB to {path}")

    def load_netB(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.sagr.load_state_dict(checkpoint['sagr'])
        self.tam.load_state_dict(checkpoint['tam'])
        self.depth_encoder.load_state_dict(checkpoint['depth_encoder'])
        if 'bnd' in checkpoint:
            self.bnd.load_state_dict(checkpoint['bnd'])
        print(f"[CMGR] Loaded NetB from {path}")
