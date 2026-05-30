"""SAGR: Structure-Aware Geometric Rectification module for CMGR.

SAGR performs cross-modal feature rectification by using cross-attention
between 3D point cloud features (from ReCon, 384-dim) and 2D depth features
(from CLIP ViT-B/32, 768-dim) at selected transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGR(nn.Module):
    """Structure-Aware Geometric Rectification module.

    Args:
        feat_dim_3d: 3D feature dimension (384 for ReCon).
        feat_dim_2d: 2D feature dimension (768 for CLIP ViT-B/32 intermediate).
        output_dim: Output feature dimension after cross-view aggregation (512).
        sagr_layers: Layer indices for cross-attention.
        mask_ratio: Ratio of attention weights to mask (0.9 = keep top 10%).
        num_sa_layers: Number of self-attention layers for regularization.
        num_heads: Number of attention heads.
    """

    def __init__(self, feat_dim_3d=384, feat_dim_2d=768, output_dim=512,
                 sagr_layers=None, mask_ratio=0.9, num_sa_layers=2, num_heads=8):
        super().__init__()
        self.feat_dim_3d = feat_dim_3d
        self.feat_dim_2d = feat_dim_2d
        self.output_dim = output_dim
        self.sagr_layers = sagr_layers if sagr_layers is not None else [0, 4, 8]
        self.mask_ratio = mask_ratio
        self.num_sa_layers = num_sa_layers
        self.num_heads = num_heads

        # Cross-attention layers for each SAGR layer
        self.cross_attn_layers = nn.ModuleDict()
        for layer_idx in self.sagr_layers:
            self.cross_attn_layers[str(layer_idx)] = CrossAttentionLayer(
                dim_q=feat_dim_3d,
                dim_kv=feat_dim_2d,
                num_heads=num_heads,
            )

        # Self-attention layers for regularization
        self.sa_layers = nn.ModuleList([
            SelfAttentionLayer(feat_dim_3d, num_heads=num_heads)
            for _ in range(num_sa_layers)
        ])

        # Cross-view aggregation module
        self.cross_view_aggregation = CrossViewAggregation(
            feat_dim_3d=feat_dim_3d,
            feat_dim_2d=feat_dim_2d,
            output_dim=output_dim,
        )

    def self_masking(self, attn_weights):
        """Apply self-masking: keep top (1 - mask_ratio) attention weights."""
        keep_ratio = 1.0 - self.mask_ratio
        k = max(1, int(attn_weights.shape[-1] * keep_ratio))
        topk_vals, _ = torch.topk(attn_weights, k, dim=-1)
        threshold = topk_vals[..., -1:]
        attn_mask = (attn_weights >= threshold).float()
        attn_masked = attn_weights * attn_mask
        attn_sum = attn_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn_masked = attn_masked / attn_sum
        return attn_masked, attn_mask

    def forward(self, recon_intermediates, depth_intermediates, recon_final, depth_final):
        """Forward pass of SAGR.

        Args:
            recon_intermediates: List of [B, N_3d, 384] tensors from ReCon at SAGR layers.
            depth_intermediates: List of [seq_len, B*V, 768] tensors from CLIP at SAGR layers.
                NOTE: CLIP hooks output in seq-first format [seq, B*V, dim].
            recon_final: [B, 1536] final features from ReCon.
            depth_final: [B*V, 512] final features from depth encoder.

        Returns:
            sagr_features: [B*V, output_dim] rectified features.
            mc_loss: Scalar mask consistency loss.
        """
        mc_loss = torch.tensor(0.0, device=recon_final.device)
        B = recon_final.shape[0]
        V = depth_final.shape[0] // B

        # Process each SAGR layer
        F_P = recon_final  # [B, 1536]
        all_mc_losses = []

        for i, layer_idx in enumerate(self.sagr_layers):
            if i < len(recon_intermediates) and i < len(depth_intermediates):
                F_P_i = recon_intermediates[i]  # [B, N_3d, 384]

                # CLIP hook outputs [seq, B*V, 768] → [B*V, seq, 768]
                F_D_i = depth_intermediates[i]
                if F_D_i.dim() == 3 and F_D_i.shape[0] != B:
                    F_D_i = F_D_i.permute(1, 0, 2)  # [B*V, seq, 768]

                # Expand 3D features to match B*V: [B, N, D] → [B*V, N, D]
                F_P_i_expanded = F_P_i.unsqueeze(1).expand(-1, V, -1, -1)
                F_P_i_expanded = F_P_i_expanded.reshape(B * V, F_P_i.shape[1], -1)

                # Cross-attention: Q=F_P_i, K=F_D_i, V=F_D_i
                cross_attn = self.cross_attn_layers[str(layer_idx)]
                attn_output, attn_weights, V_proj = cross_attn(F_P_i_expanded, F_D_i)

                # Self-masking
                attn_masked, attn_mask = self.self_masking(attn_weights)

                # Apply masked attention using projected V (384-dim, not raw 768-dim)
                F_U = torch.bmm(attn_masked, V_proj[:, :attn_masked.shape[-1], :])
                F_MU = torch.bmm(attn_weights, V_proj[:, :attn_weights.shape[-1], :])

                # Mask consistency loss
                if self.training:
                    mc = self._compute_mask_consistency_loss(F_U, F_MU)
                    all_mc_losses.append(mc)

                # Update features (pool over sequence dim)
                F_P = F_U.mean(dim=1)  # [B*V, D_2d]

        # Self-attention regularization layers
        for sa_layer in self.sa_layers:
            F_P = sa_layer(F_P)

        if self.training and all_mc_losses:
            mc_loss = torch.stack(all_mc_losses).mean()

        return F_P, mc_loss

    def _compute_mask_consistency_loss(self, F_U, F_MU):
        """L_mc = ||sim(F_U) - sim(F_MU)||^2 / B^2"""
        F_U_pooled = F_U.mean(dim=1)
        F_MU_pooled = F_MU.mean(dim=1)
        B = F_U_pooled.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=F_U.device)
        sim_U = F.cosine_similarity(F_U_pooled.unsqueeze(1), F_U_pooled.unsqueeze(0), dim=-1)
        sim_MU = F.cosine_similarity(F_MU_pooled.unsqueeze(1), F_MU_pooled.unsqueeze(0), dim=-1)
        loss = torch.norm(sim_U - sim_MU, p='fro') ** 2 / (B * B)
        return loss

    def aggregate_views(self, F_P, F_U, F_D, lambda_param=1.0, w=1.0):
        """Cross-view aggregation (Equation 5)."""
        return self.cross_view_aggregation(F_P, F_U, F_D, lambda_param, w)


class CrossAttentionLayer(nn.Module):
    """Cross-attention: 3D Query, 2D Key/Value."""

    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_q // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim_q, dim_q)
        self.k_proj = nn.Linear(dim_kv, dim_q)
        self.v_proj = nn.Linear(dim_kv, dim_q)
        self.out_proj = nn.Linear(dim_q, dim_q)

    def forward(self, F_P, F_D):
        """
        Args:
            F_P: [B, N, D_q] 3D features (Query).
            F_D: [B, M, D_kv] 2D features (Key/Value).
        Returns:
            output: [B, N, D_q] attended features.
            attn_weights: [B, N, M] attention weights (averaged over heads).
            V_projected: [B, M, D_q] projected values (for masked attention).
        """
        B, N, _ = F_P.shape
        M = F_D.shape[1]

        Q = self.q_proj(F_P).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(F_D).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(F_D)  # [B, M, D_q]
        V_heads = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, V_heads)
        output = output.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(output)

        avg_attn = attn_weights.mean(dim=1)  # [B, N, M]
        return output, avg_attn, V  # V is [B, M, D_q] (projected)


class SelfAttentionLayer(nn.Module):
    """Self-attention for feature refinement."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True
        h = self.norm(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        if squeeze:
            x = x.squeeze(1)
        return x


class CrossViewAggregation(nn.Module):
    """Cross-view aggregation (Equation 5):
    concat(F_P, F_U) -> linear_f -> + w * F_D -> * lambda -> linear_g -> F_hat
    """

    def __init__(self, feat_dim_3d=384, feat_dim_2d=768, output_dim=512):
        super().__init__()
        # F_P and F_U are both 1536-dim (from ReCon: cls+img+text+gap)
        # But after SAGR processing, F_U becomes feat_dim_2d (768)
        # We project both to a common dim for aggregation
        self.proj_3d = nn.Linear(1536, feat_dim_3d)  # Project ReCon final features
        self.linear_f = nn.Linear(feat_dim_3d * 2, feat_dim_3d)
        self.proj_depth = nn.Linear(512, feat_dim_3d)  # Project depth final features (512-dim)
        self.linear_g = nn.Linear(feat_dim_3d, output_dim)

    def forward(self, F_P, F_U, F_D, lambda_param=1.0, w=1.0):
        """
        Args:
            F_P: [B, 1536] original ReCon features.
            F_U: [B*V, D_2d] rectified features from SAGR (768-dim).
            F_D: [B*V, 512] depth encoder final features.
        Returns:
            F_hat: [B*V, output_dim] aggregated features.
        """
        B_V = F_U.shape[0]
        B = F_P.shape[0]
        V = B_V // B

        # Project ReCon features: [B, 1536] → [B, feat_dim_3d]
        F_P_proj = self.proj_3d(F_P)

        # Expand F_P to match B*V: [B, D] → [B*V, D]
        F_P_expanded = F_P_proj.unsqueeze(1).expand(-1, V, -1).reshape(B_V, -1)

        # Concatenate: [B*V, 2*feat_dim_3d]
        concat_feat = torch.cat([F_P_expanded, F_U], dim=-1)

        # Linear f: [B*V, feat_dim_3d]
        intermediate = F.gelu(self.linear_f(concat_feat))

        # Project depth features and add: [B*V, feat_dim_3d]
        F_D_proj = self.proj_depth(F_D)
        result = intermediate + w * F_D_proj
        result = result * lambda_param

        # Final projection: [B*V, output_dim]
        F_hat = self.linear_g(result)
        return F_hat
