"""Evaluation script for CMGR.

Computes:
- Micro-accuracy: standard accuracy on all seen classes
- AA (Average Accuracy): mean across all tasks
- Delta A: relative performance degradation
- AF (Average Forgetting): absolute forgetting measure

Uses BND routing: if BND logit > threshold -> use NetB, else -> use current net.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from cmgr_models import CMGR
from render.depth_renderer import create_depth_renderer
from cmgr_utils.metrics import compute_accuracy, compute_AA, compute_delta_A, compute_AF, MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description='CMGR Evaluation')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='shapenet',
                        choices=['shapenet', 'modelnet', 'co3d', 'scanobjectnn'],
                        help='Dataset to use')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory with trained model checkpoints')
    parser.add_argument('--recon_ckpt', type=str, default=None,
                        help='Path to ReCon checkpoint')
    parser.add_argument('--depth_ckpt', type=str, default=None,
                        help='Path to CLIP2Point checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Total number of classes')
    parser.add_argument('--use_bnd', action='store_true',
                        help='Use BND routing for evaluation')
    parser.add_argument('--bnd_threshold', type=float, default=0.1,
                        help='BND decision threshold')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataset(args, config):
    """Create evaluation dataset."""
    if args.dataset == 'shapenet':
        from datasets import ShapeNetDataset
        return ShapeNetDataset(root=args.data_root, split='test',
                               num_points=config['point_cloud_size'])
    elif args.dataset == 'modelnet':
        from datasets import ModelNetDataset
        return ModelNetDataset(root=args.data_root, split='test',
                               num_points=config['point_cloud_size'])
    elif args.dataset == 'co3d':
        from datasets import CO3DDataset
        return CO3DDataset(root=args.data_root, split='test',
                           num_points=config['point_cloud_size'])
    elif args.dataset == 'scanobjectnn':
        from datasets import ScanObjectNNDataset
        return ScanObjectNNDataset(root=args.data_root, split='test',
                                    num_points=config['point_cloud_size'])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


@torch.no_grad()
def evaluate_with_bnd(model, dataloader, renderer, device, class_names,
                      netB_model=None, threshold=0.1):
    """Evaluate with BND routing.

    If BND predicts base class (logit > threshold), use NetB's prediction.
    Otherwise, use the current network's prediction.

    Args:
        model: Current CMGR model.
        dataloader: Test data loader.
        renderer: Depth renderer.
        device: Device.
        class_names: List of class names.
        netB_model: Frozen NetB model for base class prediction.
        threshold: BND decision threshold.

    Returns:
        accuracy: Micro-accuracy.
        predictions: All predictions.
        targets: All targets.
    """
    model.eval()
    if netB_model is not None:
        netB_model.eval()

    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    for point_clouds, labels in tqdm(dataloader, desc='Evaluating with BND'):
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)
        B = point_clouds.shape[0]

        # Render depth maps
        depth_maps = renderer(point_clouds)

        # Get 3D features for BND
        recon_final, _ = model.recon_encoder(point_clouds)

        # BND prediction
        is_base, bnd_logits = model.bnd.predict(recon_final, threshold)

        # Forward pass with current model
        logits, _ = model(point_clouds, depth_maps, class_names=class_names)

        # If NetB is available and some samples are predicted as base
        if netB_model is not None and is_base.any():
            # Get NetB predictions for base-class samples
            netB_logits, _ = netB_model(point_clouds, depth_maps, class_names=class_names)

            # Route: base -> NetB, novel -> current net
            final_logits = torch.where(
                is_base.unsqueeze(-1).expand_as(logits),
                netB_logits,
                logits,
            )
        else:
            final_logits = logits

        predictions = final_logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += B

        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    return accuracy, all_predictions, all_targets


@torch.no_grad()
def evaluate_standard(model, dataloader, renderer, device, class_names):
    """Standard evaluation without BND routing.

    Args:
        model: CMGR model.
        dataloader: Test data loader.
        renderer: Depth renderer.
        device: Device.
        class_names: List of class names.

    Returns:
        accuracy: Micro-accuracy.
        predictions: All predictions.
        targets: All targets.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    for point_clouds, labels in tqdm(dataloader, desc='Evaluating'):
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)
        B = point_clouds.shape[0]

        # Render depth maps
        depth_maps = renderer(point_clouds)

        # Forward pass
        logits, _ = model(point_clouds, depth_maps, class_names=class_names)

        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += B

        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    return accuracy, all_predictions, all_targets


def evaluate_per_class(predictions, targets, num_classes, class_names=None):
    """Compute per-class accuracy.

    Args:
        predictions: [N] predicted class indices.
        targets: [N] ground truth class indices.
        num_classes: Total number of classes.
        class_names: Optional list of class names.

    Returns:
        per_class_acc: Dict mapping class_id -> accuracy.
    """
    per_class_acc = {}
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            acc = (predictions[mask] == c).float().mean().item()
            name = class_names[c] if class_names else f"class_{c}"
            per_class_acc[name] = acc
    return per_class_acc


def main():
    args = parse_args()
    config = load_config(args.config)

    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------------------------------------------------------
    # 1. Create dataset
    # ---------------------------------------------------------------
    print("Creating dataset...")
    dataset = create_dataset(args, config)

    num_classes = args.num_classes or dataset.get_num_classes()
    class_names = [dataset.get_class_name(i) for i in range(num_classes)]

    test_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    # ---------------------------------------------------------------
    # 2. Create model
    # ---------------------------------------------------------------
    print("Creating model...")
    model_config = {
        'recon_feat_dim': config.get('recon_feat_dim', 256),
        'depth_feat_dim': config.get('depth_feat_dim', 512),
        'clip_feat_dim': config.get('clip_feat_dim', 512),
        'recon_num_layers': config.get('recon_num_layers', 12),
        'depth_num_layers': config.get('depth_num_layers', 12),
        'sagr_layers': config.get('sagr_layers', [0, 4, 8]),
        'mask_ratio': config.get('mask_ratio', 0.9),
        'num_sa_layers': config.get('num_sa_layers', 2),
        'alpha': config.get('alpha', 1.0),
        'beta': config.get('beta', 1.0),
        'text_template': config.get('text_template', 'a photo of a {}'),
        'recon_ckpt_path': args.recon_ckpt,
        'depth_ckpt_path': args.depth_ckpt,
    }

    model = CMGR(model_config, device=device)
    model.init_classification_head(num_classes)

    # Load latest checkpoint
    checkpoint_path = os.path.join(args.model_dir, 'netB_final.pth')
    if os.path.exists(checkpoint_path):
        model.load_netB(checkpoint_path)
        print(f"Loaded model from {checkpoint_path}")

    # ---------------------------------------------------------------
    # 3. Create renderer
    # ---------------------------------------------------------------
    renderer = create_depth_renderer(
        num_views=config['num_views'],
        resolution=config['depth_resolution'],
    )
    renderer = renderer.to(device)

    # ---------------------------------------------------------------
    # 4. Evaluate
    # ---------------------------------------------------------------
    if args.use_bnd:
        print("Evaluating with BND routing...")
        accuracy, predictions, targets = evaluate_with_bnd(
            model, test_loader, renderer, device, class_names,
            netB_model=None,  # NetB is loaded into model
            threshold=args.bnd_threshold,
        )
    else:
        print("Evaluating (standard)...")
        accuracy, predictions, targets = evaluate_standard(
            model, test_loader, renderer, device, class_names,
        )

    # ---------------------------------------------------------------
    # 5. Compute metrics
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Micro-accuracy: {accuracy:.4f}")

    # Per-class accuracy
    per_class = evaluate_per_class(predictions, targets, num_classes, class_names)
    print(f"\nPer-class accuracy:")
    for name, acc in sorted(per_class.items()):
        print(f"  {name}: {acc:.4f}")

    # If we have incremental task results, compute AA, DeltaA, AF
    results_file = os.path.join(args.model_dir, 'results.yaml')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = yaml.safe_load(f)

        accuracies = results.get('accuracies', [])
        if accuracies:
            print(f"\nIncremental metrics:")
            print(f"  AA (Average Accuracy): {compute_AA(accuracies):.4f}")
            print(f"  Delta A: {compute_delta_A(accuracies):.4f}")
            print(f"  AF (Average Forgetting): {results.get('AF', 0.0):.4f}")

    # Save evaluation results
    eval_results = {
        'micro_accuracy': accuracy,
        'per_class_accuracy': per_class,
        'num_classes': num_classes,
        'dataset': args.dataset,
    }

    eval_path = os.path.join(args.model_dir, 'evaluation.yaml')
    with open(eval_path, 'w') as f:
        yaml.dump(eval_results, f)
    print(f"\nResults saved to {eval_path}")


if __name__ == '__main__':
    main()
