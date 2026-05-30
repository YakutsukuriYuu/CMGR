"""Base task training for CMGR on ModelNet40.

Trains on base classes (task 0) with:
- 10 epochs, lr=5e-4, Adam optimizer, weight_decay=1e-4
- Data augmentation: random scale + random translation
- Loss: L_cls + alpha * L_mc + beta * L_c
- Trains: SAGR, TAM, depth encoder, classification head
- Freezes: ReCon 3D encoder, CLIP
- Saves: trained network as NetB, exemplar set (1 per class)
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# Ensure CMGR package is importable
sys.path.insert(0, os.path.dirname(__file__))

from cmgr_models.cmgr import CMGR
from cmgr_utils.sampler import ExemplarSampler


def parse_args():
    parser = argparse.ArgumentParser(description='CMGR Base Task Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ModelNet40 data')
    parser.add_argument('--output_dir', type=str, default='outputs/base')
    parser.add_argument('--recon_ckpt', type=str,
                        default='deps/ReCon/pretrained/recon.pth')
    parser.add_argument('--depth_ckpt', type=str,
                        default='deps/CLIP2Point/pretrained/vit32/best_eval.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_base_classes', type=int, default=20,
                        help='Number of base classes (default: 20 for ModelNet40)')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_modelnet_dataset(data_root, split, num_points, classes, augment, seed):
    """Create ModelNet dataset for given class subset."""
    from cmgr_datasets.modelnet_dataset import ModelNetDataset
    dataset = ModelNetDataset(
        root=data_root,
        split=split,
        num_points=num_points,
        variant='modelnet40',
        classes=classes,
        augment=augment,
        seed=seed,
    )
    return dataset


def train_one_epoch(model, dataloader, optimizer, device, class_names, epoch,
                    total_epochs, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_accum_steps = max(1, int(grad_accum_steps))

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}')
    optimizer.zero_grad(set_to_none=True)
    for step, (point_clouds, labels) in enumerate(pbar):
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)
        B = point_clouds.shape[0]

        logits, losses = model(point_clouds, class_names=class_names, labels=labels)
        loss, loss_dict = model.compute_loss(logits, labels, losses)

        (loss / grad_accum_steps).backward()
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * B
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += B

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{loss_dict["cls_loss"]:.4f}',
            'mc': f'{loss_dict["mc_loss"]:.4f}',
            'color': f'{loss_dict["color_loss"]:.4f}',
            'acc': f'{total_correct/total_samples:.4f}',
        })

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device, class_names):
    model.eval()
    total_correct = 0
    total_samples = 0

    for point_clouds, labels in tqdm(dataloader, desc='Evaluating'):
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)

        logits, _ = model(point_clouds, class_names=class_names)
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += point_clouds.shape[0]

    return total_correct / total_samples if total_samples > 0 else 0.0


def save_exemplars(dataset, class_names, config, output_dir):
    """Save exemplar set (1 per base class)."""
    exemplar_sampler = ExemplarSampler(
        exemplars_per_class=config.get('exemplars_per_class', 1),
        strategy='random',
        seed=config['seed'],
    )
    class_ids = list(range(len(class_names)))
    exemplar_sampler.update_exemplars(dataset, class_ids)

    exemplar_path = os.path.join(output_dir, 'exemplars.pth')
    torch.save({
        'exemplar_indices': exemplar_sampler.exemplars,
        'class_ids': class_ids,
        'class_names': class_names,
    }, exemplar_path)
    print(f"Saved exemplars to {exemplar_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Create datasets (base classes = first N classes)
    # ---------------------------------------------------------------
    all_classes = [
        'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle',
        'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk',
        'door', 'dresser', 'flower_pot', 'glass_box', 'guitar',
        'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
        'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet',
        'tv_stand', 'vase', 'wardrobe', 'xbox',
    ]
    num_base = args.num_base_classes
    base_classes = all_classes[:num_base]
    print(f"Base classes ({num_base}): {base_classes}")

    train_dataset = create_modelnet_dataset(
        args.data_root, 'train', config['point_cloud_size'],
        base_classes, augment=config.get('augmentation', True), seed=config['seed'],
    )
    val_dataset = create_modelnet_dataset(
        args.data_root, 'test', config['point_cloud_size'],
        base_classes, augment=False, seed=config['seed'],
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=True,
    )

    # ---------------------------------------------------------------
    # 2. Create model
    # ---------------------------------------------------------------
    print("Creating CMGR model...")
    model_config = {
        'num_views': config.get('num_views', 10),
        'sagr_layers': config.get('sagr_layers', [0, 4, 8]),
        'mask_ratio': config.get('mask_ratio', 0.9),
        'num_sa_layers': config.get('num_sa_layers', 2),
        'alpha': config.get('alpha', 1.0),
        'beta': config.get('beta', 1.0),
        'gamma': config.get('gamma', 10.0),
        'use_clip_logits_during_training': config.get(
            'use_clip_logits_during_training', False
        ),
        'clip_logit_weight': config.get('clip_logit_weight', 1.0),
        'text_template': config.get('text_template', 'a photo of a {}'),
        'recon_ckpt_path': args.recon_ckpt,
        'depth_ckpt_path': args.depth_ckpt,
    }

    model = CMGR(model_config, device=device)

    # ---------------------------------------------------------------
    # 3. Optimizer
    # ---------------------------------------------------------------
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=config['base_lr'],
        weight_decay=config['base_weight_decay'],
    )

    # ---------------------------------------------------------------
    # 4. Training loop
    # ---------------------------------------------------------------
    print(f"\nStarting base training for up to {config['base_epochs']} epochs...")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['base_lr']}")
    print(f"  Num views: {config.get('num_views', 10)}")
    grad_accum_steps = max(1, int(config.get('grad_accum_steps', 1)))
    if grad_accum_steps > 1:
        print(f"  Gradient accumulation: {grad_accum_steps} "
              f"(effective batch size: {config['batch_size'] * grad_accum_steps})")
    print()

    early_stop_enabled = config.get('early_stopping', False)
    early_stop_min_delta = config.get('early_stop_min_delta', 0.0)
    early_stop_patience = config.get('base_early_stop_patience', 0)
    early_stop_warmup = config.get('base_early_stop_warmup', 0)
    epochs_without_improvement = 0

    if early_stop_enabled and early_stop_patience > 0:
        print("  Early stopping: "
              f"patience={early_stop_patience}, "
              f"min_delta={early_stop_min_delta}, "
              f"warmup={early_stop_warmup}")

    best_acc = -1.0
    for epoch in range(config['base_epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, base_classes, epoch,
            config['base_epochs'], grad_accum_steps=grad_accum_steps,
        )
        val_acc = evaluate(model, val_loader, device, base_classes)

        print(f"Epoch {epoch+1}/{config['base_epochs']}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}")

        improved = val_acc > best_acc + early_stop_min_delta
        if improved:
            best_acc = val_acc
            epochs_without_improvement = 0
            model.save_netB(os.path.join(args.output_dir, 'netB_best.pth'))
        elif early_stop_enabled and epoch + 1 >= early_stop_warmup:
            epochs_without_improvement += 1

        if (early_stop_enabled and early_stop_patience > 0 and
                epoch + 1 >= early_stop_warmup and
                epochs_without_improvement >= early_stop_patience):
            print(f"Early stopping base training at epoch {epoch+1}: "
                  f"best Val Acc={best_acc:.4f}, "
                  f"no improvement for {epochs_without_improvement} epochs.")
            break

    # ---------------------------------------------------------------
    # 5. Save final model and exemplars
    # ---------------------------------------------------------------
    model.save_netB(os.path.join(args.output_dir, 'netB_final.pth'))
    save_exemplars(train_dataset, base_classes, config, args.output_dir)

    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Save base accuracy for incremental training to load
    with open(os.path.join(args.output_dir, 'best_acc.yaml'), 'w') as f:
        yaml.dump({'base_accuracy': best_acc, 'num_base_classes': num_base}, f)

    print(f"\nBase training complete!")
    print(f"  Best validation accuracy: {best_acc:.4f}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
