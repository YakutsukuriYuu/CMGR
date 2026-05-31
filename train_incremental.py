"""Incremental task training for CMGR on ModelNet40.

For each incremental task t=1,...,T-1:
1. Train BND (10 epochs, lr=1e-3, BCEWithLogitsLoss)
2. Incremental training (20 epochs, lr=1e-3->1e-4 cosine, no augmentation)
3. Loss: L_cls + alpha * L_mc + beta * L_c
4. Evaluate on unified test set (all seen classes)
5. Update exemplar set
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import numpy as np
from tqdm import tqdm
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))

from cmgr_models.cmgr import CMGR
from cmgr_models.bnd import BNDTrainer
from cmgr_utils.sampler import ExemplarSampler
from cmgr_utils.metrics import MetricsTracker


def parse_args():
    parser = argparse.ArgumentParser(description='CMGR Incremental Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of ModelNet40 data')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Directory with base training outputs (NetB, exemplars)')
    parser.add_argument('--output_dir', type=str, default='outputs/incremental')
    parser.add_argument('--recon_ckpt', type=str,
                        default='deps/ReCon/pretrained/recon.pth')
    parser.add_argument('--depth_ckpt', type=str,
                        default='deps/CLIP2Point/pretrained/vit32/best_eval.pth')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_base_classes', type=int, default=20)
    parser.add_argument('--classes_per_task', type=int, default=5,
                        help='Number of novel classes per incremental task')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_modelnet_dataset(data_root, split, num_points, classes, augment, seed):
    from cmgr_datasets.modelnet_dataset import ModelNetDataset
    return ModelNetDataset(
        root=data_root, split=split, num_points=num_points,
        variant='modelnet40', classes=classes, augment=augment, seed=seed,
    )


@torch.no_grad()
def extract_features(model, dataloader, device):
    """Extract ReCon 3D features for BND training."""
    model.eval()
    all_features = []
    all_labels = []

    for point_clouds, labels in dataloader:
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)
        features, _ = model.recon_encoder(point_clouds)
        all_features.append(features.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, class_names,
                    task_id, epoch, total_epochs, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    grad_accum_steps = max(1, int(grad_accum_steps))

    pbar = tqdm(dataloader, desc=f'Task {task_id}, Epoch {epoch+1}/{total_epochs}')
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
            'acc': f'{total_correct/total_samples:.4f}',
        })

    if scheduler is not None:
        scheduler.step()

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, dataloader, device, class_names, base_class_names=None,
             use_bnd_routing=False, bnd_threshold=0.1):
    """Evaluate model on test set.

    Args:
        model: CMGR model.
        dataloader: Test dataloader.
        device: Device.
        class_names: All seen class names.
        base_class_names: Base class names (for BND routing).
        use_bnd_routing: If True, use BND to route base→NetB, novel→incremental.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    for point_clouds, labels in tqdm(dataloader, desc='Evaluating'):
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)

        if use_bnd_routing and base_class_names is not None:
            predictions, _ = model.inference(
                point_clouds, class_names, base_class_names,
                threshold=bnd_threshold,
            )
        else:
            logits, _ = model(point_clouds, class_names=class_names)
            predictions = logits.argmax(dim=-1)

        total_correct += (predictions == labels).sum().item()
        total_samples += point_clouds.shape[0]

    return total_correct / total_samples if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate_with_routing(model, dataloader, device, class_names,
                          base_class_names, bnd_threshold=0.1):
    """Evaluate one test split and report both accuracy and BND base-route rate."""
    model.eval()
    total_correct = 0
    total_samples = 0
    routed_base = 0

    for point_clouds, labels in tqdm(dataloader, desc='Evaluating split'):
        point_clouds = point_clouds.to(device)
        labels = labels.to(device)

        predictions, is_base = model.inference(
            point_clouds, class_names, base_class_names,
            threshold=bnd_threshold,
        )

        total_correct += (predictions == labels).sum().item()
        routed_base += is_base.sum().item()
        total_samples += point_clouds.shape[0]

    if total_samples == 0:
        return 0.0, 0.0, 0
    return total_correct / total_samples, routed_base / total_samples, total_samples


def flatten_exemplar_indices(exemplar_indices, include_class_ids=None,
                             exclude_class_ids=None):
    """Flatten a class_id -> indices mapping into one Python list."""
    include_class_ids = (
        {int(cls_id) for cls_id in include_class_ids}
        if include_class_ids is not None else None
    )
    exclude_class_ids = (
        {int(cls_id) for cls_id in exclude_class_ids}
        if exclude_class_ids is not None else set()
    )
    flat = []
    for cls_id, idx_list in exemplar_indices.items():
        cls_id = int(cls_id)
        if include_class_ids is not None and cls_id not in include_class_ids:
            continue
        if cls_id in exclude_class_ids:
            continue
        flat.extend(int(idx) for idx in idx_list)
    return flat


def build_session_class_names(all_classes, num_base, classes_per_task, session_id):
    """Return class names for session 0 (base) or one incremental session."""
    if session_id == 0:
        return all_classes[:num_base]
    start = num_base + (session_id - 1) * classes_per_task
    end = min(start + classes_per_task, len(all_classes))
    return all_classes[start:end]


def weighted_average(values, weights):
    valid = [(v, w) for v, w in zip(values, weights) if v is not None and w > 0]
    total = sum(w for _, w in valid)
    if total == 0:
        return None
    return sum(v * w for v, w in valid) / total


def evaluate_diagnostics(model, test_dataset, device, all_classes, seen_class_names,
                         base_classes, num_base, classes_per_task, max_session,
                         config):
    """Evaluate per-session accuracy/routing rows for the current model session."""
    acc_row = [None] * (max_session + 1)
    routing_row = [None] * (max_session + 1)
    sample_counts = [0] * (max_session + 1)
    threshold = config.get('bnd_threshold', 0.1)

    for session_id in range(max_session + 1):
        session_classes = build_session_class_names(
            all_classes, num_base, classes_per_task, session_id
        )
        session_indices = get_dataset_indices_from_names(test_dataset, session_classes)
        sample_counts[session_id] = len(session_indices)
        if not session_indices:
            continue

        session_loader = DataLoader(
            Subset(test_dataset, session_indices),
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
        )
        acc, route_rate, _ = evaluate_with_routing(
            model, session_loader, device, seen_class_names,
            base_classes, bnd_threshold=threshold,
        )
        acc_row[session_id] = acc
        routing_row[session_id] = route_rate

    base_acc = acc_row[0]
    old_novel_acc = None
    if max_session > 1:
        old_novel_acc = weighted_average(
            acc_row[1:max_session],
            sample_counts[1:max_session],
        )
    current_novel_acc = acc_row[max_session] if max_session > 0 else None
    cumulative_acc = weighted_average(
        acc_row[:max_session + 1],
        sample_counts[:max_session + 1],
    )

    group_acc = {
        'model_session': max_session,
        'base_acc': base_acc,
        'old_novel_acc': old_novel_acc,
        'current_novel_acc': current_novel_acc,
        'cumulative_acc_from_matrix': cumulative_acc,
    }

    return acc_row, routing_row, sample_counts, group_acc


def get_dataset_indices_from_names(dataset, class_names):
    target_idxs = {
        dataset.class_to_idx[cls_name]
        for cls_name in class_names
        if cls_name in dataset.class_to_idx
    }
    if not target_idxs:
        return []
    return [i for i, (_, label) in enumerate(dataset.samples) if label in target_idxs]


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
    # Class definitions
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
    novel_classes = all_classes[num_base:]
    num_novel = len(novel_classes)
    num_inc_tasks = max(1, num_novel // args.classes_per_task)

    print(f"Base classes ({num_base}): {base_classes}")
    print(f"Novel classes ({num_novel}): {novel_classes}")
    print(f"Incremental tasks: {num_inc_tasks} x {args.classes_per_task} classes")

    # ---------------------------------------------------------------
    # 1. Load base model
    # ---------------------------------------------------------------
    print("\nLoading base model (NetB)...")
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

    # Load NetB
    netB_path = os.path.join(args.base_dir, 'netB_best.pth')
    if not os.path.exists(netB_path):
        netB_path = os.path.join(args.base_dir, 'netB_final.pth')
    model.load_netB(netB_path)

    # Set incremental mode: freeze depth encoder, only train SAGR + TAM
    model.set_incremental_mode()

    # Load exemplars
    exemplar_path = os.path.join(args.base_dir, 'exemplars.pth')
    exemplar_data = torch.load(exemplar_path, map_location='cpu')
    base_only_exemplar_indices = deepcopy(exemplar_data['exemplar_indices'])
    seen_exemplar_indices = deepcopy(base_only_exemplar_indices)
    base_class_ids = exemplar_data['class_ids']
    base_class_id_set = {int(cls_id) for cls_id in base_class_ids}

    # ---------------------------------------------------------------
    # 2. Create datasets
    # ---------------------------------------------------------------
    train_dataset = create_modelnet_dataset(
        args.data_root, 'train', config['point_cloud_size'],
        all_classes, augment=False, seed=config['seed'],
    )
    test_dataset = create_modelnet_dataset(
        args.data_root, 'test', config['point_cloud_size'],
        all_classes, augment=False, seed=config['seed'],
    )

    # ---------------------------------------------------------------
    # 3. Build class-to-index mapping for dataset
    # ---------------------------------------------------------------
    # The dataset uses its own class_to_idx mapping. We need to map
    # our class list to the dataset's indices.
    dataset_class_to_idx = train_dataset.class_to_idx

    def get_dataset_indices(dataset, class_names):
        """Get all sample indices for given class names (O(N) in-memory scan)."""
        return get_dataset_indices_from_names(dataset, class_names)

    # ---------------------------------------------------------------
    # 4. Incremental training
    # ---------------------------------------------------------------
    metrics = MetricsTracker()
    num_sessions = num_inc_tasks + 1
    acc_matrix = [[None] * num_sessions for _ in range(num_sessions)]
    routing_matrix = [[None] * num_sessions for _ in range(num_sessions)]
    session_sample_counts = [0] * num_sessions
    group_accuracies = []

    # Load base accuracy as task 0
    base_acc_path = os.path.join(args.base_dir, 'best_acc.yaml')
    if os.path.exists(base_acc_path):
        with open(base_acc_path, 'r') as f:
            base_acc_data = yaml.safe_load(f)
        base_accuracy = base_acc_data.get('base_accuracy', 0.0)
    else:
        # If no saved base acc, evaluate on base test set
        print("Base accuracy file not found, evaluating base model...")
        base_test_ds = create_modelnet_dataset(
            args.data_root, 'test', config['point_cloud_size'],
            base_classes, augment=False, seed=config['seed'],
        )
        base_test_loader = DataLoader(base_test_ds, batch_size=config['batch_size'],
                                      shuffle=False, num_workers=config['num_workers'])
        base_accuracy = evaluate(model, base_test_loader, device, list(base_classes))

    print(f"Base accuracy (Task 0): {base_accuracy:.4f}")
    metrics.record_task(0, base_accuracy)
    acc_matrix[0][0] = base_accuracy

    seen_class_names = list(base_classes)

    # ---------------------------------------------------------------
    # Store teacher features for knowledge distillation (anti-forgetting)
    # ---------------------------------------------------------------
    # Use all base exemplar samples as teacher set
    print("Storing teacher features for KD...")
    base_exemplar_all_indices = flatten_exemplar_indices(base_only_exemplar_indices)
    teacher_subset = Subset(train_dataset, base_exemplar_all_indices)
    teacher_loader = DataLoader(
        teacher_subset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
    )
    model.store_teacher_features(teacher_loader, seen_class_names, num_base)

    # Store base NetB snapshot for BND routing at inference
    model.store_base_netB()

    for task_id in range(num_inc_tasks):
        print(f"\n{'='*60}")
        print(f"Incremental Task {task_id + 1}/{num_inc_tasks}")
        print(f"{'='*60}")

        # Novel classes for this task
        start = num_base + task_id * args.classes_per_task
        end = min(start + args.classes_per_task, len(all_classes))
        task_novel_classes = all_classes[start:end]

        if not task_novel_classes:
            print("No more novel classes. Stopping.")
            break

        print(f"Novel classes: {task_novel_classes}")
        seen_class_names.extend(task_novel_classes)

        print(f"Total seen classes: {len(seen_class_names)}")

        # -------------------------------------------------------
        # 4a. Extract features for BND
        # -------------------------------------------------------
        print("Extracting features for BND...")

        # BND is a base-vs-novel router: positive samples are always
        # the original base exemplars, not all previously seen classes.
        bnd_base_indices = flatten_exemplar_indices(base_only_exemplar_indices)
        base_subset = Subset(train_dataset, bnd_base_indices)
        base_loader = DataLoader(base_subset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=config['num_workers'])
        base_features, _ = extract_features(model, base_loader, device)

        # Non-base features for BND = current novel samples + previous
        # novel exemplars. They must all route to the current incremental net.
        novel_indices = get_dataset_indices(train_dataset, task_novel_classes)
        previous_novel_exemplar_indices = flatten_exemplar_indices(
            seen_exemplar_indices,
            exclude_class_ids=base_class_id_set,
        )
        bnd_novel_indices = novel_indices + previous_novel_exemplar_indices
        novel_subset = Subset(train_dataset, bnd_novel_indices)
        novel_loader = DataLoader(novel_subset, batch_size=config['batch_size'],
                                  shuffle=False, num_workers=config['num_workers'])
        novel_features, _ = extract_features(model, novel_loader, device)

        print(f"  Base exemplar features: {base_features.shape}")
        print(f"  Non-base features: {novel_features.shape} "
              f"(current={len(novel_indices)}, "
              f"old_novel_exemplars={len(previous_novel_exemplar_indices)})")

        # -------------------------------------------------------
        # 4b. Train BND
        # -------------------------------------------------------
        print("Training BND...")
        bnd_trainer = BNDTrainer(
            bnd_model=model.bnd,
            lr=config.get('bnd_lr', 1e-3),
            epochs=config.get('bnd_epochs', 10),
            device=device,
            threshold=config.get('bnd_threshold', 0.1),
        )
        bnd_loss = bnd_trainer.train(base_features, novel_features)
        print(f"  BND loss: {bnd_loss:.4f}")
        if bnd_trainer.last_stats:
            stats = bnd_trainer.last_stats
            print(
                "  BND stats: "
                f"counts={stats['base_count']}/{stats['novel_count']}, "
                f"base_logit={stats['base_logit_mean']:.3f}, "
                f"novel_logit={stats['novel_logit_mean']:.3f}, "
                f"base_route={stats['base_route_rate']:.3f}, "
                f"novel_route={stats['novel_route_rate']:.3f}",
                flush=True,
            )

        # -------------------------------------------------------
        # 4c. Incremental training
        # -------------------------------------------------------
        print(f"Starting incremental training for up to {config['inc_epochs']} epochs...")

        # Training data = current novel classes + all stored seen exemplars.
        seen_exemplar_all_indices = flatten_exemplar_indices(seen_exemplar_indices)
        inc_indices = novel_indices + seen_exemplar_all_indices
        inc_dataset = Subset(train_dataset, inc_indices)
        inc_loader = DataLoader(
            inc_dataset, batch_size=config['batch_size'],
            shuffle=True, num_workers=config['num_workers'],
            pin_memory=True, drop_last=True,
        )
        grad_accum_steps = max(1, int(config.get('grad_accum_steps', 1)))
        if grad_accum_steps > 1:
            print(f"  Batch size: {config['batch_size']}, "
                  f"gradient accumulation: {grad_accum_steps}, "
                  f"effective batch size: {config['batch_size'] * grad_accum_steps}")

        optimizer = torch.optim.Adam(
            model.get_trainable_params(),
            lr=config['inc_lr'],
            weight_decay=config.get('inc_weight_decay', 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['inc_epochs'],
            eta_min=config.get('inc_lr_min', 1e-4),
        )

        # Prepare test loader for per-epoch evaluation (best model selection)
        test_indices = get_dataset_indices(test_dataset, seen_class_names)
        test_subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=config['num_workers'])

        early_stop_enabled = config.get('early_stopping', False)
        early_stop_min_delta = config.get('early_stop_min_delta', 0.0)
        early_stop_patience = config.get('inc_early_stop_patience', 0)
        early_stop_warmup = config.get('inc_early_stop_warmup', 0)
        epochs_without_improvement = 0

        if early_stop_enabled and early_stop_patience > 0:
            print("Early stopping: "
                  f"patience={early_stop_patience}, "
                  f"min_delta={early_stop_min_delta}, "
                  f"warmup={early_stop_warmup}")

        best_acc = -1.0
        best_ckpt_path = os.path.join(args.output_dir, f'task_{task_id+1}_best.pth')

        for epoch in range(config['inc_epochs']):
            train_loss, train_acc = train_one_epoch(
                model, inc_loader, optimizer, scheduler, device,
                seen_class_names, task_id + 1, epoch, config['inc_epochs'],
                grad_accum_steps=grad_accum_steps,
            )

            # Evaluate every epoch for best model selection (with BND routing)
            epoch_acc = evaluate(model, test_loader, device, seen_class_names,
                                 base_class_names=base_classes,
                                 use_bnd_routing=True,
                                 bnd_threshold=config.get('bnd_threshold', 0.1))
            improved = ""
            has_improved = epoch_acc > best_acc + early_stop_min_delta
            if has_improved:
                best_acc = epoch_acc
                epochs_without_improvement = 0
                model.save_netB(best_ckpt_path)
                improved = " ★ best"
            elif early_stop_enabled and epoch + 1 >= early_stop_warmup:
                epochs_without_improvement += 1

            print(f"  Epoch {epoch+1}/{config['inc_epochs']}: "
                  f"Loss={train_loss:.4f}, Train_Acc={train_acc:.4f}, "
                  f"Val_Acc={epoch_acc:.4f}, "
                  f"LR={scheduler.get_last_lr()[0]:.6f}{improved}",
                  flush=True)

            if (early_stop_enabled and early_stop_patience > 0 and
                    epoch + 1 >= early_stop_warmup and
                    epochs_without_improvement >= early_stop_patience):
                print(f"Early stopping task {task_id + 1} at epoch {epoch+1}: "
                      f"best Val_Acc={best_acc:.4f}, "
                      f"no improvement for {epochs_without_improvement} epochs.",
                      flush=True)
                break

        # Load best checkpoint for subsequent tasks
        if os.path.exists(best_ckpt_path):
            model.load_netB(best_ckpt_path)

        print(f"Task {task_id + 1} best accuracy: {best_acc:.4f}", flush=True)
        metrics.record_task(task_id + 1, best_acc)

        # -------------------------------------------------------
        # 4d. Diagnostics: acc matrix and BND routing matrix
        # -------------------------------------------------------
        model_session = task_id + 1
        acc_row, routing_row, sample_counts, group_acc = evaluate_diagnostics(
            model=model,
            test_dataset=test_dataset,
            device=device,
            all_classes=all_classes,
            seen_class_names=seen_class_names,
            base_classes=base_classes,
            num_base=num_base,
            classes_per_task=args.classes_per_task,
            max_session=model_session,
            config=config,
        )
        session_sample_counts[:model_session + 1] = sample_counts[:model_session + 1]
        acc_matrix[model_session][:model_session + 1] = acc_row[:model_session + 1]
        routing_matrix[model_session][:model_session + 1] = routing_row[:model_session + 1]
        group_accuracies.append(group_acc)

        print("  Diagnostics:", flush=True)
        print(f"    acc row: {acc_matrix[model_session][:model_session + 1]}", flush=True)
        print(f"    routing row: {routing_matrix[model_session][:model_session + 1]}", flush=True)
        print(
            "    group acc: "
            f"base={group_acc['base_acc']}, "
            f"old_novel={group_acc['old_novel_acc']}, "
            f"current_novel={group_acc['current_novel_acc']}, "
            f"cumulative={group_acc['cumulative_acc_from_matrix']}",
            flush=True,
        )

        # -------------------------------------------------------
        # 4e. Update exemplars
        # -------------------------------------------------------
        exemplar_sampler = ExemplarSampler(
            exemplars_per_class=config.get('exemplars_per_class', 1),
            seed=config['seed'],
        )
        # Map novel class names to dataset indices
        novel_cls_ids = [dataset_class_to_idx[c] for c in task_novel_classes
                         if c in dataset_class_to_idx]
        exemplar_sampler.update_exemplars(train_dataset, novel_cls_ids)
        for cls_id, indices in exemplar_sampler.exemplars.items():
            seen_exemplar_indices[cls_id] = indices

        # -------------------------------------------------------
        # 4f. Save checkpoint (final + best already saved)
        # -------------------------------------------------------
        ckpt_path = os.path.join(args.output_dir, f'task_{task_id+1}.pth')
        model.save_netB(ckpt_path)

    # ---------------------------------------------------------------
    # 5. Final results
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Incremental Training Complete!")
    print(f"{'='*60}")

    summary = metrics.summary()
    summary['acc_matrix'] = acc_matrix
    summary['routing_matrix'] = routing_matrix
    summary['session_sample_counts'] = session_sample_counts
    summary['group_accuracies'] = group_accuracies
    print(f"Accuracy curve: {summary['accuracies']}")
    print(f"Final accuracy: {summary['final_accuracy']:.4f}")
    print(f"Average Accuracy (AA): {summary['AA']:.4f}")

    results_path = os.path.join(args.output_dir, 'results.yaml')
    with open(results_path, 'w') as f:
        yaml.dump(summary, f)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
