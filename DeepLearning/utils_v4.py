##### utils_v4.py (Oversampling + 2 Stage for efficientnet, resnet, convnext, swin)
import os
import time
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)


##### Path
BASE_DIR = ".../data/raw/IMA205-challenge 2" ##### change it if needed
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")

##### hyperparameters
IMG_SIZE = 256
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
NUM_CLASSES = 13
OVERSAMPLE_MIN = 500
SEED = 42
N_TTA = 10

##### stage1
S1_EPOCHS = 35
S1_LR_HEAD = 2e-4
S1_LR_BACKBONE = 2e-5
S1_FREEZE_EPOCHS = 3
S1_PATIENCE = 8

##### stage2
S2_EPOCHS = 15
S2_LR = 1e-5
S2_PATIENCE = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

##### setup

def setup(seed=SEED):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f" GPU : {props.name} ({props.total_memory / 1e9:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"   PyTorch {torch.__version__}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    return device


##### dataset

class WBCDataset(Dataset):
    def __init__(self, df, img_dir, id_col, label_col=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.id_col = id_col
        self.label_col = label_col
        self.transform = transform
        self._warned = set()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row[self.id_col])
        if img_id.lower().endswith('.png'):
            img_path = os.path.join(self.img_dir, img_id)
        else:
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            if img_path not in self._warned:
                print(f"Manquant : {img_path}")
                self._warned.add(img_path)
            return self.__getitem__((idx + 1) % len(self.df))
        if self.transform:
            image = self.transform(image)
        if self.label_col is not None:
            return image, row[self.label_col]
        return image, img_id

##### oversampling
def oversample_dataframe(df, label_col, min_count, seed=SEED):
    parts = []
    for cls in df[label_col].unique():
        cls_df = df[df[label_col] == cls]
        if len(cls_df) >= min_count:
            parts.append(cls_df)
        else:
            extra = cls_df.sample(n=min_count - len(cls_df), replace=True, random_state=seed)
            parts.append(pd.concat([cls_df, extra], ignore_index=True))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


###### transforms
def get_train_transform(img_size=IMG_SIZE, use_grayscale=False):
    t_list = [
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
    ]
    if use_grayscale:
        t_list.append(transforms.RandomGrayscale(p=0.03))
    t_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ]
    return transforms.Compose(t_list)


def get_val_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transform(img_size=IMG_SIZE, use_affine=False):
    t_list = [
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(180),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
    ]
    if use_affine:
        t_list.append(transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)))
    t_list += [
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(t_list)

##### datapreparation
def prepare_data(device, batch_size=32, use_grayscale=False):
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    id_col = train_df.columns[0]
    label_col = train_df.columns[1]

    class_names = sorted(train_df[label_col].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(class_names)}
    idx2label = {idx: lbl for lbl, idx in label2idx.items()}
    train_df['label_idx'] = train_df[label_col].map(label2idx)

    print(f"Train original: {len(train_df)} | Test: {len(test_df)} | Classes: {len(class_names)}")

    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)), test_size=VAL_SPLIT,
        random_state=SEED, stratify=train_df['label_idx'].values
    )
    train_subset_raw = train_df.iloc[train_idx].copy()
    val_subset = train_df.iloc[val_idx].copy()

    train_subset = oversample_dataframe(train_subset_raw, label_col, OVERSAMPLE_MIN)

    print(f"   Train raw: {len(train_subset_raw)} → oversampled: {len(train_subset)}")
    print(f"   Val: {len(val_subset)}")

    dist_before = train_subset_raw[label_col].value_counts()
    dist_after = train_subset[label_col].value_counts()
    for cn in class_names:
        b = dist_before.get(cn, 0)
        a = dist_after.get(cn, 0)
        ratio = f" (x{a / max(b, 1):.0f})" if a > b else ""
        print(f"   {cn:4s}: {b:5d} → {a:5d}{ratio}")

    train_transform = get_train_transform(IMG_SIZE, use_grayscale=use_grayscale)
    val_transform = get_val_transform(IMG_SIZE)

    train_ds = WBCDataset(train_subset, TRAIN_DIR, id_col, 'label_idx', train_transform)
    val_ds = WBCDataset(val_subset, TRAIN_DIR, id_col, 'label_idx', val_transform)
    test_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, val_transform)

    train_labels = train_subset['label_idx'].values
    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())

    raw_weights = [total_samples / class_counts[i] for i in range(NUM_CLASSES)]
    sqrt_weights = [np.sqrt(w) for w in raw_weights]
    mean_w = np.mean(sqrt_weights)
    sqrt_weights = [w / mean_w for w in sqrt_weights]
    weight_tensor = torch.FloatTensor(sqrt_weights).to(device)

    sample_weights = [np.sqrt(total_samples / class_counts[label]) for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    num_workers = 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"\n   Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    print(f"\nClass weights (sqrt) :")
    for i, cn in enumerate(class_names):
        cnt = class_counts.get(i, 0)
        print(f"   {cn:4s}: {cnt:5d} imgs | weight: {weight_tensor[i].item():.3f}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_ds': train_ds,
        'class_names': class_names,
        'label2idx': label2idx,
        'idx2label': idx2label,
        'weight_tensor': weight_tensor,
        'train_labels': train_labels,
        'class_counts': class_counts,
        'total_samples': total_samples,
        'test_df': test_df,
        'label_col': label_col,
        'id_col': id_col,
    }



##### stage2
def create_stage2_loader_and_criterion(train_ds, train_labels, class_counts,
                                        total_samples, weight_tensor_s1, device,
                                        batch_size=32):
    s2_raw_w = [(total_samples / class_counts[i]) ** 0.75 for i in range(NUM_CLASSES)]
    s2_mean = np.mean(s2_raw_w)
    s2_w = [w / s2_mean for w in s2_raw_w]
    s2_weight_tensor = torch.FloatTensor(s2_w).to(device)

    s2_sample_w = [(total_samples / class_counts[label]) ** 0.75 for label in train_labels]
    s2_sampler = WeightedRandomSampler(s2_sample_w, len(s2_sample_w), replacement=True)

    s2_train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=s2_sampler,
                                 num_workers=4, pin_memory=True, drop_last=True)

    s2_criterion = nn.CrossEntropyLoss(weight=s2_weight_tensor, label_smoothing=LABEL_SMOOTHING)

    return s2_train_loader, s2_criterion


##### models

def create_model(model_name, num_classes=NUM_CLASSES, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None

    if model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        head_key = 'classifier'

    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        head_key = 'fc'

    elif model_name == 'convnext_small':
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        head_key = 'classifier'

    elif model_name == 'swin_tiny':
        model = models.swin_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        head_key = 'head'

    else:
        raise ValueError(f"Modèle inconnu: {model_name}")

    return model, head_key


def set_backbone_frozen(model, head_key, frozen=True):
    for name, param in model.named_parameters():
        if head_key not in name:
            param.requires_grad = not frozen
    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"   Backbone {'locked' if frozen else 'unlocked'} | Trainable: {n:.1f}M")


def create_optimizer(model, head_key, lr_backbone, lr_head):
    bb_p, head_p = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if head_key in name:
                head_p.append(param)
            else:
                bb_p.append(param)
    groups = [{'params': head_p, 'lr': lr_head}]
    if bb_p:
        groups.insert(0, {'params': bb_p, 'lr': lr_backbone})
    return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Ep {epoch:02d} [Train]", leave=False, ncols=120,
                bar_format='{l_bar}{bar:30}{r_bar}')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix(loss=f'{running_loss/total:.4f}', acc=f'{correct/total:.4f}')
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(loader, desc=f"Ep {epoch:02d} [Val]  ", leave=False, ncols=120,
                bar_format='{l_bar}{bar:30}{r_bar}')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f'{running_loss/total:.4f}', acc=f'{correct/total:.4f}')
    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)
    f1 = f1_score(labels_arr, preds, average='macro', zero_division=0)
    return running_loss / total, correct / total, f1, preds, labels_arr


def run_stage(model, head_key, train_loader, val_loader, criterion, device, scaler,
              num_epochs, lr_backbone, lr_head, patience, stage_name,
              freeze_epochs=0, best_f1_init=0.0):
    print(f"  {stage_name}")
    print(f"  Epochs: {num_epochs} | LR: bb={lr_backbone}, head={lr_head}")
    if freeze_epochs > 0:
        print(f"  Freeze: {freeze_epochs} epochs")
    if freeze_epochs > 0:
        set_backbone_frozen(model, head_key, frozen=True)
    else:
        set_backbone_frozen(model, head_key, frozen=False)

    optimizer = create_optimizer(model, head_key, lr_backbone, lr_head)
    scheduler = None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = best_f1_init
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print(f"\n  🔓 Dégel backbone")
            set_backbone_frozen(model, head_key, frozen=False)
            optimizer = create_optimizer(model, head_key, lr_backbone, lr_head)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - freeze_epochs, eta_min=1e-7)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, val_f1, v_preds, v_labels = validate(model, val_loader, criterion, device, epoch)

        if scheduler and epoch > freeze_epochs:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        dt = time.time() - t0
        lr = optimizer.param_groups[-1]['lr']
        is_best = val_f1 > best_f1
        marker = "Greatest" if is_best else ""
        phase = "FRZ" if epoch <= freeze_epochs else "FUL"

        vram = ""
        if torch.cuda.is_available():
            vram = f" | VRAM:{torch.cuda.max_memory_allocated()/1e9:.1f}GB"

        print(f"  [{phase}] Ep {epoch:02d}/{num_epochs} | "
              f"T:{train_loss:.3f}/{train_acc:.3f} | "
              f"V:{val_loss:.3f}/{val_acc:.3f}/F1:{val_f1:.4f} | "
              f"LR:{lr:.1e} | {dt:.0f}s{vram}{marker}")

        if is_best:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch > freeze_epochs + 3 and patience_counter >= patience:
            print(f"Early stop (epoch {epoch})")
            break

    model.load_state_dict(best_state)
    print(f"{stage_name} terminé | Best F1: {best_f1:.4f}")
    return model, history, best_f1, v_preds, v_labels


##### Evaluation
def print_evaluation(val_preds, val_labels, class_names, model_name, best_f1_s1, best_f1_s2):
    final_f1 = max(best_f1_s1, best_f1_s2)
    print(f"{model_name} F1 final : {final_f1:.4f}")
    print(f"Stage 1: {best_f1_s1:.4f} | Stage 2: {best_f1_s2:.4f}")
    print(f"Val accuracy: {accuracy_score(val_labels, val_preds):.4f}")
    print(f"\n{classification_report(val_labels, val_preds, target_names=class_names, zero_division=0)}")


##### TTA/Submission
@torch.no_grad()
def predict_tta(model, test_df, device, idx2label, label_col,
                batch_size=16, n_tta=N_TTA, use_affine=False):
    model.eval()
    val_transform = get_val_transform(IMG_SIZE)
    tta_transform = get_tta_transform(IMG_SIZE, use_affine=use_affine)

    all_probs = None
    all_ids = None

    for p in range(n_tta + 1):
        t = val_transform if p == 0 else tta_transform
        ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
        batch_probs, batch_ids = [], []
        for images, ids in tqdm(loader, desc=f"TTA {p}/{n_tta}", leave=False, ncols=100):
            images = images.to(device, non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
            batch_probs.append(F.softmax(outputs, dim=1).cpu())
            if p == 0:
                batch_ids.extend(ids)
        probs = torch.cat(batch_probs, dim=0)
        if all_probs is None:
            all_probs = probs
            all_ids = batch_ids
        else:
            all_probs += probs

    all_probs /= (n_tta + 1)
    _, preds = all_probs.max(1)
    labels = [idx2label[pred.item()] for pred in preds]

    return all_ids, labels, all_probs


def save_submission(ids, labels, test_df, label_col, filename):
    os.makedirs("submissions", exist_ok=True)
    sub = pd.DataFrame({test_df.columns[0]: ids, label_col: labels})
    sub.to_csv(f"submissions/{filename}", index=False)
    print(f"submissions/{filename}")
    print(sub[label_col].value_counts().to_string())
    return sub


##### Full Pipeline
def run_full_pipeline(model_name, batch_size=32, use_grayscale=False, use_affine=False):
    device = setup()
    print(f"{model_name} V4 : IMG={IMG_SIZE}, BS={batch_size}")
    print(f"   Stage1({S1_EPOCHS}ep) + Stage2({S2_EPOCHS}ep) | Oversample min={OVERSAMPLE_MIN}")

    # Data
    data = prepare_data(device, batch_size=batch_size, use_grayscale=use_grayscale)

    # Model
    model, head_key = create_model(model_name, NUM_CLASSES, pretrained=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{model_name} : {n_params:.1f}M params")

    # Stage 1
    criterion = nn.CrossEntropyLoss(weight=data['weight_tensor'], label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    model, hist1, best_f1_s1, _, _ = run_stage(
        model, head_key, data['train_loader'], data['val_loader'],
        criterion, device, scaler,
        num_epochs=S1_EPOCHS, lr_backbone=S1_LR_BACKBONE, lr_head=S1_LR_HEAD,
        patience=S1_PATIENCE,
        stage_name=f"STAGE 1 : {model_name} (oversamplé)",
        freeze_epochs=S1_FREEZE_EPOCHS, best_f1_init=0.0,
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'val_f1': best_f1_s1, 'model_name': model_name,
        'class_names': data['class_names'], 'label2idx': data['label2idx'],
    }, f'checkpoints/best_{model_name}_v4_stage1.pth')

    # Stage 2
    s2_train_loader, s2_criterion = create_stage2_loader_and_criterion(
        data['train_ds'], data['train_labels'], data['class_counts'],
        data['total_samples'], data['weight_tensor'], device,
        batch_size=batch_size,
    )

    model, hist2, best_f1_s2, val_preds, val_labels = run_stage(
        model, head_key, s2_train_loader, data['val_loader'],
        s2_criterion, device, scaler,
        num_epochs=S2_EPOCHS, lr_backbone=S2_LR, lr_head=S2_LR * 2,
        patience=S2_PATIENCE,
        stage_name=f"STAGE 2 : {model_name} fine-tune rares",
        freeze_epochs=0, best_f1_init=best_f1_s1,
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'val_f1': max(best_f1_s1, best_f1_s2), 'model_name': model_name,
        'class_names': data['class_names'], 'label2idx': data['label2idx'],
    }, f'checkpoints/best_{model_name}_v4_final.pth')

    # Évaluation
    print_evaluation(val_preds, val_labels, data['class_names'],
                     model_name, best_f1_s1, best_f1_s2)

    # TTA & Soumission
    print(f"\nTTA x{N_TTA}...")
    ids, labels, _ = predict_tta(
        model, data['test_df'], device, data['idx2label'], data['label_col'],
        batch_size=batch_size, use_affine=use_affine,
    )
    save_submission(ids, labels, data['test_df'], data['label_col'],
                    f"submission_{model_name}_v4_tta.csv")

    return model, data