# %% [markdown]
# # 🏆 ResNet50 — Config V4 (Oversampling + 2-Stage)
# 
# Même stratégie que V4 (EfficientNet-B3) mais avec ResNet50.
# Architecture différente → erreurs différentes → meilleur ensemble.

# %%
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"🖥️  GPU : {props.name} ({props.total_memory / 1e9:.1f} GB)")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
print(f"   PyTorch {torch.__version__}")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %% [markdown]
# ## Config

# %%
BASE_DIR = "/home/infres/anadanak-24/projetkaggle/data/raw/IMA205-challenge 2"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")

IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
NUM_CLASSES = 13
OVERSAMPLE_MIN = 500

# Stage 1
S1_EPOCHS = 35
S1_LR_HEAD = 2e-4
S1_LR_BACKBONE = 2e-5
S1_FREEZE_EPOCHS = 3
S1_PATIENCE = 8

# Stage 2
S2_EPOCHS = 15
S2_LR = 1e-5
S2_PATIENCE = 5

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("submissions", exist_ok=True)

print(f"✅ ResNet50 V4 : IMG={IMG_SIZE}, BS={BATCH_SIZE}")
print(f"   Stage1({S1_EPOCHS}ep) + Stage2({S2_EPOCHS}ep) | Oversample min={OVERSAMPLE_MIN}")

# %% [markdown]
# ## Données + Oversampling

# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

id_col = train_df.columns[0]
label_col = train_df.columns[1]

class_names = sorted(train_df[label_col].unique())
label2idx = {lbl: idx for idx, lbl in enumerate(class_names)}
idx2label = {idx: lbl for lbl, idx in label2idx.items()}
train_df['label_idx'] = train_df[label_col].map(label2idx)

print(f"📊 Train: {len(train_df)} | Test: {len(test_df)} | Classes: {len(class_names)}")

# Split AVANT oversampling
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=VAL_SPLIT,
    random_state=SEED, stratify=train_df['label_idx'].values
)
train_subset_raw = train_df.iloc[train_idx].copy()
val_subset = train_df.iloc[val_idx].copy()

# Oversampling offline
def oversample_dataframe(df, label_col, min_count):
    parts = []
    for cls in df[label_col].unique():
        cls_df = df[df[label_col] == cls]
        if len(cls_df) >= min_count:
            parts.append(cls_df)
        else:
            extra = cls_df.sample(n=min_count - len(cls_df), replace=True, random_state=SEED)
            parts.append(pd.concat([cls_df, extra], ignore_index=True))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)

train_subset = oversample_dataframe(train_subset_raw, label_col, OVERSAMPLE_MIN)

print(f"   Train raw: {len(train_subset_raw)} → oversampled: {len(train_subset)}")
print(f"   Val: {len(val_subset)}")

dist_before = train_subset_raw[label_col].value_counts()
dist_after = train_subset[label_col].value_counts()
for cn in class_names:
    b = dist_before.get(cn, 0)
    a = dist_after.get(cn, 0)
    ratio = f" (x{a/max(b,1):.0f})" if a > b else ""
    print(f"   {cn:4s}: {b:5d} → {a:5d}{ratio}")

# %% [markdown]
# ## Transforms & Dataset

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(degrees=180),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.06),
    transforms.RandomGrayscale(p=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


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
                print(f"⚠️ Manquant : {img_path}")
                self._warned.add(img_path)
            return self.__getitem__((idx + 1) % len(self.df))
        if self.transform:
            image = self.transform(image)
        if self.label_col is not None:
            return image, row[self.label_col]
        return image, img_id

# %% [markdown]
# ## Loaders & Weights

# %%
train_ds = WBCDataset(train_subset, TRAIN_DIR, id_col, 'label_idx', train_transform)
val_ds = WBCDataset(val_subset, TRAIN_DIR, id_col, 'label_idx', val_transform)
test_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, val_transform)

train_labels = train_subset['label_idx'].values
class_counts = Counter(train_labels)
total_samples = sum(class_counts.values())

# Sqrt weights (comme V3/V4)
raw_w = [total_samples / class_counts[i] for i in range(NUM_CLASSES)]
sqrt_w = [np.sqrt(w) for w in raw_w]
mean_w = np.mean(sqrt_w)
sqrt_w = [w / mean_w for w in sqrt_w]
weight_tensor = torch.FloatTensor(sqrt_w).to(device)

sample_weights = [np.sqrt(total_samples / class_counts[label]) for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"   Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

# %% [markdown]
# ## Modèle ResNet50

# %%
def create_resnet50(num_classes=13, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model

model = create_resnet50(NUM_CLASSES, pretrained=True).to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✅ ResNet50 : {n_params:.1f}M params")

# %% [markdown]
# ## Utilities

# %%
def set_backbone_frozen(model, frozen=True):
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = not frozen
    n = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"   Backbone {'🔒' if frozen else '🔓'} | Trainable: {n:.1f}M")


def create_optimizer(model, lr_backbone, lr_head):
    bb_p, head_p = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fc' in name:
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


def run_stage(model, train_loader, val_loader, criterion, device, scaler,
              num_epochs, lr_backbone, lr_head, patience, stage_name,
              freeze_epochs=0, best_f1_init=0.0):
    print(f"\n{'='*70}")
    print(f"  {stage_name}")
    print(f"  Epochs: {num_epochs} | LR: bb={lr_backbone}, head={lr_head}")
    print(f"{'='*70}\n")

    if freeze_epochs > 0:
        set_backbone_frozen(model, frozen=True)
    else:
        set_backbone_frozen(model, frozen=False)

    optimizer = create_optimizer(model, lr_backbone, lr_head)
    scheduler = None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    best_f1 = best_f1_init
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print(f"\n  🔓 Dégel backbone")
            set_backbone_frozen(model, frozen=False)
            optimizer = create_optimizer(model, lr_backbone, lr_head)
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
        marker = " 🏆" if is_best else ""
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
            print(f"\n  ⏹️ Early stop (epoch {epoch})")
            break

    model.load_state_dict(best_state)
    print(f"\n  ✅ {stage_name} terminé | Best F1: {best_f1:.4f}")
    return model, history, best_f1, v_preds, v_labels

# %% [markdown]
# ## Stage 1 : Entraînement principal

# %%
criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=LABEL_SMOOTHING)
scaler = GradScaler(enabled=torch.cuda.is_available())

model, hist1, best_f1_s1, _, _ = run_stage(
    model, train_loader, val_loader, criterion, device, scaler,
    num_epochs=S1_EPOCHS, lr_backbone=S1_LR_BACKBONE, lr_head=S1_LR_HEAD,
    patience=S1_PATIENCE,
    stage_name="📗 STAGE 1 : ResNet50 (oversamplé)",
    freeze_epochs=S1_FREEZE_EPOCHS, best_f1_init=0.0,
)

torch.save({
    'model_state_dict': model.state_dict(),
    'val_f1': best_f1_s1, 'model_name': 'resnet50',
    'class_names': class_names, 'label2idx': label2idx,
}, 'checkpoints/best_resnet50_v4_stage1.pth')

# %% [markdown]
# ## Stage 2 : Fine-tune classes rares

# %%
s2_raw_w = [(total_samples / class_counts[i]) ** 0.75 for i in range(NUM_CLASSES)]
s2_mean = np.mean(s2_raw_w)
s2_w = [w / s2_mean for w in s2_raw_w]
s2_weight_tensor = torch.FloatTensor(s2_w).to(device)

s2_sample_w = [(total_samples / class_counts[label]) ** 0.75 for label in train_labels]
s2_sampler = WeightedRandomSampler(s2_sample_w, len(s2_sample_w), replacement=True)
s2_train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=s2_sampler,
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

s2_criterion = nn.CrossEntropyLoss(weight=s2_weight_tensor, label_smoothing=LABEL_SMOOTHING)

model, hist2, best_f1_s2, val_preds, val_labels = run_stage(
    model, s2_train_loader, val_loader, s2_criterion, device, scaler,
    num_epochs=S2_EPOCHS, lr_backbone=S2_LR, lr_head=S2_LR * 2,
    patience=S2_PATIENCE,
    stage_name="📘 STAGE 2 : ResNet50 fine-tune rares",
    freeze_epochs=0, best_f1_init=best_f1_s1,
)

torch.save({
    'model_state_dict': model.state_dict(),
    'val_f1': max(best_f1_s1, best_f1_s2), 'model_name': 'resnet50',
    'class_names': class_names, 'label2idx': label2idx,
}, 'checkpoints/best_resnet50_v4_final.pth')

# %% [markdown]
# ## Évaluation

# %%
final_f1 = max(best_f1_s1, best_f1_s2)
print(f"\n📊 ResNet50 F1 final : {final_f1:.4f}")
print(f"📊 Stage 1: {best_f1_s1:.4f} | Stage 2: {best_f1_s2:.4f}")
print(f"\n{classification_report(val_labels, val_preds, target_names=class_names, zero_division=0)}")

# %% [markdown]
# ## Soumission ResNet50 seul + TTA

# %%
print("\n📮 Prédiction test + TTA x10...")

tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.04),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

model.eval()
N_TTA = 10
all_probs = None
tta_ids = None

for p in range(N_TTA + 1):
    t = val_transform if p == 0 else tta_transform
    ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    batch_probs, batch_ids = [], []
    with torch.no_grad():
        for images, ids in tqdm(loader, desc=f"TTA {p}/{N_TTA}", leave=False, ncols=100):
            images = images.to(device, non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
            batch_probs.append(F.softmax(outputs, dim=1).cpu())
            if p == 0:
                batch_ids.extend(ids)
    probs = torch.cat(batch_probs, dim=0)
    if all_probs is None:
        all_probs = probs
        tta_ids = batch_ids
    else:
        all_probs += probs

all_probs /= (N_TTA + 1)
_, tta_preds = all_probs.max(1)

sub = pd.DataFrame({
    test_df.columns[0]: tta_ids,
    label_col: [idx2label[p.item()] for p in tta_preds]
})
sub.to_csv("submissions/submission_resnet50_v4_tta.csv", index=False)
print(f"\n✅ submissions/submission_resnet50_v4_tta.csv")
print(sub[label_col].value_counts().to_string())

print("\n🎯 Prochaine étape : lancer 10_ensemble.py pour combiner ResNet50 + EfficientNet-B3 !")