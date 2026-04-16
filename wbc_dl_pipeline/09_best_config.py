# %% [markdown]
# # 🏆 Best Config V3 — EfficientNet-B3 (class weights fixés)
# 
# **Bugs identifiés dans V1/V2** :
# - Class weights PLY=9.24 vs LY=0.01 → modèle ignore les classes majoritaires
# - Focal Loss amplifie le problème
# - Mixup brouille des classes déjà très similaires
# 
# **Corrections V3** :
# - Class weights en RACINE CARRÉE (atténue le déséquilibre sans l'ignorer)
# - CrossEntropyLoss simple + label smoothing (pas de Focal Loss)
# - PAS de Mixup/CutMix (trop dangereux sur ce dataset)
# - Phase freeze → fine-tune (comme V2)

# %% [markdown]
# ## 1. Setup

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
# ## 2. Configuration

# %%
# ===== CHEMINS — ADAPTER =====
BASE_DIR = "/home/infres/anadanak-24/projetkaggle/data/raw/IMA205-challenge 2"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")

# ===== HYPERPARAMÈTRES V3 =====
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 50
FREEZE_EPOCHS = 3
LR_MAX = 2e-4
LR_BACKBONE = 2e-5
WEIGHT_DECAY = 1e-4
PATIENCE = 10
LABEL_SMOOTHING = 0.1
VAL_SPLIT = 0.15
NUM_CLASSES = 13

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("submissions", exist_ok=True)

print(f"✅ Config V3 : IMG={IMG_SIZE}, BS={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}")
print(f"   PAS de Mixup/CutMix | PAS de Focal Loss")
print(f"   CrossEntropy + LabelSmoothing={LABEL_SMOOTHING} + sqrt(weights)")

# %% [markdown]
# ## 3. Dataset

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
print(f"\nDistribution :")
print(train_df[label_col].value_counts().to_string())

# %% [markdown]
# ## 4. Augmentation (modérée)

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(degrees=180),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
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
# ## 5. Données & Class Weights (CORRIGÉS)

# %%
# Split stratifié
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=VAL_SPLIT,
    random_state=SEED, stratify=train_df['label_idx'].values
)
train_subset = train_df.iloc[train_idx]
val_subset = train_df.iloc[val_idx]

train_ds = WBCDataset(train_subset, TRAIN_DIR, id_col, 'label_idx', train_transform)
val_ds = WBCDataset(val_subset, TRAIN_DIR, id_col, 'label_idx', val_transform)
test_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, val_transform)

# === CLASS WEIGHTS : RACINE CARRÉE ===
# Avant (V1/V2 cassé) : w = total/count → PLY=9.24, LY=0.01 (ratio 924x)
# Maintenant (V3)      : w = sqrt(total/count) → ratio max ~30x (raisonnable)
train_labels = train_subset['label_idx'].values
class_counts = Counter(train_labels)
total_samples = sum(class_counts.values())

# Méthode sqrt : atténue les extrêmes
raw_weights = [total_samples / class_counts[i] for i in range(NUM_CLASSES)]
sqrt_weights = [np.sqrt(w) for w in raw_weights]
# Normaliser pour que la moyenne = 1
mean_w = np.mean(sqrt_weights)
sqrt_weights = [w / mean_w for w in sqrt_weights]
weight_tensor = torch.FloatTensor(sqrt_weights).to(device)

print(f"\n📊 Class weights (sqrt) — comparaison :")
print(f"   {'Classe':5s} | {'Count':>6s} | {'Raw':>6s} | {'Sqrt':>6s}")
print(f"   {'-'*5} | {'-'*6} | {'-'*6} | {'-'*6}")
for i, cn in enumerate(class_names):
    cnt = class_counts.get(i, 0)
    print(f"   {cn:5s} | {cnt:6d} | {raw_weights[i]:6.2f} | {sqrt_weights[i]:6.3f}")

print(f"\n   Ratio max/min (raw)  : {max(raw_weights)/max(min(raw_weights),1e-8):.0f}x")
print(f"   Ratio max/min (sqrt) : {max(sqrt_weights)/max(min(sqrt_weights),1e-8):.1f}x")

# === SAMPLER PONDÉRÉ (aussi en sqrt) ===
sample_weights = [np.sqrt(total_samples / class_counts[label]) for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Loaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

print(f"\n   Train: {len(train_subset)} ({len(train_loader)} batches)")
print(f"   Val:   {len(val_subset)} ({len(val_loader)} batches)")

# %% [markdown]
# ## 6. Modèle

# %%
def create_model(num_classes=13, pretrained=True):
    weights = 'IMAGENET1K_V1' if pretrained else None
    model = models.efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


model = create_model(NUM_CLASSES, pretrained=True).to(device)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"✅ EfficientNet-B3 : {n_params:.1f}M params")

# %% [markdown]
# ## 7. Optimizer & Loss

# %%
def set_backbone_frozen(model, frozen=True):
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = not frozen
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"   Backbone {'🔒 gelé' if frozen else '🔓 dégelé'} | Trainable: {n_train:.1f}M")


def create_optimizer(model, lr_backbone, lr_head):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    groups = [{'params': head_params, 'lr': lr_head}]
    if backbone_params:
        groups.insert(0, {'params': backbone_params, 'lr': lr_backbone})
    return optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


# Phase 1 : freeze
set_backbone_frozen(model, frozen=True)
optimizer = create_optimizer(model, LR_BACKBONE, LR_MAX)

# === LOSS : Simple CrossEntropy + Label Smoothing + Sqrt Weights ===
# PAS de Focal Loss (c'était le problème principal)
criterion = nn.CrossEntropyLoss(
    weight=weight_tensor,
    label_smoothing=LABEL_SMOOTHING
)

scaler = GradScaler(enabled=torch.cuda.is_available())

print(f"✅ Loss : CrossEntropyLoss(label_smoothing={LABEL_SMOOTHING}, sqrt_weights)")
print(f"✅ PAS de Focal Loss | PAS de Mixup/CutMix")

# %% [markdown]
# ## 8. Training Functions

# %%
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False, ncols=120,
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

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Val]  ", leave=False, ncols=120,
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

# %% [markdown]
# ## 9. Training Loop

# %%
print(f"\n{'='*70}")
print(f"  🏆 ENTRAÎNEMENT V3 : EfficientNet-B3 (weights fixés)")
print(f"  Phase 1: Freeze ({FREEZE_EPOCHS} epochs) → Phase 2: Fine-tune")
print(f"{'='*70}\n")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
best_f1 = 0.0
best_model_state = None
patience_counter = 0
start_time = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # Phase transition
    if epoch == FREEZE_EPOCHS + 1:
        print(f"\n  🔓 Phase 2 : Dégel du backbone")
        set_backbone_frozen(model, frozen=False)
        optimizer = create_optimizer(model, LR_BACKBONE, LR_MAX)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS, eta_min=1e-7)

    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, device, epoch
    )

    # Val
    val_loss, val_acc, val_f1, val_preds, val_labels = validate(
        model, val_loader, criterion, device, epoch
    )

    # Scheduler
    if epoch > FREEZE_EPOCHS:
        scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)

    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[-1]['lr']

    vram_str = ""
    if torch.cuda.is_available():
        vram_peak = torch.cuda.max_memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_str = f" | VRAM:{vram_peak:.1f}/{vram_total:.0f}GB"

    phase = "FREEZE" if epoch <= FREEZE_EPOCHS else "FULL"
    is_best = val_f1 > best_f1
    marker = " 🏆 BEST" if is_best else ""

    print(f"  [{phase:6s}] Ep {epoch:02d}/{NUM_EPOCHS} | "
          f"T:{train_loss:.3f}/{train_acc:.3f} | "
          f"V:{val_loss:.3f}/{val_acc:.3f}/F1:{val_f1:.4f} | "
          f"LR:{current_lr:.1e} | {epoch_time:.0f}s{vram_str}{marker}")

    if is_best:
        best_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'val_f1': best_f1, 'val_acc': val_acc, 'epoch': epoch,
            'class_names': class_names, 'label2idx': label2idx,
        }, 'checkpoints/best_v3.pth')
    else:
        patience_counter += 1

    if epoch > FREEZE_EPOCHS + 5 and patience_counter >= PATIENCE:
        print(f"\n  ⏹️ Early stopping (epoch {epoch})")
        break

if best_model_state:
    model.load_state_dict(best_model_state)

total_time = time.time() - start_time
print(f"\n  ✅ Terminé en {total_time/60:.1f} min | Best F1: {best_f1:.4f}")

# %% [markdown]
# ## 10. Courbes

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ep = range(1, len(history['train_loss']) + 1)

axes[0].plot(ep, history['train_loss'], 'b-', label='Train', lw=2)
axes[0].plot(ep, history['val_loss'], 'r-', label='Val', lw=2)
axes[0].axvline(x=FREEZE_EPOCHS, color='gray', ls='--', label='Dégel')
axes[0].set_title('Loss', fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, history['train_acc'], 'b-', label='Train', lw=2)
axes[1].plot(ep, history['val_acc'], 'r-', label='Val', lw=2)
axes[1].axvline(x=FREEZE_EPOCHS, color='gray', ls='--', label='Dégel')
axes[1].set_title('Accuracy', fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(ep, history['val_f1'], 'g-', label='Val F1', lw=2)
axes[2].axhline(y=0.74, color='orange', ls='--', label='Cible 0.74')
axes[2].axhline(y=best_f1, color='red', ls=':', label=f'Best {best_f1:.4f}')
axes[2].axvline(x=FREEZE_EPOCHS, color='gray', ls='--', label='Dégel')
axes[2].set_title('F1-Score', fontweight='bold')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("best_v3_curves.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Évaluation

# %%
print(f"\n📊 Best F1-macro : {best_f1:.4f}")
print(f"📊 Accuracy     : {accuracy_score(val_labels, val_preds):.4f}")
print(f"\n{classification_report(val_labels, val_preds, target_names=class_names, zero_division=0)}")

cm = confusion_matrix(val_labels, val_preds)
cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, annot_kws={'size': 9})
ax.set_title(f'V3 (F1: {best_f1:.4f})', fontsize=14, fontweight='bold')
ax.set_ylabel('Vrai'); ax.set_xlabel('Prédit')
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig("best_v3_cm.png", dpi=150, bbox_inches='tight')
plt.show()

per_class_f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)
print("\n📊 F1 par classe :")
for idx in np.argsort(per_class_f1):
    cnt = class_counts.get(idx, 0)
    print(f"   {class_names[idx]:4s}: F1={per_class_f1[idx]:.3f} ({cnt} imgs)")

# %% [markdown]
# ## 12. Soumission

# %%
print("\n📮 Prédiction test...")
model.eval()
test_preds, test_ids_list = [], []
with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Test", ncols=100):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
        _, predicted = outputs.max(1)
        test_preds.extend(predicted.cpu().numpy())
        test_ids_list.extend(ids)

sub = pd.DataFrame({
    test_df.columns[0]: test_ids_list,
    label_col: [idx2label[p] for p in test_preds]
})
sub.to_csv("submissions/submission_v3.csv", index=False)
print(f"✅ submissions/submission_v3.csv")
print(sub[label_col].value_counts().to_string())

# %%
# TTA x5
print("\n📮 TTA x5...")
tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

N_TTA = 5
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

sub_tta = pd.DataFrame({
    test_df.columns[0]: tta_ids,
    label_col: [idx2label[p.item()] for p in tta_preds]
})
sub_tta.to_csv("submissions/submission_v3_tta.csv", index=False)
print(f"✅ submissions/submission_v3_tta.csv")
print(sub_tta[label_col].value_counts().to_string())