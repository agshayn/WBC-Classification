##### utils.py, prepare the dataset for mobilenet

import os
import time
import copy
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)


BASE_DIR = "../data/raw/IMA205-challenge 2" #### change the root if needed
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")
CHECKPOINTS_DIR = "./checkpoints"
SUBMISSIONS_DIR = "./submissions"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
VAL_SPLIT = 0.15
SEED = 42
NUM_CLASSES = 13

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def setup_device(gpu_id=None):
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"   GPUs visibles  : {n_gpus}")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"   GPU {i}          : {props.name} ({props.total_memory / 1e9:.1f} GB, "
                  f"compute {props.major}.{props.minor})")
        print(f"   PyTorch CUDA   : {torch.version.cuda}")
        print(f"   cuDNN version  : {torch.backends.cudnn.version()}")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"cuDNN benchmark + TF32 activés (optimal pour Ampere/3090)")
    else:
        print("Pas de GPU détecté — entraînement sur CPU (très lent)")
    
    print(f"   PyTorch        : {torch.__version__}")
    return device


def setup_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dirs():
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

###### Dataset
class WBCDataset(Dataset):
    _missing_files_warned = set()
    
    def __init__(self, df, img_dir, id_col, label_col=None, transform=None,
                 verify_files=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.id_col = id_col
        self.label_col = label_col
        self.transform = transform
        if verify_files:
            self._verify_files()
    
    def _verify_files(self):
        print(f"Vérification de l'existence des {len(self.df)} fichiers...")
        exists_mask = []
        for idx in range(len(self.df)):
            img_id = str(self.df.iloc[idx][self.id_col])
            if not img_id.lower().endswith('.png'):
                img_id = f"{img_id}.png"
            exists_mask.append(os.path.exists(os.path.join(self.img_dir, img_id)))
        
        n_missing = sum(1 for e in exists_mask if not e)
        if n_missing > 0:
            print(f"{n_missing}/{len(self.df)} fichiers manquants — ils seront ignorés")
            self.df = self.df[exists_mask].reset_index(drop=True)
            print(f"{len(self.df)} fichiers valides conservés")
        else:
            print(f"Tous les fichiers sont présents")
    
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
        except (FileNotFoundError, OSError) as e:
            if img_path not in WBCDataset._missing_files_warned:
                print(f"Fichier manquant ignoré : {img_path}")
                WBCDataset._missing_files_warned.add(img_path)
            return self.__getitem__((idx + 1) % len(self.df))
        
        if self.transform:
            image = self.transform(image)
        
        if self.label_col is not None:
            return image, row[self.label_col]
        return image, img_id


##### Transforms

def get_transforms(img_size=IMG_SIZE):
    train_t = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])
    val_t = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_t, val_t


###### Data preparation
def prepare_data(img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                 num_workers=NUM_WORKERS, val_split=VAL_SPLIT, seed=SEED,
                 verify_files=False):

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    id_col = train_df.columns[0]
    label_col = train_df.columns[1]
    
    class_names = sorted(train_df[label_col].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(class_names)}
    idx2label = {idx: lbl for lbl, idx in label2idx.items()}
    train_df['label_idx'] = train_df[label_col].map(label2idx)
    
    print(f"Train : {len(train_df)} | Test : {len(test_df)} | Classes : {len(class_names)}")
    
    if verify_files:
        print(f"\nVérification des fichiers train...")
        valid_mask = []
        for idx in tqdm(range(len(train_df)), desc="Verify train", ncols=100):
            img_id = str(train_df.iloc[idx][id_col])
            if not img_id.lower().endswith('.png'):
                img_id = f"{img_id}.png"
            valid_mask.append(os.path.exists(os.path.join(TRAIN_DIR, img_id)))
        
        n_missing = sum(1 for v in valid_mask if not v)
        if n_missing > 0:
            print(f"{n_missing}/{len(train_df)} fichiers train manquants — filtrés")
            train_df = train_df[valid_mask].reset_index(drop=True)
        else:
            print(f"Tous les fichiers train présents")
    
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=val_split,
        random_state=seed,
        stratify=train_df['label_idx'].values
    )
    train_subset = train_df.iloc[train_idx]
    val_subset = train_df.iloc[val_idx]
    print(f"Split : {len(train_subset)} train / {len(val_subset)} val")
    
    train_transform, val_transform = get_transforms(img_size)
    
    train_ds = WBCDataset(train_subset, TRAIN_DIR, id_col, 'label_idx', train_transform)
    val_ds = WBCDataset(val_subset, TRAIN_DIR, id_col, 'label_idx', val_transform)
    test_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, val_transform)
    
    train_labels = train_subset['label_idx'].values
    class_counts = Counter(train_labels)
    total = sum(class_counts.values())
    sample_weights = [total / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    weight_tensor = torch.FloatTensor(
        [total / class_counts[i] for i in range(len(class_names))]
    )
    weight_tensor = weight_tensor / weight_tensor.sum() * len(class_names)
    
    return (train_loader, val_loader, test_loader,
            class_names, label2idx, idx2label,
            train_subset, val_subset, test_df,
            weight_tensor)

##### Models

def create_model(model_name, num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, num_classes)
        )
    
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
    return model


def get_model_input_size(model_name):
    if model_name == 'inception_v3':
        return 299
    return 224



class EarlyStopping:    
    def __init__(self, patience=7, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    is_inception=False, epoch_num=None, total_epochs=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    desc = f"Epoch {epoch_num}/{total_epochs} [Train]" if epoch_num else "Train"
    pbar = tqdm(loader, desc=desc, leave=False, ncols=120,
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            if is_inception and model.training:
                outputs, aux_outputs = model(images)
                loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            else:
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
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc':  f'{correct/total:.4f}',
        })
    
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch_num=None, total_epochs=None):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    desc = f"Epoch {epoch_num}/{total_epochs} [Val]  " if epoch_num else "Val"
    pbar = tqdm(loader, desc=desc, leave=False, ncols=120,
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
        
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc':  f'{correct/total:.4f}',
        })
    
    return (running_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels))


def train_model(model_name, train_loader, val_loader, weight_tensor, device,
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                patience=PATIENCE, freeze_backbone=False, save_best=True):

    print(f"  ENTRAÎNEMENT : {model_name.upper()}")
    print(f"  Epochs: {num_epochs} | LR: {lr} | Freeze: {freeze_backbone}")
    
    is_inception = (model_name == 'inception_v3')
    

    model = create_model(model_name, NUM_CLASSES, pretrained=True, freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
    
    if model_name == 'custom_cnn':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif freeze_backbone:
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        backbone_params, head_params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if any(k in name for k in ['fc', 'classifier', 'head', 'AuxLogits']):
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr},
        ], weight_decay=weight_decay)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    scaler = GradScaler(enabled=torch.cuda.is_available())
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            is_inception, epoch_num=epoch+1, total_epochs=num_epochs
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device,
            epoch_num=epoch+1, total_epochs=num_epochs
        )
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        vram_str = ""
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1e9
            vram_peak = torch.cuda.max_memory_allocated() / 1e9
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_str = f" | VRAM: {vram_used:.1f}/{vram_total:.0f}GB (peak {vram_peak:.1f}GB)"
        
        print(f"  Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s{vram_str}")
        
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping à l'époque {epoch+1}")
            break
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    
    total_time = time.time() - start_time
    print(f"\nTerminé en {total_time:.0f}s | Best Val Acc: {best_val_acc:.4f}")
    
    if save_best:
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        save_path = os.path.join(CHECKPOINTS_DIR, f"best_{model_name}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'val_acc': best_val_acc,
            'history': history,
        }, save_path)
        print(f" Modèle sauvegardé : {save_path}")
    
    _, _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    return model, history, val_preds, val_labels


##### Visualisation 
def plot_history(history, model_name, save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_title(f'{model_name} — Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_title(f'{model_name} — Accuracy', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sauvegardé : {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=True, annot_kws={'size': 8})
    
    acc = accuracy_score(y_true, y_pred)
    ax.set_title(f'{model_name} — Matrice de confusion (Acc: {acc:.4f})',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Vrai')
    ax.set_xlabel('Prédit')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sauvegardé : {save_path}")
    plt.show()
    
    return cm


def print_classification_report(y_true, y_pred, class_names):
    print(f"\nAccuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-macro : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names, zero_division=0)}")


##### Submission
@torch.no_grad()
def predict_test(model, test_df, idx2label, device, model_name,
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):

    img_size = get_model_input_size(model_name)
    _, val_transform = get_transforms(img_size)
    
    test_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, val_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    model.eval()
    all_preds, all_ids = [], []
    
    for images, ids in tqdm(test_loader, desc="Prédiction test"):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_ids.extend(ids)
    
    labels = [idx2label[p] for p in all_preds]
    return all_ids, labels


def save_submission(ids, labels, model_name, label_col='label', id_col='id'):
    """Sauvegarde la soumission au format demandé (index=False)."""
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    save_path = os.path.join(SUBMISSIONS_DIR, f"submission_{model_name}.csv")
    
    submission = pd.DataFrame({id_col: ids, label_col: labels})
    submission.to_csv(save_path, index=False)
    
    print(f"\nSoumission sauvegardée : {save_path}")
    print(f"   Nombre de prédictions : {len(submission)}")
    print(f"\nDistribution :")
    print(submission[label_col].value_counts())
    return submission



def load_checkpoint(model_name, device):
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{model_name}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trouvé : {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_model(model_name, NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Modèle chargé : {model_name} (Val Acc: {checkpoint['val_acc']:.4f})")
    return model, checkpoint