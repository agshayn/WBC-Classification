# %% [markdown]
# # Classification des Globules Blancs (WBC) — Pipeline Deep Learning (GPU)
# 
# **Architectures implémentées** (basées sur la littérature du projet) :
# 1. **CNN Custom** — Architecture Conv2D + BatchNorm + Dropout (Healthcare 2022, Fig.9a)
# 2. **VGG16** — Transfer Learning (Habibzadeh et al., 2013 ; Healthcare 2022)
# 3. **ResNet50** — Transfer Learning (Habibzadeh et al., 2018 — 99.84% accuracy)
# 4. **DenseNet121** — Transfer Learning (Sharma et al., 2022 — 98.84% accuracy)
# 5. **InceptionV3** — Transfer Learning (Habibzadeh et al., 2018 — 99.46% accuracy)
# 6. **MobileNetV2** — Léger, rapide (Cheuque et al., 2022)
# 7. **EfficientNetB0** — État de l'art moderne, bon ratio accuracy/coût
# 
# **Optimisations GPU** : Mixed Precision (FP16), DataLoader multi-workers,
# Data Augmentation on-the-fly, Learning Rate Scheduling, Early Stopping
# 
# **Exécution** : Conçu pour être exécuté sur un serveur GPU distant via SSH

# %% [markdown]
# ## 1. Imports et Configuration GPU

# %%
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

# ============================================================
# CONFIGURATION GPU
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device : {device}")

if torch.cuda.is_available():
    print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM   : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    # Optimisation cuDNN pour des inputs de taille fixe
    torch.backends.cudnn.benchmark = True
    # Activer TF32 pour les GPU Ampere+ (A100, RTX 30xx, etc.)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("⚠️  Pas de GPU détecté. L'entraînement sera très lent sur CPU.")

print(f"   PyTorch: {torch.__version__}")

# %% [markdown]
# ## 2. Hyperparamètres Globaux

# %%
# ===== CHEMINS — À MODIFIER =====
BASE_DIR = "./data"
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")
# ==================================

# ===== HYPERPARAMÈTRES =====
IMG_SIZE = 224           # Taille d'entrée standard pour les modèles pré-entraînés
BATCH_SIZE = 32          # Augmenter si assez de VRAM (64, 128)
NUM_WORKERS = 4          # Nombre de workers pour le DataLoader (ajuster selon CPU)
NUM_EPOCHS = 30          # Nombre d'époques
LEARNING_RATE = 1e-4     # LR pour transfer learning (fine-tuning)
WEIGHT_DECAY = 1e-4      # Régularisation L2
PATIENCE = 7             # Early stopping patience
VAL_SPLIT = 0.15         # Ratio de validation
SEED = 42
NUM_CLASSES = 13         # Nombre de classes WBC
# ============================

# Reproductibilité
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"✅ Config : IMG={IMG_SIZE}, BS={BATCH_SIZE}, EPOCHS={NUM_EPOCHS}, LR={LEARNING_RATE}")

# %% [markdown]
# ## 3. Chargement et Analyse du Dataset

# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

id_col = train_df.columns[0]
label_col = train_df.columns[1]

print(f"Train : {len(train_df)} images")
print(f"Test  : {len(test_df)} images")
print(f"Classes ({train_df[label_col].nunique()}) :")
class_dist = train_df[label_col].value_counts()
print(class_dist)

# Mapping label -> index
class_names = sorted(train_df[label_col].unique())
label2idx = {lbl: idx for idx, lbl in enumerate(class_names)}
idx2label = {idx: lbl for lbl, idx in label2idx.items()}
train_df['label_idx'] = train_df[label_col].map(label2idx)

print(f"\nMapping : {label2idx}")

# %%
# Visualisation de la distribution
fig, ax = plt.subplots(figsize=(12, 5))
colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
class_dist.plot(kind='bar', ax=ax, color=colors)
ax.set_title("Distribution des classes WBC", fontsize=14, fontweight='bold')
ax.set_xlabel("Classe")
ax.set_ylabel("Nombre d'images")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Dataset et Data Augmentation
# 
# Data augmentation pour améliorer la robustesse :
# - Rotation, flip horizontal/vertical
# - Color jitter (variations de couleur liées au staining)
# - RandomAffine (translation, légère mise à l'échelle)
# - Normalisation ImageNet (pour transfer learning)

# %%
class WBCDataset(Dataset):
    """
    Dataset PyTorch pour les images WBC.
    Supporte les données train (avec labels) et test (sans labels).
    """
    def __init__(self, df, img_dir, id_col, label_col=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.id_col = id_col
        self.label_col = label_col
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row[self.id_col]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        
        # Charger l'image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.label_col is not None:
            label = row[self.label_col]
            return image, label
        else:
            return image, img_id


# ===== TRANSFORMS =====

# Normalisation ImageNet (obligatoire pour les modèles pré-entraînés)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Augmentation pour l'entraînement
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),       # Marge pour crop
    transforms.RandomCrop(IMG_SIZE),                          # Random crop
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(
        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
    ),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),      # Cutout
])

# Pas d'augmentation pour validation/test
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# %%
# Split train/val (stratifié)
train_indices, val_indices = train_test_split(
    np.arange(len(train_df)),
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=train_df['label_idx'].values
)

train_subset = train_df.iloc[train_indices]
val_subset = train_df.iloc[val_indices]

print(f"Train : {len(train_subset)} images")
print(f"Val   : {len(val_subset)} images")

# Datasets
train_dataset = WBCDataset(train_subset, TRAIN_DIR, id_col, 'label_idx', train_transform)
val_dataset = WBCDataset(val_subset, TRAIN_DIR, id_col, 'label_idx', val_transform)
test_dataset = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, val_transform)

# %%
# ===== GESTION DU DÉSÉQUILIBRE DES CLASSES =====
# WeightedRandomSampler pour sur-échantillonner les classes minoritaires

train_labels = train_subset['label_idx'].values
class_counts = Counter(train_labels)
total = sum(class_counts.values())
class_weights = {cls: total / count for cls, count in class_counts.items()}
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Class weights pour la loss function (alternative)
weight_tensor = torch.FloatTensor(
    [class_weights[i] for i in range(NUM_CLASSES)]
).to(device)
# Normaliser
weight_tensor = weight_tensor / weight_tensor.sum() * NUM_CLASSES

# DataLoaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f"✅ DataLoaders créés (batch_size={BATCH_SIZE}, workers={NUM_WORKERS})")
print(f"   Batches train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

# %%
# Visualiser un batch augmenté
batch_imgs, batch_labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for i, ax in enumerate(axes.ravel()):
    if i >= len(batch_imgs):
        break
    img = batch_imgs[i].numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(f"Classe: {idx2label[batch_labels[i].item()]}", fontsize=9)
    ax.axis('off')
plt.suptitle("Batch augmenté (train)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Définition des Modèles
# 
# ### Architectures de la littérature WBC :
# 
# | Modèle | Réf. | Accuracy rapportée |
# |--------|------|-------------------|
# | CNN Custom | Healthcare 2022, Fig.9a | 73% (oversampled) |
# | VGG16 | Healthcare 2022 ; Habibzadeh 2013 | 74% → 98%+ |
# | ResNet50 | Habibzadeh et al. 2018 | 99.84% |
# | DenseNet121 | Sharma et al. 2022 | 98.84% |
# | InceptionV3 | Habibzadeh et al. 2018 | 99.46% |
# | MobileNetV2 | Cheuque et al. 2022 | 98.4% |
# | EfficientNetB0 | État de l'art | — |

# %%
# =====================================================================
# 5.1 CNN CUSTOM
# Architecture inspirée de Healthcare 2022, Fig.9(a) :
# Conv2D(64,3x3) → MaxPool → BN → ReLU → Dropout
# → Dense(512) → BN → ReLU → Dense(256) → BN → ReLU → Dense(N_CLASSES)
# =====================================================================

class CustomCNN(nn.Module):
    """
    CNN custom pour classification WBC.
    Réf: Healthcare 2022, 10, 2230, Figure 9(a).
    """
    def __init__(self, num_classes=13):
        super(CustomCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Bloc 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Bloc 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Bloc 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Bloc 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# %%
# =====================================================================
# 5.2 FACTORY POUR LES MODÈLES PRÉ-ENTRAÎNÉS (TRANSFER LEARNING)
# =====================================================================

def create_model(model_name, num_classes=13, pretrained=True, freeze_backbone=False):
    """
    Crée un modèle pré-entraîné avec une nouvelle tête de classification.
    
    Args:
        model_name: 'vgg16', 'resnet50', 'densenet121', 'inception_v3',
                    'mobilenet_v2', 'efficientnet_b0', 'custom_cnn'
        num_classes: nombre de classes de sortie
        pretrained: utiliser les poids ImageNet
        freeze_backbone: geler les couches du backbone (feature extraction only)
    """
    
    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes)
        return model
    
    # Poids pré-entraînés
    weights = 'IMAGENET1K_V1' if pretrained else None
    
    if model_name == 'vgg16':
        # Réf: Healthcare 2022 — VGG16 pour WBC
        model = models.vgg16(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[6] = nn.Linear(4096, num_classes)
    
    elif model_name == 'resnet50':
        # Réf: Habibzadeh et al. 2018 — ResNet50, 99.84% accuracy
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )
    
    elif model_name == 'densenet121':
        # Réf: Sharma et al. 2022 — DenseNet121, 98.84% accuracy
        model = models.densenet121(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier.in_features, num_classes)
        )
    
    elif model_name == 'inception_v3':
        # Réf: Habibzadeh et al. 2018 — Inception, 99.46% accuracy
        model = models.inception_v3(weights=weights, aux_logits=True)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if 'fc' not in name and 'AuxLogits' not in name:
                    param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    
    elif model_name == 'mobilenet_v2':
        # Réf: Cheuque et al. 2022 — MobileNet, 98.4% accuracy
        model = models.mobilenet_v2(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.last_channel, num_classes)
        )
    
    elif model_name == 'efficientnet_b0':
        # État de l'art — EfficientNet
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    
    else:
        raise ValueError(f"Modèle inconnu: {model_name}")
    
    return model


# Test : vérifier que tous les modèles se créent correctement
print("Vérification des architectures :")
for name in ['custom_cnn', 'vgg16', 'resnet50', 'densenet121',
             'inception_v3', 'mobilenet_v2', 'efficientnet_b0']:
    m = create_model(name, NUM_CLASSES, pretrained=False)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"  ✅ {name:20s} | Total: {n_params:6.1f}M | Trainable: {n_trainable:6.1f}M")
    del m

# %% [markdown]
# ## 6. Boucle d'Entraînement (optimisée GPU)
# 
# Inclut :
# - **Mixed Precision (AMP)** : FP16 automatique pour accélérer sur GPU
# - **Gradient Accumulation** : optionnel si VRAM limitée
# - **Learning Rate Scheduling** : CosineAnnealing ou ReduceOnPlateau
# - **Early Stopping** : arrêt si pas d'amélioration

# %%
class EarlyStopping:
    """Early stopping pour éviter le surapprentissage."""
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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, is_inception=False):
    """Entraîne le modèle sur une époque avec Mixed Precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Plus rapide que zero_grad()
        
        # Mixed Precision forward
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            if is_inception and model.training:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        # Backward avec gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_model(model_name, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                freeze_backbone=False, save_best=True):
    """
    Pipeline complète d'entraînement d'un modèle.
    
    Returns:
        model: modèle entraîné (best checkpoint)
        history: dictionnaire avec les métriques par époque
        val_preds, val_labels: prédictions sur la validation
    """
    print(f"\n{'='*70}")
    print(f"  ENTRAÎNEMENT : {model_name.upper()}")
    print(f"  Freeze backbone: {freeze_backbone} | LR: {lr} | Epochs: {num_epochs}")
    print(f"{'='*70}\n")
    
    # Créer le modèle
    is_inception = (model_name == 'inception_v3')
    model = create_model(model_name, NUM_CLASSES, pretrained=True,
                         freeze_backbone=freeze_backbone)
    model = model.to(device)
    
    # Adapter le DataLoader pour Inception (299x299)
    if is_inception:
        inception_train_transform = transforms.Compose([
            transforms.Resize((340, 340)),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        inception_val_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        inc_train_ds = WBCDataset(train_subset, TRAIN_DIR, id_col, 'label_idx', inception_train_transform)
        inc_val_ds = WBCDataset(val_subset, TRAIN_DIR, id_col, 'label_idx', inception_val_transform)
        
        current_train_loader = DataLoader(
            inc_train_ds, batch_size=BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
        )
        current_val_loader = DataLoader(
            inc_val_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True
        )
    else:
        current_train_loader = train_loader
        current_val_loader = val_loader
    
    # Loss avec class weights
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # Optimizer : paramètres différents pour backbone vs classifier
    if freeze_backbone and model_name != 'custom_cnn':
        # Seuls les params non gelés sont optimisés
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    else:
        # Discriminative learning rates : backbone LR plus bas
        if model_name == 'custom_cnn':
            optimizer = optim.AdamW(model.parameters(), lr=lr * 10,
                                    weight_decay=WEIGHT_DECAY)
        else:
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if any(key in name for key in ['fc', 'classifier', 'head', 'AuxLogits']):
                        head_params.append(param)
                    else:
                        backbone_params.append(param)
            
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': lr * 0.1},   # Backbone : LR/10
                {'params': head_params, 'lr': lr},              # Head : LR normal
            ], weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=torch.cuda.is_available())
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, mode='max')
    
    # Historique
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, current_train_loader, criterion, optimizer, scaler,
            device, is_inception
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, current_val_loader, criterion, device
        )
        
        # Scheduler step
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Epoch {epoch+1:02d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Early stopping
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"\n  ⏹️  Early stopping à l'époque {epoch+1}")
            break
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Restaurer le meilleur modèle
    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    
    total_time = time.time() - start_time
    print(f"\n  ✅ Terminé en {total_time:.0f}s | Best Val Acc: {best_val_acc:.4f}")
    
    # Sauvegarder le meilleur modèle
    if save_best:
        os.makedirs("checkpoints", exist_ok=True)
        save_path = f"checkpoints/best_{model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': model_name,
            'val_acc': best_val_acc,
            'class_names': class_names,
            'label2idx': label2idx,
        }, save_path)
        print(f"  💾 Modèle sauvegardé : {save_path}")
    
    # Évaluation finale sur validation
    _, _, final_preds, final_labels = validate(model, current_val_loader, criterion, device)
    
    return model, history, final_preds, final_labels

# %% [markdown]
# ## 7. Entraînement de Tous les Modèles

# %%
# ===== ENTRAÎNER TOUS LES MODÈLES =====
# Vous pouvez commenter les modèles que vous ne souhaitez pas entraîner

model_configs = [
    ('custom_cnn',     {'num_epochs': NUM_EPOCHS, 'lr': 1e-3,  'freeze_backbone': False}),
    ('vgg16',          {'num_epochs': NUM_EPOCHS, 'lr': 1e-4,  'freeze_backbone': False}),
    ('resnet50',       {'num_epochs': NUM_EPOCHS, 'lr': 1e-4,  'freeze_backbone': False}),
    ('densenet121',    {'num_epochs': NUM_EPOCHS, 'lr': 1e-4,  'freeze_backbone': False}),
    ('inception_v3',   {'num_epochs': NUM_EPOCHS, 'lr': 1e-4,  'freeze_backbone': False}),
    ('mobilenet_v2',   {'num_epochs': NUM_EPOCHS, 'lr': 1e-4,  'freeze_backbone': False}),
    ('efficientnet_b0',{'num_epochs': NUM_EPOCHS, 'lr': 1e-4,  'freeze_backbone': False}),
]

all_results = {}
all_histories = {}
all_predictions = {}

for model_name, config in model_configs:
    try:
        model, history, preds, labels = train_model(model_name, **config)
        
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        
        all_results[model_name] = {'accuracy': acc, 'f1_macro': f1}
        all_histories[model_name] = history
        all_predictions[model_name] = {'preds': preds, 'labels': labels}
        
        # Libérer la mémoire GPU entre les modèles
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ❌ Erreur avec {model_name}: {e}")
        import traceback
        traceback.print_exc()

# %% [markdown]
# ## 8. Courbes d'Apprentissage

# %%
n_models = len(all_histories)
fig, axes = plt.subplots(n_models, 2, figsize=(16, 5 * n_models))
if n_models == 1:
    axes = axes.reshape(1, -1)

for idx, (model_name, history) in enumerate(all_histories.items()):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[idx, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[idx, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[idx, 0].set_title(f'{model_name} — Loss', fontsize=12, fontweight='bold')
    axes[idx, 0].set_xlabel('Epoch')
    axes[idx, 0].set_ylabel('Loss')
    axes[idx, 0].legend()
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[idx, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[idx, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[idx, 1].set_title(f'{model_name} — Accuracy', fontsize=12, fontweight='bold')
    axes[idx, 1].set_xlabel('Epoch')
    axes[idx, 1].set_ylabel('Accuracy')
    axes[idx, 1].legend()
    axes[idx, 1].grid(True, alpha=0.3)

plt.suptitle("Courbes d'Apprentissage — Tous les Modèles", fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("💾 Sauvegardé : learning_curves.png")

# %% [markdown]
# ## 9. Comparaison des Modèles

# %%
results_df = pd.DataFrame(all_results).T
results_df.columns = ['Accuracy', 'F1-Score (macro)']
results_df = results_df.sort_values('Accuracy', ascending=False)
print("📊 Comparaison des modèles :\n")
print(results_df.to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_df)))

# Accuracy
bars1 = axes[0].barh(results_df.index, results_df['Accuracy'], color=colors)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Accuracy par modèle', fontweight='bold')
axes[0].set_xlim(0, 1)
for bar, val in zip(bars1, results_df['Accuracy']):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

# F1
bars2 = axes[1].barh(results_df.index, results_df['F1-Score (macro)'], color=colors)
axes[1].set_xlabel('F1-Score (macro)')
axes[1].set_title('F1-Score par modèle', fontweight='bold')
axes[1].set_xlim(0, 1)
for bar, val in zip(bars2, results_df['F1-Score (macro)']):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Matrices de Confusion

# %%
n_models = len(all_predictions)
cols = 3
rows = (n_models + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
axes = np.array(axes).ravel()

for idx, (model_name, pred_data) in enumerate(all_predictions.items()):
    cm = confusion_matrix(pred_data['labels'], pred_data['preds'])
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx], cbar=True, annot_kws={'size': 7})
    
    acc = all_results[model_name]['accuracy']
    axes[idx].set_title(f'{model_name} (Acc: {acc:.3f})', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Vrai')
    axes[idx].set_xlabel('Prédit')
    axes[idx].tick_params(axis='both', labelsize=6)
    plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right')
    plt.setp(axes[idx].get_yticklabels(), rotation=0)

# Masquer les axes vides
for idx in range(len(all_predictions), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Matrices de Confusion — Deep Learning", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrices_dl.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 11. Rapport de Classification Détaillé (Meilleur Modèle)

# %%
best_model_name = results_df.index[0]
best_preds = all_predictions[best_model_name]

print(f"📊 Meilleur modèle : {best_model_name}")
print(f"   Accuracy : {all_results[best_model_name]['accuracy']:.4f}")
print(f"   F1-macro : {all_results[best_model_name]['f1_macro']:.4f}")
print(f"\nRapport de classification détaillé :\n")
print(classification_report(
    best_preds['labels'], best_preds['preds'],
    target_names=class_names, zero_division=0
))

# %% [markdown]
# ## 12. Prédiction sur le Test Set et Soumission

# %%
print(f"🔮 Prédiction avec le meilleur modèle : {best_model_name}")

# Charger le meilleur modèle
checkpoint = torch.load(f"checkpoints/best_{best_model_name}.pth", map_location=device)
model_final = create_model(best_model_name, NUM_CLASSES, pretrained=False)
model_final.load_state_dict(checkpoint['model_state_dict'])
model_final = model_final.to(device)
model_final.eval()

# Adapter le test loader si Inception
if best_model_name == 'inception_v3':
    inception_test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_ds_final = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, inception_test_transform)
    test_loader_final = DataLoader(test_ds_final, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)
else:
    test_loader_final = test_loader

# Prédiction
all_test_preds = []
all_test_ids = []

with torch.no_grad():
    for images, ids in tqdm(test_loader_final, desc="Prédiction test"):
        images = images.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            outputs = model_final(images)
        
        _, predicted = outputs.max(1)
        all_test_preds.extend(predicted.cpu().numpy())
        all_test_ids.extend(ids)

# Convertir les indices en labels
test_labels_pred = [idx2label[p] for p in all_test_preds]

# %%
# Créer le fichier de soumission
submission = pd.DataFrame({
    test_df.columns[0]: all_test_ids,
    label_col: test_labels_pred
})

submission.to_csv("submission_dl.csv", index=False)
print(f"\n✅ Fichier de soumission créé : submission_dl.csv")
print(f"   Nombre de prédictions : {len(submission)}")
print(f"\nDistribution des prédictions :")
print(submission[label_col].value_counts())
print(f"\nAperçu :")
print(submission.head(10))

# %% [markdown]
# ## 13. Test-Time Augmentation (TTA) — Boost de Performance

# %%
def predict_with_tta(model, test_loader, device, n_augments=5):
    """
    Test-Time Augmentation : applique des augmentations aléatoires au test
    et moyenne les prédictions pour des résultats plus robustes.
    """
    model.eval()
    
    tta_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    # Prédiction standard (sans augmentation)
    all_logits = []
    all_ids = []
    
    # Pass 1 : sans augmentation
    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
            all_logits.append(outputs.cpu())
            if len(all_ids) == 0 or not isinstance(all_ids[0], str):
                all_ids.extend(ids)
    
    base_logits = torch.cat(all_logits, dim=0)
    avg_logits = base_logits.clone()
    
    # Passes TTA avec augmentations
    tta_test_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, tta_transforms)
    tta_loader = DataLoader(tta_test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    for aug_idx in range(n_augments):
        aug_logits = []
        with torch.no_grad():
            for images, _ in tta_loader:
                images = images.to(device, non_blocking=True)
                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    outputs = model(images)
                aug_logits.append(outputs.cpu())
        
        avg_logits += torch.cat(aug_logits, dim=0)
    
    # Moyenne des logits
    avg_logits /= (n_augments + 1)
    _, predicted = avg_logits.max(1)
    
    return predicted.numpy(), all_ids


# TTA avec le meilleur modèle
print(f"🔮 TTA (5 augmentations) avec {best_model_name}...")
tta_preds, tta_ids = predict_with_tta(model_final, test_loader_final, device, n_augments=5)

tta_labels = [idx2label[p] for p in tta_preds]
submission_tta = pd.DataFrame({
    test_df.columns[0]: tta_ids,
    label_col: tta_labels
})

submission_tta.to_csv("submission_dl_tta.csv", index=False)
print(f"\n✅ Soumission TTA créée : submission_dl_tta.csv")
print(f"Distribution :")
print(submission_tta[label_col].value_counts())

# %% [markdown]
# ## 14. Ensemble des Meilleurs Modèles

# %%
def ensemble_predict(model_names, test_loader, device, top_k=3):
    """
    Ensemble par vote majoritaire des top-k modèles.
    """
    print(f"🔮 Ensemble des {top_k} meilleurs modèles : {model_names[:top_k]}")
    
    all_model_preds = []
    
    for mname in model_names[:top_k]:
        ckpt_path = f"checkpoints/best_{mname}.pth"
        if not os.path.exists(ckpt_path):
            print(f"  ⚠️ Checkpoint {ckpt_path} non trouvé, skip.")
            continue
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = create_model(mname, NUM_CLASSES, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Adapter loader si inception
        if mname == 'inception_v3':
            inc_transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            inc_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, inc_transform)
            loader = DataLoader(inc_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
        else:
            loader = test_loader
        
        preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    outputs = model(images)
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())
        
        all_model_preds.append(preds)
        del model
        torch.cuda.empty_cache()
    
    # Vote majoritaire
    all_model_preds = np.array(all_model_preds)
    ensemble_preds = []
    for i in range(all_model_preds.shape[1]):
        votes = all_model_preds[:, i]
        ensemble_preds.append(np.bincount(votes, minlength=NUM_CLASSES).argmax())
    
    return np.array(ensemble_preds)


# Ensemble des 3 meilleurs modèles
top_models = results_df.index.tolist()
ensemble_preds = ensemble_predict(top_models, test_loader, device, top_k=3)
ensemble_labels = [idx2label[p] for p in ensemble_preds]

# IDs depuis le test loader
test_ids_list = test_df[test_df.columns[0]].values.tolist()

submission_ensemble = pd.DataFrame({
    test_df.columns[0]: test_ids_list[:len(ensemble_labels)],
    label_col: ensemble_labels
})

submission_ensemble.to_csv("submission_dl_ensemble.csv", index=False)
print(f"\n✅ Soumission Ensemble créée : submission_dl_ensemble.csv")
print(f"Distribution :")
print(submission_ensemble[label_col].value_counts())

# %% [markdown]
# ## 15. Résumé
# 
# | Modèle | Type | Réf. littérature | Particularités |
# |--------|------|------------------|----------------|
# | CNN Custom | From scratch | Healthcare 2022, Fig.9(a) | Conv+BN+Dropout, léger |
# | VGG16 | Transfer Learning | Healthcare 2022 ; Habibzadeh 2013 | Profond, 138M params |
# | ResNet50 | Transfer Learning | Habibzadeh et al. 2018 | Skip connections, 25M params |
# | DenseNet121 | Transfer Learning | Sharma et al. 2022 | Dense connections, 8M params |
# | InceptionV3 | Transfer Learning | Habibzadeh et al. 2018 | Multi-scale, aux loss |
# | MobileNetV2 | Transfer Learning | Cheuque et al. 2022 | Léger, rapide, depthwise conv |
# | EfficientNetB0 | Transfer Learning | État de l'art | Compound scaling, 5M params |
# 
# **Techniques d'optimisation utilisées :**
# - Mixed Precision (AMP/FP16) pour accélérer le training sur GPU
# - Discriminative Learning Rates (backbone vs head)
# - WeightedRandomSampler pour gérer le déséquilibre des classes
# - Data Augmentation agressive (rotation, flip, color jitter, cutout)
# - CosineAnnealingWarmRestarts scheduler
# - Early Stopping avec restauration du meilleur modèle
# - Test-Time Augmentation (TTA) pour booster les prédictions
# - Ensemble par vote majoritaire des top-k modèles
