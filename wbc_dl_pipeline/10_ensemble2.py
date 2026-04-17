# %% [markdown]
# # 🏆 Ensemble — Soft Voting des checkpoints disponibles
#
# Moyenne pondérée des probabilités softmax + TTA

# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🖥️  GPU : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# %% [markdown]
# ## Config

# %%
BASE_DIR = "/home/infres/anadanak-24/projetkaggle/data/raw/IMA205-challenge 2"
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")

IMG_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 13
N_TTA = 10

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

os.makedirs("submissions", exist_ok=True)

# %% [markdown]
# ## Checkpoints disponibles

# %%
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

label_col = train_df.columns[1]
class_names = sorted(train_df[label_col].unique())
label2idx = {lbl: idx for idx, lbl in enumerate(class_names)}
idx2label = {idx: lbl for lbl, idx in label2idx.items()}

# Détecter les checkpoints
CKPT_DIR = "/home/infres/anadanak-24/projetkaggle/checkpoints/"
candidates = [
    ("V3",             "best_v3.pth"),
    ("V2",             "best_v2.pth"),
    ("B3 Optim",       "best_efficientnet_b3_optimized.pth"),
    ("V4 Final",       "best_v4_final.pth"),
    ("V4 Stage1",      "best_v4_stage1.pth"),
    ("ResNet50 V4",    "best_resnet50_v4_final.pth"),
    ("ResNet50 V4 S1", "best_resnet50_v4_stage1.pth"),
]

available = []
for name, fname in candidates:
    path = os.path.join(CKPT_DIR, fname)
    if os.path.exists(path):
        available.append((name, path))
        print(f"  ✅ {name:15s} → {fname}")

print(f"\n📊 {len(available)} modèles pour l'ensemble")

# %% [markdown]
# ## Modèle & Dataset

# %%
def create_efficientnet_b3(num_classes=13):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


def create_resnet50(num_classes=13):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    return model


def create_efficientnet_b3_v1(num_classes=13):
    """Head V1/V2 : Dropout → Linear → BatchNorm → ReLU → Dropout → Linear."""
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def load_model_from_checkpoint(ckpt_path, num_classes=13):
    """Charge un modèle en détectant l'architecture ET le head automatiquement."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    
    # Détecter ResNet vs EfficientNet
    is_resnet = any('layer4' in k for k in state.keys())
    
    if is_resnet:
        model = create_resnet50(num_classes)
        arch = "ResNet50"
    else:
        # Détecter quel head : V1/V2 a 'classifier.2.weight' (BatchNorm), V3/V4 n'en a pas
        has_batchnorm_head = 'classifier.2.weight' in state and 'classifier.2.running_mean' in state
        if has_batchnorm_head:
            model = create_efficientnet_b3_v1(num_classes)
            arch = "EfficientNet-B3 (head V1/V2)"
        else:
            model = create_efficientnet_b3(num_classes)
            arch = "EfficientNet-B3 (head V3/V4)"
    
    model.load_state_dict(state)
    print(f"     Architecture détectée : {arch}")
    return model


class WBCDataset(Dataset):
    def __init__(self, df, img_dir, id_col, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.id_col = id_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx][self.id_col])
        if img_id.lower().endswith('.png'):
            img_path = os.path.join(self.img_dir, img_id)
        else:
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            return self.__getitem__((idx + 1) % len(self.df))
        if self.transform:
            image = self.transform(image)
        return image, img_id


val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

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

# %% [markdown]
# ## Prédiction avec TTA

# %%
@torch.no_grad()
def predict_tta(model, test_df, n_tta=N_TTA):
    """Retourne (ids, probs_moyennées)."""
    model.eval()
    all_probs = None
    all_ids = None

    for tta_pass in range(n_tta + 1):
        t = val_transform if tta_pass == 0 else tta_transform
        ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

        batch_probs, batch_ids = [], []
        for images, ids in tqdm(loader, desc=f"  TTA {tta_pass}/{n_tta}", leave=False, ncols=100):
            images = images.to(device, non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
            batch_probs.append(F.softmax(outputs, dim=1).cpu())
            if tta_pass == 0:
                batch_ids.extend(ids)

        probs = torch.cat(batch_probs, dim=0)
        if all_probs is None:
            all_probs = probs
            all_ids = batch_ids
        else:
            all_probs += probs

    all_probs /= (n_tta + 1)
    return all_ids, all_probs

# %% [markdown]
# ## Ensemble

# %%
print(f"\n{'='*60}")
print(f"  🏆 ENSEMBLE SOFT VOTING — {len(available)} modèles × TTA x{N_TTA}")
print(f"{'='*60}")

ensemble_probs = None
ensemble_ids = None

for name, ckpt_path in available:
    print(f"\n🔍 {name}...")

    model = load_model_from_checkpoint(ckpt_path, NUM_CLASSES)
    model = model.to(device)

    ids, probs = predict_tta(model, test_df, n_tta=N_TTA)

    if ensemble_probs is None:
        ensemble_probs = probs.clone()
        ensemble_ids = ids
    else:
        ensemble_probs += probs

    print(f"  ✅ {name} done")
    del model
    torch.cuda.empty_cache()

# Moyenne
ensemble_probs /= len(available)

# Prédictions
_, ensemble_preds = ensemble_probs.max(1)
ensemble_labels = [idx2label[p.item()] for p in ensemble_preds]

# %% [markdown]
# ## Soumissions

# %%
# 1. Ensemble de tous les modèles
sub = pd.DataFrame({
    test_df.columns[0]: ensemble_ids,
    label_col: ensemble_labels
})
sub.to_csv("submissions/submission_ensemble_all.csv", index=False)
print(f"\n✅ submissions/submission_ensemble_all.csv")
print(sub[label_col].value_counts().to_string())

# %%
# 2. Si on a V3 + V4, faire aussi un ensemble juste V3+V4 (les meilleurs)
if len(available) >= 2:
    # Refaire avec seulement les 2 meilleurs
    # On prend V3 et le dernier (qui est probablement le meilleur)
    best_two = [available[0], available[-1]] if len(available) > 2 else available[:2]

    print(f"\n📊 Ensemble des 2 meilleurs : {[n for n,_ in best_two]}")
    probs_2 = None
    ids_2 = None

    for name, ckpt_path in best_two:
        model = load_model_from_checkpoint(ckpt_path, NUM_CLASSES)
        model = model.to(device)
        ids, probs = predict_tta(model, test_df, n_tta=N_TTA)
        if probs_2 is None:
            probs_2 = probs.clone()
            ids_2 = ids
        else:
            probs_2 += probs
        del model
        torch.cuda.empty_cache()

    probs_2 /= 2
    _, preds_2 = probs_2.max(1)
    labels_2 = [idx2label[p.item()] for p in preds_2]

    sub2 = pd.DataFrame({
        test_df.columns[0]: ids_2,
        label_col: labels_2
    })
    sub2.to_csv("submissions/submission_ensemble_top2.csv", index=False)
    print(f"✅ submissions/submission_ensemble_top2.csv")
    print(sub2[label_col].value_counts().to_string())

# %% [markdown]
# ## Résumé
#
# Trois soumissions générées :
# - `submission_ensemble_all.csv` — moyenne des 3 modèles (recommandé)
# - `submission_ensemble_top2.csv` — moyenne des 2 meilleurs
#
# **Soumets les deux et compare les scores !**
# L'ensemble de 3 devrait être le meilleur car les modèles ont été
# entraînés avec des configs différentes → erreurs complémentaires.