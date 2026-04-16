import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from collections import Counter

# === ADAPTER ===
BASE_DIR = "/home/infres/anadanak-24/projetkaggle/data/raw/IMA205-challenge 2"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")

df = pd.read_csv(TRAIN_CSV)
id_col = df.columns[0]
label_col = df.columns[1]

print("=== CSV ===")
print(df.head(10))
print(f"\nTypes des colonnes : {df.dtypes.to_dict()}")
print(f"\nValeurs uniques label : {sorted(df[label_col].unique())}")
print(f"\nDistribution :")
print(df[label_col].value_counts())

# Vérifier quelques images
print("\n=== IMAGES ===")
for i in range(5):
    img_id = str(df.iloc[i][id_col])
    if not img_id.lower().endswith('.png'):
        img_id += '.png'
    path = os.path.join(TRAIN_DIR, img_id)
    if os.path.exists(path):
        img = Image.open(path)
        arr = np.array(img)
        print(f"  {img_id}: shape={arr.shape}, dtype={arr.dtype}, "
              f"min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, "
              f"label={df.iloc[i][label_col]}")
    else:
        print(f"  {img_id}: MANQUANT")

# Vérifier que les labels sont bien des entiers ou strings cohérents
print(f"\n=== LABELS ===")
print(f"Type du label : {type(df.iloc[0][label_col])}")
print(f"Premiers labels : {df[label_col].head(20).tolist()}")

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 13

# Simple transform
t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class_names = sorted(df[label_col].unique())
label2idx = {lbl: idx for idx, lbl in enumerate(class_names)}
df['label_idx'] = df[label_col].map(label2idx)

# Split
train_idx, val_idx = train_test_split(
    np.arange(len(df)), test_size=0.15, random_state=42,
    stratify=df['label_idx'].values
)

# Check que les labels val sont bien distribués
val_labels = df.iloc[val_idx]['label_idx'].values
print("\n=== VAL DISTRIBUTION ===")
print(Counter(val_labels))

# Charger 1 batch val et faire un forward pass
print("\n=== FORWARD TEST ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.efficientnet_b3(weights='IMAGENET1K_V1')
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)
model.eval()

# Charger 5 images val
images_batch = []
labels_batch = []
for i in val_idx[:5]:
    row = df.iloc[i]
    img_id = str(row[id_col])
    if not img_id.lower().endswith('.png'):
        img_id += '.png'
    img = Image.open(os.path.join(TRAIN_DIR, img_id)).convert('RGB')
    images_batch.append(t(img))
    labels_batch.append(row['label_idx'])

x = torch.stack(images_batch).to(device)
y = torch.tensor(labels_batch)

with torch.no_grad():
    logits = model(x)

print(f"Logits shape : {logits.shape}")
print(f"Logits range : min={logits.min().item():.3f}, max={logits.max().item():.3f}")
print(f"Logits std   : {logits.std().item():.3f}")
probs = torch.softmax(logits, dim=1)
preds = probs.argmax(dim=1)
print(f"\nVrais labels  : {y.tolist()}")
print(f"Prédictions   : {preds.cpu().tolist()}")
print(f"Max proba     : {probs.max(dim=1).values.cpu().tolist()}")
print(f"\nExemple probs image 0 :")
for j, cn in enumerate(class_names):
    print(f"  {cn}: {probs[0, j].item():.4f}")

# Test avec simple CrossEntropy (sans focal, sans weights)
criterion_simple = torch.nn.CrossEntropyLoss()
loss_simple = criterion_simple(logits, y.to(device))
print(f"\nSimple CE loss : {loss_simple.item():.4f}")

# Test avec weighted CE
train_labels_all = df.iloc[train_idx]['label_idx'].values
cc = Counter(train_labels_all)
total = sum(cc.values())
w = torch.FloatTensor([total / cc[i] for i in range(NUM_CLASSES)])
w = w / w.sum() * NUM_CLASSES
print(f"\nClass weights  : {[f'{v:.2f}' for v in w.tolist()]}")
criterion_w = torch.nn.CrossEntropyLoss(weight=w.to(device))
loss_w = criterion_w(logits, y.to(device))
print(f"Weighted CE loss: {loss_w.item():.4f}")