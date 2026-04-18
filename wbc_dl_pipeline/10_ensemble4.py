# save as 10_ensemble_top3.py
import os, torch, numpy as np, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import torchvision.transforms as transforms, torchvision.models as models
from PIL import Image
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

BASE_DIR = "/home/infres/anadanak-24/projetkaggle/data/raw/IMA205-challenge 2"
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")
IMG_SIZE = 256; BATCH_SIZE = 16; NUM_WORKERS = 4; NUM_CLASSES = 13; N_TTA = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]

train_df = pd.read_csv(TRAIN_CSV); test_df = pd.read_csv(TEST_CSV)
label_col = train_df.columns[1]
class_names = sorted(train_df[label_col].unique())
idx2label = {idx: lbl for idx, lbl in enumerate(class_names)}

class WBCDataset(Dataset):
    def __init__(self, df, img_dir, id_col, transform):
        self.df = df.reset_index(drop=True); self.img_dir = img_dir
        self.id_col = id_col; self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_id = str(self.df.iloc[idx][self.id_col])
        if img_id.lower().endswith('.png'): p = os.path.join(self.img_dir, img_id)
        else: p = os.path.join(self.img_dir, f"{img_id}.png")
        try: img = Image.open(p).convert('RGB')
        except: return self.__getitem__((idx+1) % len(self.df))
        return self.transform(img), img_id

val_t = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
tta_t = transforms.Compose([transforms.Resize((IMG_SIZE+20,IMG_SIZE+20)), transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(180), transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(0.15,0.15,0.15,0.04), transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def create_swin(nc=13):
    m = models.swin_t(weights=None); inf = m.head.in_features
    m.head = nn.Sequential(nn.Dropout(0.3),nn.Linear(inf,512),nn.ReLU(True),nn.Dropout(0.2),nn.Linear(512,nc))
    return m
def create_convnext(nc=13):
    m = models.convnext_small(weights=None); inf = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(nn.Dropout(0.3),nn.Linear(inf,512),nn.ReLU(True),nn.Dropout(0.2),nn.Linear(512,nc))
    return m
def create_effnet(nc=13):
    m = models.efficientnet_b3(weights=None); inf = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3),nn.Linear(inf,512),nn.ReLU(True),nn.Dropout(0.2),nn.Linear(512,nc))
    return m

# TOP 3 : Swin (0.7675) + ConvNeXt (0.7616) + EfficientNet V4 (0.7473)
top3 = [
    ("Swin-T",      "/home/infres/anadanak-24/projetkaggle/checkpoints/best_swin_v4_final.pth",     create_swin),
    ("ConvNeXt",    "/home/infres/anadanak-24/projetkaggle/checkpoints/best_convnext_v4_final.pth",  create_convnext),
    ("EffNet V4",   "/home/infres/anadanak-24/projetkaggle/checkpoints/best_v4_final.pth",           create_effnet),
]

@torch.no_grad()
def predict_tta(model, n_tta=N_TTA):
    model.eval(); all_probs=None; all_ids=None
    for p in range(n_tta+1):
        t = val_t if p==0 else tta_t
        ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        bp, bi = [], []
        for imgs, ids in tqdm(loader, desc=f"  TTA {p}/{n_tta}", leave=False, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(enabled=True): out = model(imgs)
            bp.append(F.softmax(out,dim=1).cpu())
            if p==0: bi.extend(ids)
        probs = torch.cat(bp,dim=0)
        if all_probs is None: all_probs=probs; all_ids=bi
        else: all_probs += probs
    return all_ids, all_probs / (n_tta+1)

# Weighted ensemble (poids = score Kaggle)
weights = [0.7675, 0.7616, 0.7473]
ens_probs = None; ens_ids = None

for (name, path, create_fn), w in zip(top3, weights):
    print(f"\n🔍 {name} (poids={w})...")
    model = create_fn(NUM_CLASSES)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    ids, probs = predict_tta(model)
    if ens_probs is None: ens_probs = probs * w; ens_ids = ids
    else: ens_probs += probs * w
    del model; torch.cuda.empty_cache()

ens_probs /= sum(weights)
_, preds = ens_probs.max(1)
labels = [idx2label[p.item()] for p in preds]

os.makedirs("submissions", exist_ok=True)
sub = pd.DataFrame({test_df.columns[0]: ens_ids, label_col: labels})
sub.to_csv("submissions/submission_ensemble_top3_weighted.csv", index=False)
print(f"\n✅ submissions/submission_ensemble_top3_weighted.csv")
print(sub[label_col].value_counts().to_string())