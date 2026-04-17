# %% [markdown]
# # 🏆 Swin Transformer Tiny — Config V4 (Oversampling + 2-Stage)
# Vision Transformer avec attention spatiale. Très différent des CNN.

# %%
import os, time, copy, random
import numpy as np, pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm
import warnings; warnings.filterwarnings('ignore')

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms, torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"🖥️  GPU : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

# %% Config
BASE_DIR = "/home/infres/anadanak-24/projetkaggle/data/raw/IMA205-challenge 2"
TRAIN_DIR = os.path.join(BASE_DIR, "train"); TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv"); TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")
IMG_SIZE=256; BATCH_SIZE=24; NUM_WORKERS=4; LABEL_SMOOTHING=0.1; WEIGHT_DECAY=1e-4
VAL_SPLIT=0.15; NUM_CLASSES=13; OVERSAMPLE_MIN=500
S1_EPOCHS=35; S1_LR_HEAD=2e-4; S1_LR_BB=2e-5; S1_FREEZE=3; S1_PAT=8
S2_EPOCHS=15; S2_LR=1e-5; S2_PAT=5
IMAGENET_MEAN=[0.485,0.456,0.406]; IMAGENET_STD=[0.229,0.224,0.225]
os.makedirs("checkpoints",exist_ok=True); os.makedirs("submissions",exist_ok=True)
print(f"✅ Swin Transformer Tiny V4 : IMG={IMG_SIZE}, BS={BATCH_SIZE}")

# %% Data + Oversampling
train_df=pd.read_csv(TRAIN_CSV); test_df=pd.read_csv(TEST_CSV)
id_col=train_df.columns[0]; label_col=train_df.columns[1]
class_names=sorted(train_df[label_col].unique())
label2idx={l:i for i,l in enumerate(class_names)}; idx2label={i:l for l,i in label2idx.items()}
train_df['label_idx']=train_df[label_col].map(label2idx)
train_idx,val_idx=train_test_split(np.arange(len(train_df)),test_size=VAL_SPLIT,random_state=SEED,stratify=train_df['label_idx'].values)
train_raw=train_df.iloc[train_idx].copy(); val_sub=train_df.iloc[val_idx].copy()

def oversample(df,lc,mn):
    ps=[]
    for c in df[lc].unique():
        d=df[df[lc]==c]
        ps.append(d if len(d)>=mn else pd.concat([d,d.sample(n=mn-len(d),replace=True,random_state=SEED)]))
    return pd.concat(ps).sample(frac=1,random_state=SEED).reset_index(drop=True)

train_sub=oversample(train_raw,label_col,OVERSAMPLE_MIN)
print(f"📊 Train: {len(train_raw)}→{len(train_sub)} | Val: {len(val_sub)}")

# %% Transforms & Dataset
train_t=transforms.Compose([transforms.Resize((IMG_SIZE+20,IMG_SIZE+20)),transforms.RandomCrop(IMG_SIZE),transforms.RandomRotation(180),transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.5),transforms.RandomAffine(0,translate=(0.08,0.08),scale=(0.85,1.15)),transforms.ColorJitter(0.25,0.25,0.25,0.06),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),transforms.RandomErasing(p=0.1,scale=(0.02,0.1))])
val_t=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])

class DS(Dataset):
    def __init__(s,df,d,ic,lc=None,t=None): s.df=df.reset_index(drop=True);s.d=d;s.ic=ic;s.lc=lc;s.t=t
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]; n=str(r[s.ic]); n=n if n.lower().endswith('.png') else f"{n}.png"
        try: img=Image.open(os.path.join(s.d,n)).convert('RGB')
        except: return s.__getitem__((i+1)%len(s.df))
        if s.t: img=s.t(img)
        return (img,r[s.lc]) if s.lc else (img,str(r[s.ic]))

# %% Loaders
tds=DS(train_sub,TRAIN_DIR,id_col,'label_idx',train_t); vds=DS(val_sub,TRAIN_DIR,id_col,'label_idx',val_t)
test_ds=DS(test_df,TEST_DIR,test_df.columns[0],None,val_t)
tl=train_sub['label_idx'].values; cc=Counter(tl); ts=sum(cc.values())
rw=[ts/cc[i] for i in range(NUM_CLASSES)]; sw=[np.sqrt(w) for w in rw]; mw=np.mean(sw)
sw=[w/mw for w in sw]; wt=torch.FloatTensor(sw).to(device)
smpw=[np.sqrt(ts/cc[l]) for l in tl]; smp=WeightedRandomSampler(smpw,len(smpw),replacement=True)
train_ld=DataLoader(tds,batch_size=BATCH_SIZE,sampler=smp,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
val_ld=DataLoader(vds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)
test_ld=DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)

# %% Model
def make_model(nc=13,pt=True):
    m=models.swin_t(weights='IMAGENET1K_V1' if pt else None)
    inf=m.head.in_features
    m.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(inf,512),nn.ReLU(True),nn.Dropout(0.2),nn.Linear(512,nc))
    return m
model=make_model(NUM_CLASSES).to(device)
print(f"✅ Swin Transformer Tiny: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

# %% Train utils
def freeze(m,f=True):
    for n,p in m.named_parameters():
        if 'head' not in n: p.requires_grad=not f
    print(f"   {'🔒' if f else '🔓'} Trainable: {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.1f}M")

def mk_opt(m,lrb,lrh):
    b,h=[],[]
    for n,p in m.named_parameters():
        if p.requires_grad: (h if 'head' in n else b).append(p)
    g=[{'params':h,'lr':lrh}]
    if b: g.insert(0,{'params':b,'lr':lrb})
    return optim.AdamW(g,weight_decay=WEIGHT_DECAY)

def trn(m,ld,cr,op,sc,dv,ep):
    m.train(); rl,c,t=0.,0,0
    pb=tqdm(ld,desc=f"Ep{ep:02d}[T]",leave=False,ncols=120,bar_format='{l_bar}{bar:30}{r_bar}')
    for x,y in pb:
        x,y=x.to(dv,non_blocking=True),y.to(dv,non_blocking=True)
        op.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()): o=m(x);l=cr(o,y)
        sc.scale(l).backward();sc.unscale_(op);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);sc.step(op);sc.update()
        rl+=l.item()*x.size(0);_,p=o.max(1);t+=y.size(0);c+=p.eq(y).sum().item()
        pb.set_postfix(loss=f'{rl/t:.4f}',acc=f'{c/t:.4f}')
    return rl/t,c/t

@torch.no_grad()
def val(m,ld,cr,dv,ep):
    m.eval();rl,c,t=0.,0,0;ap,al=[],[]
    pb=tqdm(ld,desc=f"Ep{ep:02d}[V]",leave=False,ncols=120,bar_format='{l_bar}{bar:30}{r_bar}')
    for x,y in pb:
        x,y=x.to(dv,non_blocking=True),y.to(dv,non_blocking=True)
        with autocast(enabled=torch.cuda.is_available()): o=m(x);l=cr(o,y)
        rl+=l.item()*x.size(0);_,p=o.max(1);t+=y.size(0);c+=p.eq(y).sum().item()
        ap.extend(p.cpu().numpy());al.extend(y.cpu().numpy())
        pb.set_postfix(loss=f'{rl/t:.4f}',acc=f'{c/t:.4f}')
    p,l=np.array(ap),np.array(al)
    return rl/t,c/t,f1_score(l,p,average='macro',zero_division=0),p,l

def stage(m,tld,vld,cr,dv,sc,ne,lrb,lrh,pat,nm,fe=0,bf=0.):
    print(f"\n{'='*70}\n  {nm}\n{'='*70}\n")
    if fe>0: freeze(m,True)
    else: freeze(m,False)
    op=mk_opt(m,lrb,lrh);sd=None;bs=copy.deepcopy(m.state_dict());pc=0
    for ep in range(1,ne+1):
        t0=time.time()
        if fe>0 and ep==fe+1: print(f"\n  🔓 Dégel");freeze(m,False);op=mk_opt(m,lrb,lrh);sd=CosineAnnealingLR(op,T_max=ne-fe,eta_min=1e-7)
        tl_,ta=trn(m,tld,cr,op,sc,dv,ep);vl_,va,vf,vp,vlb=val(m,vld,cr,dv,ep)
        if sd and ep>fe: sd.step()
        dt=time.time()-t0;lr=op.param_groups[-1]['lr'];bst='🏆' if vf>bf else '';ph='FRZ' if ep<=fe else 'FUL'
        vm=f"|VRAM:{torch.cuda.max_memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else ""
        print(f"  [{ph}] Ep{ep:02d}/{ne} | T:{tl_:.3f}/{ta:.3f} | V:{vl_:.3f}/{va:.3f}/F1:{vf:.4f} | LR:{lr:.1e} | {dt:.0f}s {vm} {bst}")
        if vf>bf: bf=vf;bs=copy.deepcopy(m.state_dict());pc=0
        else: pc+=1
        if ep>fe+3 and pc>=pat: print(f"\n  ⏹️ Early stop");break
    m.load_state_dict(bs);print(f"\n  ✅ Best F1: {bf:.4f}");return m,bf,vp,vlb

# %% Stage 1 & 2
crit=nn.CrossEntropyLoss(weight=wt,label_smoothing=LABEL_SMOOTHING); sc=GradScaler(enabled=torch.cuda.is_available())
model,f1s1,_,_=stage(model,train_ld,val_ld,crit,device,sc,S1_EPOCHS,S1_LR_BB,S1_LR_HEAD,S1_PAT,"📗 STAGE1: Swin",S1_FREEZE)
torch.save({'model_state_dict':model.state_dict(),'val_f1':f1s1,'model_name':'swin_tiny'},'checkpoints/best_swin_v4_stage1.pth')

s2w=[(ts/cc[i])**0.75 for i in range(NUM_CLASSES)];s2m=np.mean(s2w);s2w=[w/s2m for w in s2w]
s2wt=torch.FloatTensor(s2w).to(device);s2sw=[(ts/cc[l])**0.75 for l in tl]
s2smp=WeightedRandomSampler(s2sw,len(s2sw),replacement=True)
s2ld=DataLoader(tds,batch_size=BATCH_SIZE,sampler=s2smp,num_workers=NUM_WORKERS,pin_memory=True,drop_last=True)
s2cr=nn.CrossEntropyLoss(weight=s2wt,label_smoothing=LABEL_SMOOTHING)
model,f1s2,vp,vl=stage(model,s2ld,val_ld,s2cr,device,sc,S2_EPOCHS,S2_LR,S2_LR*2,S2_PAT,"📘 STAGE2: Swin rares",0,f1s1)
torch.save({'model_state_dict':model.state_dict(),'val_f1':max(f1s1,f1s2),'model_name':'swin_tiny'},'checkpoints/best_swin_v4_final.pth')
print(f"\n📊 Swin F1: S1={f1s1:.4f} S2={f1s2:.4f} Best={max(f1s1,f1s2):.4f}")
print(classification_report(vl,vp,target_names=class_names,zero_division=0))

# %% TTA
print("\n📮 TTA x10...")
tta_t=transforms.Compose([transforms.Resize((IMG_SIZE+20,IMG_SIZE+20)),transforms.RandomCrop(IMG_SIZE),transforms.RandomRotation(180),transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.5),transforms.ColorJitter(0.15,0.15,0.15,0.04),transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])
model.eval();N=10;ap=None;ti=None
for p in range(N+1):
    t=val_t if p==0 else tta_t
    ds=DS(test_df,TEST_DIR,test_df.columns[0],None,t);ld=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True)
    bp,bi=[],[]
    with torch.no_grad():
        for x,ids in tqdm(ld,desc=f"TTA{p}/{N}",leave=False,ncols=100):
            x=x.to(device,non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()): o=model(x)
            bp.append(F.softmax(o,dim=1).cpu())
            if p==0: bi.extend(ids)
    pr=torch.cat(bp,0)
    if ap is None: ap=pr;ti=bi
    else: ap+=pr
ap/=(N+1);_,tp=ap.max(1)
sub=pd.DataFrame({test_df.columns[0]:ti,label_col:[idx2label[x.item()] for x in tp]})
sub.to_csv("submissions/submission_swin_v4_tta.csv",index=False)
print(f"✅ submissions/submission_swin_v4_tta.csv")
print(sub[label_col].value_counts().to_string())