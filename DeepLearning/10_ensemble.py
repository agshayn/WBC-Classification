# 10_ensemble.py — Ensemble pondéré top-3 (Swin + ConvNeXt + EfficientNet-B3)
import os
import torch
import torch.nn as nn
import pandas as pd
from utils_v4 import (
    setup, create_model, predict_tta, save_submission,
    TRAIN_CSV, TEST_CSV, TEST_DIR, NUM_CLASSES, N_TTA,
)

device = setup()

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
label_col = train_df.columns[1]
class_names = sorted(train_df[label_col].unique())
idx2label = {idx: lbl for idx, lbl in enumerate(class_names)}

# TOP 3 : Swin (0.7675) + ConvNeXt (0.7616) + EfficientNet V4 (0.7473)
top3 = [
    ("swin_tiny",       "checkpoints/best_swin_tiny_v4_final.pth",       0.7675, False),
    ("convnext_small",  "checkpoints/best_convnext_small_v4_final.pth",  0.7616, False),
    ("efficientnet_b3", "checkpoints/best_efficientnet_b3_v4_final.pth", 0.7473, True),
]
# (model_name, checkpoint_path, poids_kaggle, use_affine_tta)

ens_probs = None
ens_ids = None
weights = []

for model_name, ckpt_path, w, use_affine in top3:
    print(f"\n🔍 {model_name} (poids={w})...")

    model, _ = create_model(model_name, NUM_CLASSES, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    ids, _, probs = predict_tta(
        model, test_df, device, idx2label, label_col,
        batch_size=16, n_tta=N_TTA, use_affine=use_affine,
    )

    if ens_probs is None:
        ens_probs = probs * w
        ens_ids = ids
    else:
        ens_probs += probs * w
    weights.append(w)

    del model
    torch.cuda.empty_cache()

# Moyenne pondérée
ens_probs /= sum(weights)
_, preds = ens_probs.max(1)
labels = [idx2label[p.item()] for p in preds]

save_submission(ens_ids, labels, test_df, label_col,
                "submission_ensemble_top3_weighted.csv")