# %% [markdown]
# # ResNet50 (Transfer Learning)
# 
# **Référence littérature** : Habibzadeh et al. 2018 — 99.84% accuracy rapportée
# 
# **Notes** :
# - Skip connections, 25M params — souvent le meilleur compromis.
# - LR 1e-4 standard pour fine-tuning.
# - Batch size 32 (ajuster selon VRAM dispo).

# %% [markdown]
# ## 1. Setup

# %%
# Import du module commun
from utils import *

# Configuration
device = setup_device()
setup_seed(42)
create_dirs()

MODEL_NAME = "resnet50"
print(f"\n🎯 Modèle : {MODEL_NAME}")

# %% [markdown]
# ## 2. Hyperparamètres spécifiques à ce modèle

# %%
IMG_SIZE_LOCAL   = 224
BATCH_SIZE_LOCAL = 32
LR_LOCAL         = 0.0001
EPOCHS_LOCAL     = 30

print(f"  img_size   : {IMG_SIZE_LOCAL}")
print(f"  batch_size : {BATCH_SIZE_LOCAL}")
print(f"  lr         : {LR_LOCAL}")
print(f"  epochs     : {EPOCHS_LOCAL}")

# %% [markdown]
# ## 3. Chargement des données

# %%
(train_loader, val_loader, test_loader,
 class_names, label2idx, idx2label,
 train_subset, val_subset, test_df,
 weight_tensor) = prepare_data(
    img_size=IMG_SIZE_LOCAL,
    batch_size=BATCH_SIZE_LOCAL,
)

print(f"\n  Classes ({len(class_names)}) : {class_names}")

# %% [markdown]
# ## 4. Vérification du modèle

# %%
test_model = create_model(MODEL_NAME, NUM_CLASSES, pretrained=True)
n_params = sum(p.numel() for p in test_model.parameters()) / 1e6
n_trainable = sum(p.numel() for p in test_model.parameters() if p.requires_grad) / 1e6
print(f"  Total params      : {n_params:.2f} M")
print(f"  Trainable params  : {n_trainable:.2f} M")
del test_model

# %% [markdown]
# ## 5. Entraînement

# %%
model, history, val_preds, val_labels = train_model(
    model_name=MODEL_NAME,
    train_loader=train_loader,
    val_loader=val_loader,
    weight_tensor=weight_tensor,
    device=device,
    num_epochs=EPOCHS_LOCAL,
    lr=LR_LOCAL,
    freeze_backbone=False,
    save_best=True,
)

# %% [markdown]
# ## 6. Courbes d'apprentissage

# %%
plot_history(history, MODEL_NAME, save_path=f"{MODEL_NAME}_curves.png")

# %% [markdown]
# ## 7. Évaluation sur la validation

# %%
print_classification_report(val_labels, val_preds, class_names)

# %%
plot_confusion_matrix(val_labels, val_preds, class_names, MODEL_NAME,
                      save_path=f"{MODEL_NAME}_cm.png")

# %% [markdown]
# ## 8. Prédiction sur le test set + soumission

# %%
test_ids, test_labels_pred = predict_test(model, test_df, idx2label, device, MODEL_NAME)

# Adapter les noms de colonnes selon le format de soumission
sub = save_submission(
    ids=test_ids,
    labels=test_labels_pred,
    model_name=MODEL_NAME,
    label_col=test_df.columns[1] if len(test_df.columns) > 1 else "label",
    id_col=test_df.columns[0],
)

# %% [markdown]
# ## 9. Libérer la VRAM

# %%
del model
import torch
torch.cuda.empty_cache()
print("✅ Mémoire GPU libérée")
