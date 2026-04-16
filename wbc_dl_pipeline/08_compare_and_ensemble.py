# %% [markdown]
# # 08 — Comparaison & Ensemble des Modèles
# 
# Ce notebook charge TOUS les checkpoints des modèles entraînés (notebooks 01-07),
# compare leurs performances, et génère :
# - Une comparaison globale (accuracy, F1)
# - Une matrice de confusion par modèle
# - Un ensemble par vote majoritaire des top-k modèles
# - Un ensemble par moyenne des probabilités (soft voting)

# %% [markdown]
# ## 1. Setup

# %%
from utils import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = setup_device()
setup_seed(42)

# %% [markdown]
# ## 2. Chargement des données (test uniquement)

# %%
(_, val_loader, test_loader,
 class_names, label2idx, idx2label,
 _, val_subset, test_df,
 weight_tensor) = prepare_data()

print(f"Classes : {class_names}")

# %% [markdown]
# ## 3. Détection des modèles entraînés

# %%
import os

ALL_MODELS = ['custom_cnn', 'vgg16', 'resnet50', 'densenet121',
              'inception_v3', 'mobilenet_v2', 'efficientnet_b0']

available_models = []
for m in ALL_MODELS:
    ckpt_path = os.path.join(CHECKPOINTS_DIR, f"best_{m}.pth")
    if os.path.exists(ckpt_path):
        available_models.append(m)
        print(f"  ✅ {m}")
    else:
        print(f"  ❌ {m} (pas de checkpoint)")

print(f"\n📊 {len(available_models)} modèles disponibles pour comparaison.")

# %% [markdown]
# ## 4. Évaluation de chaque modèle sur la validation

# %%
import torch.nn as nn
from torch.cuda.amp import autocast

results = {}
all_val_preds = {}
all_val_probs = {}  # Pour soft voting

criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))

for model_name in available_models:
    print(f"\n🔍 Évaluation : {model_name}")
    
    # Charger le modèle
    model, ckpt = load_checkpoint(model_name, device)
    model.eval()
    
    # Adapter le val loader pour Inception (299x299)
    if model_name == 'inception_v3':
        from torch.utils.data import DataLoader
        _, inc_transform = get_transforms(299)
        inc_val_ds = WBCDataset(val_subset, TRAIN_DIR, val_subset.columns[0],
                                'label_idx', inc_transform)
        current_val_loader = DataLoader(inc_val_ds, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=NUM_WORKERS,
                                        pin_memory=True)
    else:
        current_val_loader = val_loader
    
    # Validation + récupération des logits
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in current_val_loader:
            images = images.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)
    
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    
    results[model_name] = {'accuracy': acc, 'f1_macro': f1, 'val_acc_ckpt': ckpt['val_acc']}
    all_val_preds[model_name] = preds
    all_val_probs[model_name] = probs
    
    print(f"  Acc (live)     : {acc:.4f}")
    print(f"  F1 (macro)     : {f1:.4f}")
    print(f"  Acc (checkpoint): {ckpt['val_acc']:.4f}")
    
    # Sauvegarder les vrais labels une fois
    if 'true_labels' not in dir() or true_labels is None:
        true_labels = labels
    
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. Tableau récapitulatif

# %%
results_df = pd.DataFrame(results).T
results_df.columns = ['Accuracy (live)', 'F1-Score (macro)', 'Accuracy (ckpt)']
results_df = results_df.sort_values('Accuracy (live)', ascending=False)
print("\n📊 Comparaison globale :")
print(results_df.to_string())

# %% [markdown]
# ## 6. Visualisation comparative

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results_df)))

bars1 = axes[0].barh(results_df.index, results_df['Accuracy (live)'], color=colors)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Accuracy par modèle', fontweight='bold')
axes[0].set_xlim(0, 1)
for bar, val in zip(bars1, results_df['Accuracy (live)']):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

bars2 = axes[1].barh(results_df.index, results_df['F1-Score (macro)'], color=colors)
axes[1].set_xlabel('F1-Score (macro)')
axes[1].set_title('F1-Score par modèle', fontweight='bold')
axes[1].set_xlim(0, 1)
for bar, val in zip(bars2, results_df['F1-Score (macro)']):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig("comparison_all_models.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Matrices de confusion comparatives

# %%
n_models = len(all_val_preds)
cols = 3
rows = (n_models + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
axes = np.array(axes).ravel()

for idx, (model_name, preds) in enumerate(all_val_preds.items()):
    cm = confusion_matrix(true_labels, preds)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx], cbar=True, annot_kws={'size': 7})
    axes[idx].set_title(f'{model_name} (Acc: {results[model_name]["accuracy"]:.3f})',
                        fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Vrai')
    axes[idx].set_xlabel('Prédit')
    plt.setp(axes[idx].get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(axes[idx].get_yticklabels(), rotation=0, fontsize=7)

for idx in range(len(all_val_preds), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle("Matrices de Confusion — Tous les Modèles", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("all_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 8. Ensemble — Hard Voting (vote majoritaire) sur la validation

# %%
TOP_K = 3  # Nombre de meilleurs modèles à combiner

top_k_models = results_df.head(TOP_K).index.tolist()
print(f"🤝 Top-{TOP_K} modèles : {top_k_models}")

# Hard voting
all_preds_array = np.array([all_val_preds[m] for m in top_k_models])
hard_voting_preds = np.array([
    np.bincount(all_preds_array[:, i], minlength=NUM_CLASSES).argmax()
    for i in range(all_preds_array.shape[1])
])

acc_hard = accuracy_score(true_labels, hard_voting_preds)
f1_hard = f1_score(true_labels, hard_voting_preds, average='macro', zero_division=0)
print(f"\n🗳️  Hard Voting Top-{TOP_K} :")
print(f"   Accuracy : {acc_hard:.4f}")
print(f"   F1-macro : {f1_hard:.4f}")

# %% [markdown]
# ## 9. Ensemble — Soft Voting (moyenne des probabilités) sur la validation

# %%
# Soft voting : moyenner les probabilités softmax
all_probs_array = np.array([all_val_probs[m] for m in top_k_models])
soft_voting_probs = all_probs_array.mean(axis=0)
soft_voting_preds = soft_voting_probs.argmax(axis=1)

acc_soft = accuracy_score(true_labels, soft_voting_preds)
f1_soft = f1_score(true_labels, soft_voting_preds, average='macro', zero_division=0)
print(f"\n📊 Soft Voting Top-{TOP_K} :")
print(f"   Accuracy : {acc_soft:.4f}")
print(f"   F1-macro : {f1_soft:.4f}")

# Comparaison
print(f"\n📈 Comparaison final :")
print(f"   Meilleur seul   ({top_k_models[0]:18s}): Acc {results[top_k_models[0]]['accuracy']:.4f}")
print(f"   Hard voting     (top-{TOP_K})            : Acc {acc_hard:.4f}")
print(f"   Soft voting     (top-{TOP_K})            : Acc {acc_soft:.4f}")

# %% [markdown]
# ## 10. Génération des soumissions de test (avec le meilleur seul + ensemble)

# %%
def predict_test_logits(model_name, test_df, device):
    """Prédit sur le test et retourne (ids, logits, probs)."""
    model, _ = load_checkpoint(model_name, device)
    model.eval()
    
    img_size = get_model_input_size(model_name)
    _, val_transform = get_transforms(img_size)
    
    test_ds = WBCDataset(test_df, TEST_DIR, test_df.columns[0], None, val_transform)
    test_loader_local = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True)
    
    all_logits, all_ids = [], []
    with torch.no_grad():
        for images, ids in tqdm(test_loader_local, desc=f"Test {model_name}"):
            images = images.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
            all_logits.append(outputs.cpu())
            all_ids.extend(ids)
    
    logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits, dim=1).numpy()
    
    del model
    torch.cuda.empty_cache()
    
    return all_ids, probs

# %%
# Récupérer les probas de chaque top-k modèle sur le test
test_probs = {}
test_ids_ref = None

for m in top_k_models:
    ids, probs = predict_test_logits(m, test_df, device)
    test_probs[m] = probs
    if test_ids_ref is None:
        test_ids_ref = ids

# %%
# Soumission 1 : meilleur modèle seul (déjà fait dans son notebook, mais on refait pour cohérence)
best_single = top_k_models[0]
preds_single = test_probs[best_single].argmax(axis=1)
labels_single = [idx2label[p] for p in preds_single]
save_submission(test_ids_ref, labels_single, f"{best_single}_alone",
                label_col=test_df.columns[1] if len(test_df.columns) > 1 else "label",
                id_col=test_df.columns[0])

# Soumission 2 : Soft voting top-k
soft_test_probs = np.mean([test_probs[m] for m in top_k_models], axis=0)
preds_soft = soft_test_probs.argmax(axis=1)
labels_soft = [idx2label[p] for p in preds_soft]
save_submission(test_ids_ref, labels_soft, f"ensemble_soft_top{TOP_K}",
                label_col=test_df.columns[1] if len(test_df.columns) > 1 else "label",
                id_col=test_df.columns[0])

# Soumission 3 : Hard voting top-k
hard_preds_test = np.array([test_probs[m].argmax(axis=1) for m in top_k_models])
preds_hard = np.array([
    np.bincount(hard_preds_test[:, i], minlength=NUM_CLASSES).argmax()
    for i in range(hard_preds_test.shape[1])
])
labels_hard = [idx2label[p] for p in preds_hard]
save_submission(test_ids_ref, labels_hard, f"ensemble_hard_top{TOP_K}",
                label_col=test_df.columns[1] if len(test_df.columns) > 1 else "label",
                id_col=test_df.columns[0])

print("\n🎉 Toutes les soumissions sont générées dans le dossier submissions/")

# %% [markdown]
# ## 11. Résumé final
# 
# Trois fichiers de soumission ont été créés dans `./submissions/` :
# 
# 1. `submission_<best>_alone.csv` — meilleur modèle seul
# 2. `submission_ensemble_soft_top3.csv` — soft voting (recommandé)
# 3. `submission_ensemble_hard_top3.csv` — hard voting
# 
# **En général, le soft voting donne les meilleurs résultats** car il prend en compte la confiance des modèles.
