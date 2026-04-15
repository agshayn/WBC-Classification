# %% [markdown]
# # Classification des Globules Blancs (WBC) par Machine Learning
# 
# **Features extraites :** Géométriques, Texture (LBP, LDP, PRICoLBP), Couleur/Intensité, Invariantes/Statistiques (DT-CWT, Bispectral, L-moments)
# 
# **Classifieurs :** SVM, Random Forest, Arbre de Classification, K-PCA + SVM, LDA, Régression Logistique

# %% [markdown]
# ## 1. Imports et Configuration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import warnings
from tqdm import tqdm

# Scikit-image pour features de texture
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import opening, closing, disk
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed, clear_border
from skimage import img_as_ubyte

# Scipy pour features avancées (DT-CWT approx, bispectral, L-moments)
from scipy import ndimage, stats
from scipy.signal import convolve2d

# Scikit-learn pour ML
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("✅ Tous les imports réussis.")

# %% [markdown]
# ## 2. Chargement des Données

# %%
# ===== MODIFIEZ CES CHEMINS SELON VOTRE STRUCTURE =====
BASE_DIR = "./data"  # Racine de votre dataset
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
TEST_DIR = os.path.join(BASE_DIR, "Test")
TRAIN_CSV = os.path.join(BASE_DIR, "train_metadata.csv")
TEST_CSV = os.path.join(BASE_DIR, "test_metadata.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
# ========================================================

# Charger les métadonnées
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"Train metadata shape : {train_df.shape}")
print(f"Test metadata shape  : {test_df.shape}")
print(f"\nColonnes train : {train_df.columns.tolist()}")
print(f"\nDistribution des classes :\n{train_df.iloc[:, 1].value_counts()}")
print(f"\nNombre de classes : {train_df.iloc[:, 1].nunique()}")

# %%
# Identifier les noms de colonnes automatiquement
id_col = train_df.columns[0]
label_col = train_df.columns[1]
print(f"Colonne ID    : '{id_col}'")
print(f"Colonne Label : '{label_col}'")

# Encoder les labels
le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df[label_col])
class_names = le.classes_
print(f"\nClasses : {class_names}")

# %% [markdown]
# ## 3. Visualisation d'Échantillons

# %%
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.ravel()

# Afficher des exemples de différentes classes
unique_labels = train_df[label_col].unique()[:10]
for i, lbl in enumerate(unique_labels):
    if i >= 10:
        break
    sample = train_df[train_df[label_col] == lbl].iloc[0]
    img_path = os.path.join(TRAIN_DIR, f"{sample[id_col]}.png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(f"{lbl}", fontsize=10)
    axes[i].axis('off')

plt.suptitle("Échantillons de WBC par classe", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Segmentation des WBC
# 
# On utilise une pipeline de segmentation combinant :
# - Conversion en espace HSV pour isoler les noyaux (composante violette/bleue)
# - Seuillage d'Otsu
# - Opérations morphologiques (ouverture/fermeture)
# - Watershed pour séparer les cellules qui se touchent

# %%
def segment_wbc(img_bgr, method='otsu_hsv'):
    """
    Segmente le noyau du WBC à partir d'une image BGR.
    
    Returns:
        mask: masque binaire du noyau (uint8, 0 ou 255)
        nucleus_region: image du noyau isolé (BGR)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # --- Méthode 1 : Seuillage HSV pour noyaux violets/bleus ---
    # Les noyaux WBC sont typiquement dans la gamme violet-bleu après coloration
    h, s, v = cv2.split(img_hsv)
    
    # Seuillage sur la saturation (noyaux bien colorés)
    thresh_val = threshold_otsu(s)
    _, mask_s = cv2.threshold(s, thresh_val, 255, cv2.THRESH_BINARY)
    
    # Seuillage d'Otsu sur le canal bleu (les noyaux sont sombres en inversé)
    blue_channel = img_bgr[:, :, 0]
    thresh_otsu = threshold_otsu(blue_channel)
    _, mask_otsu = cv2.threshold(blue_channel, thresh_otsu, 255, cv2.THRESH_BINARY)
    
    # Combiner les masques
    mask = cv2.bitwise_and(mask_s, mask_otsu)
    
    # --- Opérations morphologiques ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Remplir les trous
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Garder le plus grand contour (le noyau principal)
        largest_contour = max(contours, key=cv2.contourArea)
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest_contour], -1, 255, -1)
        mask = mask_filled
    
    # --- Watershed pour raffiner ---
    # Distance transform
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Région inconnue
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marqueurs pour watershed
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    markers_ws = cv2.watershed(img_bgr, markers)
    mask_final = np.zeros_like(img_gray)
    mask_final[markers_ws > 1] = 255
    
    # Si watershed échoue, utiliser le masque original
    if np.sum(mask_final) < 100:
        mask_final = mask
    
    # Appliquer le masque
    nucleus_region = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_final)
    
    return mask_final, nucleus_region


# Test de segmentation sur quelques images
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
sample_ids = train_df[id_col].values[:4]

for i, sid in enumerate(sample_ids):
    img_path = os.path.join(TRAIN_DIR, f"{sid}.png")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    mask, nucleus = segment_wbc(img)
    
    axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f"Original: {sid}")
    axes[0, i].axis('off')
    
    axes[1, i].imshow(mask, cmap='gray')
    axes[1, i].set_title("Masque segmenté")
    axes[1, i].axis('off')
    
    axes[2, i].imshow(cv2.cvtColor(nucleus, cv2.COLOR_BGR2RGB))
    axes[2, i].set_title("Noyau isolé")
    axes[2, i].axis('off')

plt.suptitle("Pipeline de Segmentation (Otsu + Morpho + Watershed)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Extraction de Features
# 
# ### 5.1 Features Géométriques (aire, périmètre, rayon, circularité, excentricité)
# ### 5.2 Features de Texture (LBP, LDP, PRICoLBP)
# ### 5.3 Features Couleur/Intensité (statistiques RGB, HSV)
# ### 5.4 Features Invariantes/Statistiques (DT-CWT, Bispectral, L-moments)

# %%
# =====================================================================
# 5.1 FEATURES GÉOMÉTRIQUES
# =====================================================================

def extract_geometric_features(mask):
    """
    Extrait les features géométriques du masque segmenté.
    - Aire, Périmètre, Rayon équivalent, Circularité, Excentricité,
      Compacité, Rapport d'aspect, Solidité, Étendue
    """
    features = {}
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Retourner des zéros si pas de contour
        keys = ['area', 'perimeter', 'equiv_radius', 'circularity', 
                'eccentricity', 'compactness', 'aspect_ratio', 
                'solidity', 'extent', 'convex_area']
        return {k: 0.0 for k in keys}
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Aire
    area = cv2.contourArea(cnt)
    features['area'] = area
    
    # Périmètre
    perimeter = cv2.arcLength(cnt, True)
    features['perimeter'] = perimeter
    
    # Rayon équivalent (rayon du cercle de même aire)
    features['equiv_radius'] = np.sqrt(area / np.pi) if area > 0 else 0
    
    # Circularité = 4π * Aire / Périmètre²
    features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Excentricité (via ellipse ajustée)
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), angle = ellipse
        a = max(MA, ma) / 2
        b = min(MA, ma) / 2
        features['eccentricity'] = np.sqrt(1 - (b**2 / a**2)) if a > 0 else 0
        features['aspect_ratio'] = MA / ma if ma > 0 else 0
    else:
        features['eccentricity'] = 0
        features['aspect_ratio'] = 0
    
    # Compacité = Périmètre² / Aire
    features['compactness'] = (perimeter ** 2) / area if area > 0 else 0
    
    # Solidité = Aire / Aire convexe
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    features['solidity'] = area / hull_area if hull_area > 0 else 0
    features['convex_area'] = hull_area
    
    # Étendue = Aire / Aire du bounding rect
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    features['extent'] = area / rect_area if rect_area > 0 else 0
    
    return features

# %%
# =====================================================================
# 5.2 FEATURES DE TEXTURE : LBP, LDP, PRICoLBP
# =====================================================================

def extract_lbp_features(gray_img, mask, n_points=24, radius=3, n_bins=26):
    """
    Local Binary Pattern (LBP).
    Réf: Rezatofighi et al. (2011) - LBP pour classification WBC.
    """
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    
    # Appliquer le masque
    lbp_masked = lbp[mask > 0]
    
    if len(lbp_masked) == 0:
        return {f'lbp_bin_{i}': 0.0 for i in range(n_bins)}
    
    # Histogramme normalisé
    hist, _ = np.histogram(lbp_masked, bins=n_bins, range=(0, n_points + 2), density=True)
    
    features = {f'lbp_bin_{i}': hist[i] for i in range(len(hist))}
    
    # Statistiques LBP
    features['lbp_mean'] = np.mean(lbp_masked)
    features['lbp_std'] = np.std(lbp_masked)
    features['lbp_energy'] = np.sum(hist ** 2)
    features['lbp_entropy'] = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    
    return features


def extract_ldp_features(gray_img, mask):
    """
    Local Directional Pattern (LDP).
    Réf: Su et al. (2014) - LDP pour classification WBC en 5 types.
    
    LDP utilise les réponses de 8 masques de Kirsch pour encoder
    les directions de gradient autour de chaque pixel.
    """
    # 8 masques de Kirsch pour 8 directions
    kirsch_masks = [
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),   # E
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),    # NE
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),     # N
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),     # NW
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),     # W
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),     # SW
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),     # S
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])      # SE
    ]
    
    # Calculer les réponses pour chaque direction
    responses = []
    for km in kirsch_masks:
        resp = convolve2d(gray_img.astype(np.float64), km, mode='same', boundary='symm')
        responses.append(np.abs(resp))
    
    responses = np.array(responses)
    
    # LDP : prendre les k (=3) plus fortes réponses et encoder en binaire
    k = 3
    sorted_indices = np.argsort(responses, axis=0)[::-1]
    
    ldp_code = np.zeros(gray_img.shape, dtype=np.uint8)
    for i in range(k):
        direction = sorted_indices[i]
        ldp_code += (1 << direction).astype(np.uint8)
    
    # Histogramme sur la région masquée
    ldp_masked = ldp_code[mask > 0]
    
    if len(ldp_masked) == 0:
        return {f'ldp_bin_{i}': 0.0 for i in range(56)}
    
    hist, _ = np.histogram(ldp_masked, bins=56, range=(0, 256), density=True)
    
    features = {f'ldp_bin_{i}': hist[i] for i in range(len(hist))}
    features['ldp_mean'] = np.mean(ldp_masked)
    features['ldp_std'] = np.std(ldp_masked)
    features['ldp_entropy'] = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    
    return features


def extract_pricolbp_features(gray_img, mask, radius=2):
    """
    Pairwise Rotation Invariant Co-occurrence LBP (PRICoLBP).
    Réf: Zhao et al. (2017) - Granularity feature pour WBC.
    
    PRICoLBP capture les co-occurrences entre paires de LBP 
    à des positions symétriques, invariant en rotation.
    """
    n_points = 8
    
    # LBP uniforme
    lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
    n_patterns = n_points + 2  # 10 patterns uniformes pour P=8
    
    # PRICoLBP : co-occurrence entre pixels opposés sur le cercle
    rows, cols = gray_img.shape
    co_matrix = np.zeros((n_patterns, n_patterns), dtype=np.float64)
    
    # Pour chaque paire de points diamétralement opposés
    for angle_idx in range(n_points // 2):
        angle1 = 2 * np.pi * angle_idx / n_points
        angle2 = angle1 + np.pi
        
        dx1, dy1 = int(round(radius * np.cos(angle1))), int(round(radius * np.sin(angle1)))
        dx2, dy2 = int(round(radius * np.cos(angle2))), int(round(radius * np.sin(angle2)))
        
        for r in range(radius, rows - radius):
            for c in range(radius, cols - radius):
                if mask[r, c] == 0:
                    continue
                p1 = int(lbp[r + dy1, c + dx1]) if 0 <= r+dy1 < rows and 0 <= c+dx1 < cols else 0
                p2 = int(lbp[r + dy2, c + dx2]) if 0 <= r+dy2 < rows and 0 <= c+dx2 < cols else 0
                p1 = min(p1, n_patterns - 1)
                p2 = min(p2, n_patterns - 1)
                co_matrix[p1, p2] += 1
    
    # Normaliser
    total = co_matrix.sum()
    if total > 0:
        co_matrix /= total
    
    # Extraire des features de la matrice de co-occurrence
    features = {}
    features['pricolbp_energy'] = np.sum(co_matrix ** 2)
    features['pricolbp_entropy'] = -np.sum(co_matrix[co_matrix > 0] * np.log2(co_matrix[co_matrix > 0]))
    features['pricolbp_contrast'] = sum(
        co_matrix[i, j] * (i - j) ** 2
        for i in range(n_patterns)
        for j in range(n_patterns)
    )
    features['pricolbp_homogeneity'] = sum(
        co_matrix[i, j] / (1 + abs(i - j))
        for i in range(n_patterns)
        for j in range(n_patterns)
    )
    
    # Aplatir le triangle supérieur comme features additionnelles
    upper_tri = co_matrix[np.triu_indices(n_patterns)]
    for idx, val in enumerate(upper_tri[:20]):  # Limiter à 20
        features[f'pricolbp_co_{idx}'] = val
    
    return features

# %%
# =====================================================================
# 5.3 FEATURES COULEUR / INTENSITÉ
# =====================================================================

def extract_color_features(img_bgr, mask):
    """
    Statistiques des canaux RGB et HSV sur la région segmentée.
    """
    features = {}
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    channel_names_rgb = ['red', 'green', 'blue']
    channel_names_hsv = ['hue', 'saturation', 'value']
    
    for img_space, ch_names in [(img_rgb, channel_names_rgb), (img_hsv, channel_names_hsv)]:
        for ch_idx, ch_name in enumerate(ch_names):
            channel = img_space[:, :, ch_idx]
            masked_pixels = channel[mask > 0]
            
            if len(masked_pixels) == 0:
                features[f'{ch_name}_mean'] = 0
                features[f'{ch_name}_std'] = 0
                features[f'{ch_name}_skew'] = 0
                features[f'{ch_name}_kurtosis'] = 0
                features[f'{ch_name}_median'] = 0
                features[f'{ch_name}_min'] = 0
                features[f'{ch_name}_max'] = 0
                continue
            
            features[f'{ch_name}_mean'] = np.mean(masked_pixels)
            features[f'{ch_name}_std'] = np.std(masked_pixels)
            features[f'{ch_name}_skew'] = float(stats.skew(masked_pixels))
            features[f'{ch_name}_kurtosis'] = float(stats.kurtosis(masked_pixels))
            features[f'{ch_name}_median'] = np.median(masked_pixels)
            features[f'{ch_name}_min'] = np.min(masked_pixels)
            features[f'{ch_name}_max'] = np.max(masked_pixels)
    
    # Ratios de couleur
    mean_r = features.get('red_mean', 1)
    mean_g = features.get('green_mean', 1)
    mean_b = features.get('blue_mean', 1)
    total = mean_r + mean_g + mean_b + 1e-8
    features['ratio_rg'] = mean_r / (mean_g + 1e-8)
    features['ratio_rb'] = mean_r / (mean_b + 1e-8)
    features['ratio_gb'] = mean_g / (mean_b + 1e-8)
    features['norm_r'] = mean_r / total
    features['norm_g'] = mean_g / total
    features['norm_b'] = mean_b / total
    
    return features

# %%
# =====================================================================
# 5.4 FEATURES INVARIANTES / STATISTIQUES
# =====================================================================

# --- 5.4.1 DT-CWT (Dual-Tree Complex Wavelet Transform) ---
def extract_dtcwt_features(gray_img, mask, levels=3):
    """
    Features basées sur la DT-CWT.
    Réf: Habibzadeh et al. (2013) - DT-CWT + SVM pour WBC.
    
    Implémentation simplifiée utilisant des paires de filtres 
    de Gabor à différentes orientations comme approximation de DT-CWT.
    (La vraie DT-CWT nécessite la lib dtcwt, ici on l'approxime)
    """
    features = {}
    
    # Approximation via ondelettes de Gabor multi-échelle multi-orientation
    # (6 orientations × n niveaux = sous-bandes)
    orientations = [0, 30, 60, 90, 120, 150]
    sigmas = [1, 2, 4]  # Multi-échelle (correspond aux niveaux de décomposition)
    frequencies = [0.3, 0.2, 0.1]
    
    subband_idx = 0
    for level, (sigma, freq) in enumerate(zip(sigmas, frequencies)):
        for theta_deg in orientations:
            theta = np.deg2rad(theta_deg)
            
            # Filtre de Gabor réel et imaginaire (paire analytique)
            kernel_real = cv2.getGaborKernel(
                ksize=(21, 21), sigma=sigma, theta=theta,
                lambd=1.0/freq, gamma=0.5, psi=0
            )
            kernel_imag = cv2.getGaborKernel(
                ksize=(21, 21), sigma=sigma, theta=theta,
                lambd=1.0/freq, gamma=0.5, psi=np.pi/2
            )
            
            # Réponse complexe
            resp_real = cv2.filter2D(gray_img.astype(np.float64), -1, kernel_real)
            resp_imag = cv2.filter2D(gray_img.astype(np.float64), -1, kernel_imag)
            
            magnitude = np.sqrt(resp_real**2 + resp_imag**2)
            
            # Statistiques sur la zone masquée
            mag_masked = magnitude[mask > 0]
            if len(mag_masked) > 0:
                features[f'dtcwt_mag_mean_{subband_idx}'] = np.mean(mag_masked)
                features[f'dtcwt_mag_std_{subband_idx}'] = np.std(mag_masked)
                features[f'dtcwt_mag_energy_{subband_idx}'] = np.mean(mag_masked ** 2)
            else:
                features[f'dtcwt_mag_mean_{subband_idx}'] = 0
                features[f'dtcwt_mag_std_{subband_idx}'] = 0
                features[f'dtcwt_mag_energy_{subband_idx}'] = 0
            
            subband_idx += 1
    
    return features


# --- 5.4.2 Caractéristiques Bispectrales ---
def extract_bispectral_features(gray_img, mask, n_features=15):
    """
    Features bispectrales invariantes.
    Réf: Al-Dulaimi et al. (2018) - Classification WBC en 10 classes, 96.13% accuracy.
    
    Le bispectre est le spectre d'ordre 3 (corrélation triple en fréquence).
    On extrait des invariants à partir de la magnitude du bispectre.
    """
    features = {}
    
    # Appliquer le masque et recadrer
    masked_img = gray_img.copy().astype(np.float64)
    masked_img[mask == 0] = 0
    
    # Redimensionner pour FFT uniforme
    target_size = 64
    resized = cv2.resize(masked_img, (target_size, target_size))
    
    # FFT 2D
    F = np.fft.fft2(resized)
    F_shifted = np.fft.fftshift(F)
    
    magnitude = np.abs(F_shifted)
    phase = np.angle(F_shifted)
    
    # Bispectre simplifié : B(f1, f2) = F(f1) * F(f2) * conj(F(f1+f2))
    # On calcule des invariants à partir de tranches radiales
    center = target_size // 2
    
    # Profil radial de la magnitude (invariant en rotation)
    radii = np.arange(1, center)
    radial_profile = []
    for r in radii:
        # Points sur le cercle de rayon r
        angles = np.linspace(0, 2 * np.pi, max(8, int(2 * np.pi * r)), endpoint=False)
        vals = []
        for a in angles:
            x = int(center + r * np.cos(a))
            y = int(center + r * np.sin(a))
            if 0 <= x < target_size and 0 <= y < target_size:
                vals.append(magnitude[y, x])
        if vals:
            radial_profile.append(np.mean(vals))
        else:
            radial_profile.append(0)
    
    radial_profile = np.array(radial_profile)
    
    # Invariants bispectraux : moments du profil radial
    if len(radial_profile) > 0 and np.sum(radial_profile) > 0:
        rp_norm = radial_profile / (np.sum(radial_profile) + 1e-10)
        features['bispec_mean'] = np.mean(radial_profile)
        features['bispec_std'] = np.std(radial_profile)
        features['bispec_skew'] = float(stats.skew(radial_profile))
        features['bispec_kurtosis'] = float(stats.kurtosis(radial_profile))
        features['bispec_energy'] = np.sum(radial_profile ** 2)
        features['bispec_entropy'] = -np.sum(rp_norm[rp_norm > 0] * np.log2(rp_norm[rp_norm > 0]))
        
        # Moments centraux normalisés
        x_vals = np.arange(len(radial_profile))
        mu = np.sum(x_vals * rp_norm)
        for order in range(2, 6):
            moment = np.sum(((x_vals - mu) ** order) * rp_norm)
            features[f'bispec_moment_{order}'] = moment
        
        # Phase bispectrale (bicoherence approximée)
        phase_profile = []
        for r in radii[:len(radial_profile)]:
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            p_vals = []
            for a in angles:
                x = int(center + r * np.cos(a))
                y = int(center + r * np.sin(a))
                if 0 <= x < target_size and 0 <= y < target_size:
                    p_vals.append(phase[y, x])
            phase_profile.append(np.std(p_vals) if p_vals else 0)
        
        features['bispec_phase_std'] = np.mean(phase_profile) if phase_profile else 0
    else:
        for k in ['bispec_mean', 'bispec_std', 'bispec_skew', 'bispec_kurtosis',
                   'bispec_energy', 'bispec_entropy', 'bispec_phase_std']:
            features[k] = 0.0
        for order in range(2, 6):
            features[f'bispec_moment_{order}'] = 0.0
    
    return features


# --- 5.4.3 L-moments ---
def extract_lmoments_features(gray_img, mask):
    """
    L-moments (L-mean, L-scale, L-skewness, L-kurtosis) de la projection Radon.
    Réf: Al-Dulaimi et al. (2018) - L-moments invariants, 97.23% accuracy.
    
    Les L-moments sont des statistiques robustes basées sur les combinaisons
    linéaires de statistiques d'ordre.
    """
    features = {}
    
    # Appliquer le masque
    masked_img = gray_img.copy().astype(np.float64)
    masked_img[mask == 0] = 0
    
    # Projection de Radon simplifiée sur plusieurs angles
    from skimage.transform import radon
    
    # Redimensionner
    target = 64
    resized = cv2.resize(masked_img, (target, target))
    
    angles = np.linspace(0, 180, 18, endpoint=False)
    sinogram = radon(resized, theta=angles, circle=True)
    
    # Calculer les L-moments pour chaque projection angulaire
    all_l1, all_l2, all_t3, all_t4 = [], [], [], []
    
    for col_idx in range(sinogram.shape[1]):
        projection = sinogram[:, col_idx]
        projection = projection[projection != 0]
        
        if len(projection) < 4:
            continue
        
        # Trier les données
        x = np.sort(projection)
        n = len(x)
        
        # L-moment 1 (L-mean)
        l1 = np.mean(x)
        
        # L-moment 2 (L-scale)
        b0 = np.mean(x)
        b1 = np.sum(np.arange(1, n) * x[1:]) / (n * (n - 1)) if n > 1 else 0
        l2 = 2 * b1 - b0
        
        # L-moment ratios (L-skewness, L-kurtosis)
        if n > 2:
            b2 = np.sum(np.arange(1, n-1) * np.arange(2, n) * x[2:]) / (n * (n-1) * (n-2)) if n > 2 else 0
            l3 = 6*b2 - 6*b1 + b0
            t3 = l3 / (l2 + 1e-10)  # L-skewness
        else:
            t3 = 0
        
        if n > 3:
            b3 = np.sum(
                np.arange(1, n-2) * np.arange(2, n-1) * np.arange(3, n) * x[3:]
            ) / (n * (n-1) * (n-2) * (n-3)) if n > 3 else 0
            l4 = 20*b3 - 30*b2 + 12*b1 - b0
            t4 = l4 / (l2 + 1e-10)  # L-kurtosis
        else:
            t4 = 0
        
        all_l1.append(l1)
        all_l2.append(abs(l2))
        all_t3.append(t3)
        all_t4.append(t4)
    
    # Statistiques des L-moments sur toutes les projections
    for name, vals in [('lmom_l1', all_l1), ('lmom_l2', all_l2),
                       ('lmom_t3', all_t3), ('lmom_t4', all_t4)]:
        if len(vals) > 0:
            features[f'{name}_mean'] = np.mean(vals)
            features[f'{name}_std'] = np.std(vals)
            features[f'{name}_min'] = np.min(vals)
            features[f'{name}_max'] = np.max(vals)
        else:
            features[f'{name}_mean'] = 0
            features[f'{name}_std'] = 0
            features[f'{name}_min'] = 0
            features[f'{name}_max'] = 0
    
    return features

# %% [markdown]
# ## 6. Pipeline Complète d'Extraction

# %%
def extract_all_features(img_bgr):
    """
    Pipeline complète : segmentation + extraction de toutes les features.
    """
    # Segmentation
    mask, nucleus = segment_wbc(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    all_features = {}
    
    # 1. Géométriques
    all_features.update(extract_geometric_features(mask))
    
    # 2. Texture - LBP
    all_features.update(extract_lbp_features(gray, mask))
    
    # 3. Texture - LDP
    all_features.update(extract_ldp_features(gray, mask))
    
    # 4. Texture - PRICoLBP (plus lent, on utilise une image réduite)
    gray_small = cv2.resize(gray, (128, 128))
    mask_small = cv2.resize(mask, (128, 128))
    all_features.update(extract_pricolbp_features(gray_small, mask_small))
    
    # 5. Couleur/Intensité
    all_features.update(extract_color_features(img_bgr, mask))
    
    # 6. DT-CWT (approximation Gabor)
    all_features.update(extract_dtcwt_features(gray, mask))
    
    # 7. Bispectral
    all_features.update(extract_bispectral_features(gray, mask))
    
    # 8. L-moments
    all_features.update(extract_lmoments_features(gray, mask))
    
    return all_features

# %%
# ===== EXTRACTION SUR LE DATASET TRAIN =====
print("Extraction des features sur le dataset d'entraînement...")
print("(Cela peut prendre plusieurs minutes selon la taille du dataset)\n")

train_features_list = []
train_labels = []
train_ids = []
errors = []

# Pour un premier test rapide, limiter le nombre d'images
# Mettre N_SAMPLES = None pour tout traiter
N_SAMPLES = None  # Changez à None pour traiter toutes les images

sample_df = train_df if N_SAMPLES is None else train_df.sample(n=N_SAMPLES, random_state=42)

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Extraction"):
    img_id = row[id_col]
    img_path = os.path.join(TRAIN_DIR, f"{img_id}.png")
    
    if not os.path.exists(img_path):
        errors.append(img_id)
        continue
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            errors.append(img_id)
            continue
        
        feats = extract_all_features(img)
        train_features_list.append(feats)
        train_labels.append(row['label_encoded'])
        train_ids.append(img_id)
    except Exception as e:
        errors.append(img_id)
        if len(errors) <= 5:
            print(f"  ⚠️ Erreur sur {img_id}: {e}")

# Convertir en DataFrame
X_df = pd.DataFrame(train_features_list)
X_df.fillna(0, inplace=True)
X_df.replace([np.inf, -np.inf], 0, inplace=True)

y = np.array(train_labels)

print(f"\n✅ Extraction terminée !")
print(f"   Nombre d'images traitées : {len(X_df)}")
print(f"   Nombre de features       : {X_df.shape[1]}")
print(f"   Erreurs                   : {len(errors)}")
print(f"\nAperçu des features :")
print(X_df.head())

# %% [markdown]
# ## 7. Préparation des Données

# %%
# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_df.values, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Train : {X_train_scaled.shape}")
print(f"Val   : {X_val_scaled.shape}")

# Vérification distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\nDistribution train :")
for u, c in zip(unique, counts):
    print(f"  Classe {le.inverse_transform([u])[0]}: {c}")

# %% [markdown]
# ## 8. Classification
# 
# ### 8.1 SVM (Support Vector Machine)
# ### 8.2 Random Forest
# ### 8.3 Arbre de Classification
# ### 8.4 K-PCA + SVM
# ### 8.5 LDA (Linear Discriminant Analysis)
# ### 8.6 Régression Logistique

# %%
# =====================================================================
# 8.1 SVM
# =====================================================================
print("=" * 60)
print("8.1 SVM (Support Vector Machine)")
print("=" * 60)

svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_val_scaled)
acc_svm = accuracy_score(y_val, y_pred_svm)
print(f"\nAccuracy SVM : {acc_svm:.4f}")
print(f"\nRapport de classification :")
print(classification_report(y_val, y_pred_svm, target_names=class_names, zero_division=0))

# %%
# =====================================================================
# 8.2 Random Forest
# =====================================================================
print("=" * 60)
print("8.2 Random Forest")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=5,
    random_state=42, class_weight='balanced', n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_val_scaled)
acc_rf = accuracy_score(y_val, y_pred_rf)
print(f"\nAccuracy Random Forest : {acc_rf:.4f}")
print(f"\nRapport de classification :")
print(classification_report(y_val, y_pred_rf, target_names=class_names, zero_division=0))

# %%
# =====================================================================
# 8.3 Arbre de Classification (Decision Tree)
# =====================================================================
print("=" * 60)
print("8.3 Arbre de Classification")
print("=" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=20, min_samples_split=10,
    random_state=42, class_weight='balanced'
)
dt_model.fit(X_train_scaled, y_train)

y_pred_dt = dt_model.predict(X_val_scaled)
acc_dt = accuracy_score(y_val, y_pred_dt)
print(f"\nAccuracy Decision Tree : {acc_dt:.4f}")
print(f"\nRapport de classification :")
print(classification_report(y_val, y_pred_dt, target_names=class_names, zero_division=0))

# %%
# =====================================================================
# 8.4 K-PCA + SVM
# =====================================================================
print("=" * 60)
print("8.4 K-PCA + SVM")
print("=" * 60)

# K-PCA pour réduction de dimensionnalité
n_components_kpca = min(50, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
kpca = KernelPCA(n_components=n_components_kpca, kernel='rbf', gamma=0.01, random_state=42)
X_train_kpca = kpca.fit_transform(X_train_scaled)
X_val_kpca = kpca.transform(X_val_scaled)

svm_kpca = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, class_weight='balanced')
svm_kpca.fit(X_train_kpca, y_train)

y_pred_kpca = svm_kpca.predict(X_val_kpca)
acc_kpca = accuracy_score(y_val, y_pred_kpca)
print(f"\nAccuracy K-PCA + SVM : {acc_kpca:.4f}")
print(f"Dimensions après K-PCA : {X_train_kpca.shape[1]}")
print(f"\nRapport de classification :")
print(classification_report(y_val, y_pred_kpca, target_names=class_names, zero_division=0))

# %%
# =====================================================================
# 8.5 LDA (Linear Discriminant Analysis)
# =====================================================================
print("=" * 60)
print("8.5 LDA (Linear Discriminant Analysis)")
print("=" * 60)

n_components_lda = min(len(class_names) - 1, X_train_scaled.shape[1])
lda_model = LinearDiscriminantAnalysis(n_components=n_components_lda)
X_train_lda = lda_model.fit_transform(X_train_scaled, y_train)
X_val_lda = lda_model.transform(X_val_scaled)

# LDA comme classifieur direct
y_pred_lda = lda_model.predict(X_val_scaled)
acc_lda = accuracy_score(y_val, y_pred_lda)
print(f"\nAccuracy LDA (classifieur) : {acc_lda:.4f}")
print(f"Dimensions après LDA : {n_components_lda}")
print(f"\nRapport de classification :")
print(classification_report(y_val, y_pred_lda, target_names=class_names, zero_division=0))

# %%
# =====================================================================
# 8.6 Régression Logistique
# =====================================================================
print("=" * 60)
print("8.6 Régression Logistique")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=2000, C=1.0, solver='lbfgs',
    multi_class='multinomial', random_state=42,
    class_weight='balanced'
)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_val_scaled)
acc_lr = accuracy_score(y_val, y_pred_lr)
print(f"\nAccuracy Régression Logistique : {acc_lr:.4f}")
print(f"\nRapport de classification :")
print(classification_report(y_val, y_pred_lr, target_names=class_names, zero_division=0))

# %% [markdown]
# ## 9. Comparaison des Modèles

# %%
# Résumé des résultats
results = pd.DataFrame({
    'Modèle': ['SVM (RBF)', 'Random Forest', 'Decision Tree', 
               'K-PCA + SVM', 'LDA', 'Régression Logistique'],
    'Accuracy': [acc_svm, acc_rf, acc_dt, acc_kpca, acc_lda, acc_lr],
    'F1-Score (macro)': [
        f1_score(y_val, y_pred_svm, average='macro', zero_division=0),
        f1_score(y_val, y_pred_rf, average='macro', zero_division=0),
        f1_score(y_val, y_pred_dt, average='macro', zero_division=0),
        f1_score(y_val, y_pred_kpca, average='macro', zero_division=0),
        f1_score(y_val, y_pred_lda, average='macro', zero_division=0),
        f1_score(y_val, y_pred_lr, average='macro', zero_division=0),
    ]
})

results = results.sort_values('Accuracy', ascending=False).reset_index(drop=True)
print(results.to_string(index=False))

# %%
# Visualisation comparative
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Barplot Accuracy
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
bars = axes[0].barh(results['Modèle'], results['Accuracy'], color=colors)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Comparaison des Accuracy')
axes[0].set_xlim(0, 1)
for bar, val in zip(bars, results['Accuracy']):
    axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.3f}', va='center', fontweight='bold')

# Barplot F1-Score
bars2 = axes[1].barh(results['Modèle'], results['F1-Score (macro)'], color=colors)
axes[1].set_xlabel('F1-Score (macro)')
axes[1].set_title('Comparaison des F1-Scores')
axes[1].set_xlim(0, 1)
for bar, val in zip(bars2, results['F1-Score (macro)']):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{val:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 10. Matrices de Confusion

# %%
predictions = {
    'SVM': y_pred_svm,
    'Random Forest': y_pred_rf,
    'Decision Tree': y_pred_dt,
    'K-PCA + SVM': y_pred_kpca,
    'LDA': y_pred_lda,
    'Rég. Logistique': y_pred_lr
}

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes = axes.ravel()

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_val, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx], cbar=True)
    axes[idx].set_title(f'{name}\n(Acc: {accuracy_score(y_val, y_pred):.3f})', fontsize=12)
    axes[idx].set_ylabel('Vrai')
    axes[idx].set_xlabel('Prédit')
    axes[idx].tick_params(axis='both', labelsize=7, rotation=45)

plt.suptitle("Matrices de Confusion - Tous les Modèles", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 11. Importance des Features (Random Forest)

# %%
feature_importance = pd.DataFrame({
    'feature': X_df.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Top 30 features
top_n = 30
fig, ax = plt.subplots(figsize=(12, 10))
top_features = feature_importance.head(top_n)
ax.barh(range(top_n), top_features['importance'].values, color='steelblue')
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features['feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title(f'Top {top_n} Features les plus importantes (Random Forest)')
plt.tight_layout()
plt.show()

print("\nTop 15 features :")
print(feature_importance.head(15).to_string(index=False))

# %% [markdown]
# ## 12. Validation Croisée (Cross-Validation)

# %%
print("Validation croisée (5-fold stratifié) :")
print("-" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_cv = {
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=20, class_weight='balanced', random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'Rég. Logistique': LogisticRegression(max_iter=2000, multi_class='multinomial', class_weight='balanced', random_state=42),
}

cv_results = {}
X_all_scaled = scaler.fit_transform(X_df.values)

for name, model in models_cv.items():
    scores = cross_val_score(model, X_all_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:25s} : {scores.mean():.4f} ± {scores.std():.4f}")

# %% [markdown]
# ## 13. Prédiction sur le Test Set et Génération du Fichier de Soumission

# %%
print("Extraction des features sur le dataset de test...")

test_features_list = []
test_ids_final = []
test_errors = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test extraction"):
    img_id = row[test_df.columns[0]]
    img_path = os.path.join(TEST_DIR, f"{img_id}.png")
    
    if not os.path.exists(img_path):
        test_errors.append(img_id)
        continue
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            test_errors.append(img_id)
            continue
        
        feats = extract_all_features(img)
        test_features_list.append(feats)
        test_ids_final.append(img_id)
    except Exception as e:
        test_errors.append(img_id)

X_test_df = pd.DataFrame(test_features_list)
X_test_df.fillna(0, inplace=True)
X_test_df.replace([np.inf, -np.inf], 0, inplace=True)

# S'assurer que les colonnes sont les mêmes
missing_cols = set(X_df.columns) - set(X_test_df.columns)
for col in missing_cols:
    X_test_df[col] = 0
X_test_df = X_test_df[X_df.columns]  # Même ordre

print(f"\n✅ Test extraction terminée : {len(X_test_df)} images")
print(f"   Erreurs : {len(test_errors)}")

# %%
# Entraîner le meilleur modèle sur TOUTES les données train
print("\nEntraînement du modèle final sur toutes les données train...")

# Utiliser le meilleur modèle basé sur la comparaison
best_model_name = results.iloc[0]['Modèle']
print(f"Meilleur modèle : {best_model_name}")

# On réentraîne un SVM sur toutes les données (adapter si autre modèle est meilleur)
final_scaler = StandardScaler()
X_all_scaled_final = final_scaler.fit_transform(X_df.values)
X_test_scaled_final = final_scaler.transform(X_test_df.values)

final_model = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42)
final_model.fit(X_all_scaled_final, y)

# Prédictions
y_test_pred = final_model.predict(X_test_scaled_final)
y_test_labels = le.inverse_transform(y_test_pred)

# %%
# Créer le fichier de soumission
submission = pd.DataFrame({
    test_df.columns[0]: test_ids_final,
    label_col: y_test_labels
})

submission.to_csv("submission.csv", index=False)
print(f"\n✅ Fichier de soumission créé : submission.csv")
print(f"   Nombre de prédictions : {len(submission)}")
print(f"\nDistribution des prédictions :")
print(submission[label_col].value_counts())
print(f"\nAperçu :")
print(submission.head(10))

# %% [markdown]
# ## 14. Résumé
# 
# | Feature | Méthode | Référence |
# |---------|---------|-----------|
# | Géométriques | Aire, Périmètre, Rayon, Circularité, Excentricité | Ghosh et al. (2010) |
# | Texture LBP | Local Binary Pattern uniforme | Rezatofighi et al. (2011) |
# | Texture LDP | Local Directional Pattern (Kirsch) | Su et al. (2014) |
# | Texture PRICoLBP | Pairwise Rotation Invariant Co-occurrence LBP | Zhao et al. (2017) |
# | Couleur/Intensité | Statistiques RGB + HSV | — |
# | DT-CWT | Gabor multi-échelle multi-orientation | Habibzadeh et al. (2013) |
# | Bispectral | Invariants bispectraux FFT + profil radial | Al-Dulaimi et al. (2018) |
# | L-moments | L-mean, L-scale, L-skewness, L-kurtosis via Radon | Al-Dulaimi et al. (2018) |
