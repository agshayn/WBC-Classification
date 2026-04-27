# 09_efficientnet_b3.py — EfficientNet-B3 V4 (Oversampling + 2-Stage)
from utils_v4 import run_full_pipeline

run_full_pipeline(
    model_name='efficientnet_b3',
    batch_size=32,
    use_grayscale=True,   # spécifique à ce modèle
    use_affine=True,      # RandomAffine dans TTA
)