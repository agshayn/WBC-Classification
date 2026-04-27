##### ResNet50 V4 (Oversampling/2-Stage)
from utils_v4 import run_full_pipeline

run_full_pipeline(
    model_name='resnet50',
    batch_size=32,
    use_grayscale=True,   
    use_affine=True,     
)