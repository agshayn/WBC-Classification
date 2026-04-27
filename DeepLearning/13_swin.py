##### Swin Transformer Tiny V4 (Oversampling/2-Stage)
from utils_v4 import run_full_pipeline

run_full_pipeline(
    model_name='swin_tiny',
    batch_size=24,        
    use_grayscale=False,   
    use_affine=False,     
)