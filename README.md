# WBC Classification — 4IM05 Kaggle Challenge

## Description

This project tackles the automatic classification of **white blood cells (WBC)** into 13 classes from microscopic images, as part of an internal Kaggle challenge for the 4IM05 course.

The dataset contains **28,901 images** with extreme class imbalance (from 11 to 13,015 images per class). The challenge focuses on maximizing the **F1 macro score** despite this imbalance.

**Final score: F1 macro = 0.7677** (weighted ensemble of Swin-T + ConvNeXt + EfficientNet-B3)

## Dataset

| Class | Images | Class | Images |
|-------|--------|-------|--------|
| SNE | 13,015 | BA | 415 |
| LY | 8,101 | BNE | 391 |
| MO | 2,746 | VLY | 366 |
| BL | 2,012 | MMY | 360 |
| EO | 861 | PMY | 114 |
| MY | 441 | PC | 68 |
| | | PLY | 11 |

Contact me if you want the dataset : agshay.nadanakumar@telecom-paris.fr

Place the data in a `data/` folder at the root of the project:
```
data/
├── train/                
├── test/                 
├── train_metadata.csv   
├── test_metadata.csv     
└── sample_submission.csv
```

## Methodology

We implemented two complementary approaches: a classical ML pipeline and a Deep Learning pipeline.

### 1. Classical ML Pipeline

* **Segmentation:** Otsu + HSV + morphology + watershed.
* **Feature extraction:** geometric (area, perimeter, circularity), texture (LBP, LDP, PRICoLBP), color (RGB/HSV), invariants (DT-CWT, bispectral, L-moments).
* **Classification:** SVM, Random Forest, LDA, K-PCA+SVM, Logistic Regression, Decision Tree.
* **Best result:** SVM with F1 macro = 0.432.

### 2. Deep Learning Pipeline

* **Baseline:** MobileNetV2, F1 = 0.35. Identified issue: broken class weights.
* **Weight correction:** from `w = N/Ni` (924:1 ratio) to `w = √(N/Ni)` (12:1 ratio). **+0.34 F1 gain**, the single most impactful change in the project.
* **Offline oversampling:** rare classes duplicated to a minimum of 500 images + heavy data augmentation.
* **2-stage training:** Stage 1 (general features), Stage 2 (fine-tune rare classes, LR÷20).
* **Model diversity:** EfficientNet-B3, ResNet50, ConvNeXt-Small, Swin Transformer Tiny.
* **Ensemble:** weighted soft voting using each model's F1 score + TTA ×10.

### Key Finding

> In highly imbalanced problems, **loss calibration matters more than model complexity**. Class weight correction (924:1 to 12:1 ratio) yielded a +0.34 F1 gain — more than architecture change (+0.02) or oversampling (+0.06).

## Results

| Step | Method | F1 macro |
|------|--------|----------|
| ML Pipeline | SVM + handcrafted features | 0.432 |
| DL V1 | MobileNetV2 (baseline) | 0.350 |
| DL V3 | EfficientNet-B3 + √weights | 0.690 |
| DL V4 | + offline oversampling + 2-stage training | 0.747 |
| DL | ConvNeXt-Small (V4 strategy) | 0.762 |
| DL | Swin Transformer Tiny (V4 strategy) | 0.768 |
| **Final** | **Weighted ensemble top-3 + TTA ×10** | **0.7677** |

## Project Structure

```
├── MachineLearning/
│   └── 01_machine_learning.ipynb     
│
├── DeepLearning/                 
│   ├── utils_v4.py                  
│   ├── 06_mobilenet_v2.py            
│   ├── 09_efficientnet_b3.py        
│   ├── 11_resnet50.py                
│   ├── 12_convnext.py             
│   ├── 13_swin.py                   
│   ├── 10_ensemble.py                
│   └── submissions/                 
│
├── data/                             
├── checkpoints/                      
├── requirements.txt
├── .gitignore
└── README.md
```

### DL Code Organization

`06_mobilenet_v2.py` is a **standalone** script (simple baseline — no oversampling, no 2-stage).

The 4 V4 models share the same pipeline through **`utils_v4.py`**:

```
utils_v4.py        
    ├── 09_efficientnet_b3.py    
    ├── 11_resnet50.py           
    ├── 12_convnext.py          
    ├── 13_swin.py              
    └── 10_ensemble.py          
```

Each model script is a single call to `run_full_pipeline()` with its specific parameters.

## Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### ML Pipeline

```bash
jupyter notebook MachineLearning/01_machine_learning.ipynb
```

### DL Pipeline (GPU recommended)

```bash
cd wbc_dl_pipeline/

#### Baseline
python 06_mobilenet_v2.py

#### V4 models (independent, run in any order)
python 09_efficientnet_b3.py
python 11_resnet50.py
python 12_convnext.py
python 13_swin.py

#### Final ensemble (requires checkpoints from the top-3 models)
python 10_ensemble.py
```

### With tmux (for SSH)

```bash
tmux new -s training
python 13_swin.py
#### Ctrl+B, D to detach — the job keeps running in the background
#### tmux attach -t training to reconnect
```

## References

- [1] K. AL-Dulaimi et al., "Classification of White Blood Cell Types from Microscope Images: Techniques and Challenges," 2019.
- [2] F. Rustam et al., "White Blood Cell Classification Using Texture and RGB Features with Machine Learning," Healthcare, 2022.
- [3] M. Ghosh et al., "Statistical Pattern Analysis of White Blood Cell Nuclei Morphometry," IEEE, 2010.
- [4] S. H. Rezatofighi et al., "Automatic Recognition of Five Types of White Blood Cells," ICIAR, 2010.
- [5] M. Habibzadeh et al., "Automatic WBC Classification Using Pre-trained DL Models: ResNet and Inception," ICMV, 2018.

## Author

Agshay Nadanakumar — Télécom Paris, 4IM05, April 2026