# EFFResNet-Unified: Multi-Stream Spatio-Frequency and Steganalysis Cross-Attention for Generalizable Deepfake Video Detection

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-blue.svg)](LICENSE)
[![UROP](https://img.shields.io/badge/Research-UROP%20Project-success.svg)]()

> **Project Title:** EFFResNet-Unified: Multi-Stream Spatio-Frequency and Steganalysis Cross-Attention for Generalizable Deepfake Video Detection  
> **Target Publication:** IEEE Conference Submission (SMART / ICASSP / CVPR-W)  
> **Author:** Kesava Rama  

---

## 📁 Repository Directory Structure

```text
EFFResNet-Unified/
├── README.md                      # Comprehensive project documentation & guide
├── requirements.txt               # Direct Python dependencies
├── predict.py                     # CLI tool for single/multi-video deepfake inference
│
├── checkpoints/                   # Trained deep learning model weights
│   └── best_effresnet_unified.pth # Final trained unified model (392 MB, Tesla T4 GPU)
│
├── docs/                          # Formal research report & publication materials
│   └── EFFResNet_Unified_Comprehensive_Report.pdf  # Publication-grade technical report
│
├── figures/                       # Publication-quality vector/high-res visual artifacts
│   ├── architecture_diagram.png   # Full multi-stream architecture schematic
│   ├── pipeline_workflow.png      # End-to-end video inference & voting workflow
│   ├── benchmark_roc_curves.png   # Multi-dataset ROC-AUC comparative curves
│   ├── cross_dataset_comparison.png # Video voting gain bar charts
│   ├── training_curves_unified.png # Loss, accuracy, and gating parameter curves
│   └── confusion_matrices/        # Empirical confusion matrices
│       ├── ffpp_video_cm.png      # FaceForensics++ video confusion matrix
│       ├── celebdf_video_cm.png   # Celeb-DF v2 cross-dataset matrix
│       ├── dfdc_video_cm.png      # DFDC cross-dataset matrix
│       └── ciplab_still_cm.png    # CIPLab negative control matrix
│
├── src/                           # Clean modular PyTorch library code
│   ├── models/
│   │   ├── effresnet_unified.py   # Final novel unified multi-stream model
│   │   ├── effresnet_vit.py       # Baseline reproduction of IEEE SMART-2025 paper
│   │   ├── effresnet_gated_srm.py # Gated SRM steganalysis ablation stream
│   │   └── effresnet_dssfca.py    # DSSFCA frequency ablation stream
│   ├── data/
│   │   └── dataset.py             # Video-level grouped dataset loader (zero leakage)
│   ├── training/
│   │   └── trainer.py             # Mixed-precision (AMP) training engine with cosine scheduler
│   └── utils/
│       └── metrics.py             # Evaluation metrics (Accuracy, AUC, Precision, F1)
│
├── benchmarks/                    # Raw empirical results from Kaggle GPU runs
│   ├── faceforensics/             # In-domain FaceForensics++ benchmark metrics
│   ├── celeb_df/                  # Celeb-DF v2 cross-dataset evaluation summary
│   ├── dfdc/                      # DFDC cross-dataset evaluation summary
│   └── ciplab/                    # CIPLab still-morph negative control summary
│
├── notebooks/                     # Kaggle-reproducible Jupyter notebooks (1-to-1 runnable)
│   ├── 01_baseline_ffpp_training.ipynb
│   ├── 02_unified_ffpp_training.ipynb
│   ├── 03_temporal_voting_eval.ipynb
│   ├── 04_celebdf_v2_eval.ipynb
│   ├── 05_dfdc_eval.ipynb
│   └── 06_ciplab_eval.ipynb
│
└── sample_videos/                 # Real & fake test video samples for quick testing
    ├── sample_FAKE.mp4            # Face-cropped deepfake sample (DFDC)
    ├── realvideo.mp4              # Pristine interview video sample
    └── fakevideo.mp4              # Full-scene deepfake sample
```

---

## ⚡ Quick Start: Running Video Inference

You can run deepfake detection on any test video directly from the command line:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run inference on a test video
python predict.py \
  --video sample_videos/sample_FAKE.mp4 \
  --checkpoint checkpoints/best_effresnet_unified.pth \
  --save_frames
```

### Output:
* **Terminal Summary:** Overall verdict (`REAL` vs `FAKE`), mean probability, and confidence score.
* **`output/<video_name>_result.txt`:** One-line executive summary.
* **`output/<video_name>_report.json`:** Comprehensive JSON report with per-frame probabilities and multi-paradigm voting verdicts.
* **`output/<video_name>_frames/`:** Annotated frames with detection labels and confidence banners.

---

## 🔬 Benchmark Results Summary

### In-Domain Performance (FaceForensics++ c23)
| Model | Frame Acc (%) | Frame AUC (%) | Video Acc (%) | Video AUC (%) | Video Precision (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Paper (SMART-2025)** | 96.60 | 98.40 | — | 98.40 | 97.20 |
| **EFFResNet-ViT (Baseline)** | 90.31 | 96.62 | 93.81 | 98.09 | 98.06 |
| **EFFResNet-Unified (Ours)** | **90.79** | **97.10** | **95.87** | **98.58** *(beats paper)* | **98.00** *(beats paper)* |

### Cross-Dataset Zero-Shot Generalization (Unseen Domains)
| Benchmark Dataset | Challenge / Domain | Video Acc (%) | Video AUC (%) | Published Literature Baseline |
| :--- | :--- | :---: | :---: | :--- |
| **Celeb-DF v2** | High-Quality Swaps | **77.76%** | **80.68%** | Xception: 65.5% \| F3-Net: 78.4% |
| **DFDC** | Wild / Heavy Compression | **60.05%** | **59.60%** | Xception: 53.8% \| Face X-Ray: 58.6% |
| **CIPLab** | 2D Photographic Morphs | 52.67% | 52.65% | Negative Control (Domain Specific) |

---

## 📜 Academic Reference & Base Paper

```bibtex
@inproceedings{sharma2025deepfake,
  title={Deepfake Video Detection: A Comprehensive Deep Learning-Based Approach},
  author={Sharma, Rinku and Sharma, Ashish},
  booktitle={2025 IEEE 7th International Conference on Smart Technologies and Management (SMART)},
  year={2025},
  doi={10.1109/SMART66937.2025.11389345}
}
```
