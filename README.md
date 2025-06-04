# Special: Nuclei Segmentation with Attention U-Net

## Overview
This project performs nuclei segmentation on the MoNuSeg dataset using a U-Net with a ResNet34 backbone, attention, deep supervision, SE blocks, and advanced data augmentation. It supports flexible loss functions, early stopping, TTA, ensembling, and XAI (Grad-CAM) visualizations.

## Project Structure
```
Special/
├── scripts/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation, metrics, and visualization
│   ├── monuseg_dataset.py    # Dataset and augmentations
│   ├── xai_gradcam.py        # Grad-CAM visualization
│   └── config.py             # Paths and hyperparameters
├── models/
│   └── unet_resnet34_attention.py  # Model definition
├── outputs/
│   ├── best_model.pth        # Best model checkpoint
│   ├── metrics.csv           # Per-image metrics
│   ├── metrics_summary.csv   # Summary statistics
│   ├── metrics_boxplot.png   # Boxplot of metrics
│   ├── vis_*.png             # Qualitative results
│   └── gradcam/              # Grad-CAM visualizations
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Install Captum for XAI:
   ```bash
   pip install captum
   ```

## Usage
- **Train the model:**
  ```bash
  python scripts/train.py
  ```
- **Evaluate the model:**
  ```bash
  python scripts/evaluate.py
  ```
- **Run Grad-CAM visualization:**
  ```bash
  python scripts/xai_gradcam.py
  ```
- **Check outputs:**
  - Model checkpoints, metrics, and sample predictions: `outputs/`
  - Grad-CAM visualizations: `outputs/gradcam/`

## Configuration
Edit `scripts/config.py` to change data paths, batch size, epochs, etc.

## Results
- **Per-image metrics:** `outputs/metrics.csv`
- **Summary statistics:** `outputs/metrics_summary.csv`, `outputs/metrics_summary_latex.tex`
- **Boxplot:** `outputs/metrics_boxplot.png`
- **Qualitative results:** `outputs/vis_*.png`

## LaTeX Table Example
See `outputs/metrics_summary_latex.tex` for a ready-to-use table:
```latex
\begin{table}[ht]
\centering
\caption{Segmentation Performance on MoNuSeg Test Set}
\begin{tabular}{lcc}
\toprule
Metric & Mean & Std \\
\midrule
Dice & ... & ... \\
IoU & ... & ... \\
\bottomrule
\end{tabular}
\label{tab:segmentation_metrics}
\end{table}
```

## Notes
- Ensure your data is in the correct folder structure as set in `config.py`.
- For best results, experiment with augmentations and loss weights.
- All outputs are saved in the `outputs/` directory for easy access and publication.