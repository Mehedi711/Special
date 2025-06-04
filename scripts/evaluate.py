import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.unet_resnet34_attention import UNetResNet34
from scripts.monuseg_dataset import NucleiSegmentationDataset, train_transform
from torch.utils.data import DataLoader, random_split
import os
import glob

def dice_coef(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_coef(pred, target, smooth=1.):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Update these paths for your workspace
test_img_dir = '/workspaces/Special/kmms_test/kmms_test/images'
test_mask_dir = '/workspaces/Special/kmms_test/kmms_test/masks'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('outputs/best_model.pth'):
    raise FileNotFoundError('outputs/best_model.pth not found. Please train the model first.')

dataset = NucleiSegmentationDataset(test_img_dir, test_mask_dir, patch_size=256, transform=None)
n = len(dataset)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size
_, _, test_ds = random_split(dataset, [train_size, val_size, test_size])
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

def tta_predict(model, image, device):
    model.eval()
    image = image.to(device)
    with torch.no_grad():
        preds = []
        for flip in [None, 'h', 'v']:
            if flip == 'h':
                img_aug = torch.flip(image, dims=[3])
            elif flip == 'v':
                img_aug = torch.flip(image, dims=[2])
            else:
                img_aug = image
            out = model(img_aug)
            if isinstance(out, list):
                out = out[0]
            if flip == 'h':
                out = torch.flip(out, dims=[3])
            elif flip == 'v':
                out = torch.flip(out, dims=[2])
            preds.append(out)
        pred = torch.stack(preds, dim=0).mean(dim=0)
    return pred

def load_models(model_paths, device):
    models = []
    for path in model_paths:
        model = UNetResNet34(deep_supervision=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models

def ensemble_predict(models, image, device):
    preds = []
    for model in models:
        pred = tta_predict(model, image, device)
        preds.append(pred)
    return torch.stack(preds, dim=0).mean(dim=0)

model_paths = sorted(glob.glob('outputs/best_model*.pth'))
if not model_paths:
    model_paths = ['outputs/best_model.pth']
models = load_models(model_paths, device)

dice_scores, iou_scores = [], []
os.makedirs('outputs', exist_ok=True)
for i, (imgs, masks) in enumerate(test_loader):
    imgs, masks = imgs.to(device), masks.to(device)
    for j in range(imgs.size(0)):
        pred = ensemble_predict(models, imgs[j:j+1], device)
        dice = dice_coef(pred, masks[j])
        iou = iou_coef(pred, masks[j])
        dice_scores.append(dice.item())
        iou_scores.append(iou.item())
        # Visualization
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(imgs[j].cpu().permute(1,2,0))
        plt.title('Image')
        plt.subplot(1,3,2)
        plt.imshow(masks[j].cpu().squeeze(), cmap='gray')
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(pred.cpu().squeeze().numpy(), cmap='gray')
        plt.title('Prediction')
        plt.savefig(f"outputs/vis_{i}_{j}.png")
        plt.close()
# Save metrics
df = pd.DataFrame({'Dice': dice_scores, 'IoU': iou_scores})
df.to_csv('outputs/metrics.csv', index=False)

# Compute summary statistics
mean_dice = np.mean(dice_scores)
std_dice = np.std(dice_scores)
mean_iou = np.mean(iou_scores)
std_iou = np.std(iou_scores)

# Print and save summary
summary = f"Mean Dice: {mean_dice:.4f} ± {std_dice:.4f}\nMean IoU: {mean_iou:.4f} ± {std_iou:.4f}"
print(summary)
with open('outputs/metrics_summary.txt', 'w') as f:
    f.write(summary + '\n')

# Save summary CSV
summary_df = pd.DataFrame({
    'Metric': ['Dice', 'IoU'],
    'Mean': [mean_dice, mean_iou],
    'Std': [std_dice, std_iou]
})
summary_df.to_csv('outputs/metrics_summary.csv', index=False)

# Boxplot of Dice/IoU
plt.figure(figsize=(6,4))
plt.boxplot([dice_scores, iou_scores], labels=['Dice', 'IoU'])
plt.title('Segmentation Metrics Distribution')
plt.ylabel('Score')
plt.savefig('outputs/metrics_boxplot.png')
plt.close()

# --- Auto-generate LaTeX table with results ---
latex_table = f"""% LaTeX-ready table for segmentation metrics summary
\\begin{{table}}[ht]
\\centering
\\caption{{Segmentation Performance on MoNuSeg Test Set}}
\\begin{{tabular}}{{lcc}}
\\toprule
Metric & Mean & Std \\
\\midrule
Dice & {mean_dice:.4f} & {std_dice:.4f} \\
IoU & {mean_iou:.4f} & {std_iou:.4f} \\
\\bottomrule
\\end{{tabular}}
\\label{{tab:segmentation_metrics}}
\\end{{table}}
"""
with open('outputs/metrics_summary_latex.tex', 'w') as f:
    f.write(latex_table)

print("Evaluation metrics saved to outputs/metrics.csv, summary to outputs/metrics_summary.csv, boxplot to outputs/metrics_boxplot.png, and LaTeX table to outputs/metrics_summary_latex.tex")
