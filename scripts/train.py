import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet_resnet34_attention import UNetResNet34
from scripts.monuseg_dataset import NucleiSegmentationDataset, train_transform
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
import random
from scripts.config import TRAIN_IMG_DIR, TRAIN_MASK_DIR, OUTPUT_DIR, PATCH_SIZE, BATCH_SIZE, NUM_WORKERS, EPOCHS, LEARNING_RATE

# Set up logging
logging.basicConfig(filename=os.path.join(OUTPUT_DIR, 'train.log'), level=logging.INFO, format='%(asctime)s %(message)s')

# Extra: log to console as well for deeper debug
def log_and_print(msg):
    print(msg)
    logging.info(msg)

# Update these paths for your workspace
train_img_dir = TRAIN_IMG_DIR
train_mask_dir = TRAIN_MASK_DIR

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dataset = NucleiSegmentationDataset(train_img_dir, train_mask_dir, patch_size=PATCH_SIZE, transform=train_transform)
n = len(dataset)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_size = n - train_size - val_size
train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    def forward(self, pred, target):
        # Dice loss as before
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        dice = 1 - ((2. * intersection + 1.) /
                    (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + 1.))
        dice = dice.mean()
        bce = self.bce(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice

# Optionally, Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, pred, target):
        bce = nn.BCELoss(reduction='none')(pred, target)
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

def save_predictions(images, masks, outputs, epoch, num_samples=3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(min(num_samples, images.size(0))):
        img = images[i].cpu().numpy().transpose(1,2,0)
        img = (img * 255).clip(0,255).astype(np.uint8)  # Fix: proper visualization
        mask = masks[i].cpu().numpy().squeeze()
        pred = outputs[i].cpu().numpy().squeeze()
        fig, axs = plt.subplots(1,3, figsize=(9,3))
        axs[0].imshow(img)
        axs[0].set_title('Image')
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title('Mask')
        axs[2].imshow(pred > 0.5, cmap='gray')
        axs[2].set_title('Prediction')
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'epoch_{epoch+1}_sample_{i}.png'))
        plt.close()

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = 1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return loss.mean()

def iou_score(pred, target, threshold=0.5, smooth=1.):
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()

def train_model(train_loader, val_loader, device):
    model = UNetResNet34(deep_supervision=True, dropout_p=0.3, se_reduction=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    best_dice = 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patience = 10
    patience_counter = 0
    best_epoch = 0

    # Log model summary
    log_and_print(str(model))
    log_and_print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader)):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            # Deep supervision: outputs is a list during training
            if isinstance(outputs, list):
                loss = 0
                for out in outputs:
                    loss += criterion(out, masks)
                loss /= len(outputs)
                main_output = outputs[0]
            else:
                loss = criterion(outputs, masks)
                main_output = outputs
            # Debug: log tensor shapes and stats (commented out for clean logs)
            # if epoch == 0 and batch_idx < 2:
            #     if isinstance(outputs, list):
            #         log_and_print(f"Batch {batch_idx} imgs shape: {imgs.shape}, masks shape: {masks.shape}, outputs[0] shape: {outputs[0].shape}")
            #         log_and_print(f"outputs[0] min/max/mean: {outputs[0].min().item():.4f}/{outputs[0].max().item():.4f}/{outputs[0].mean().item():.4f}")
            #     else:
            #         log_and_print(f"Batch {batch_idx} imgs shape: {imgs.shape}, masks shape: {masks.shape}, outputs shape: {outputs.shape}")
            #         log_and_print(f"outputs min/max/mean: {outputs.min().item():.4f}/{outputs.max().item():.4f}/{outputs.mean().item():.4f}")
            # Check for NaNs/Infs in outputs (commented out for clean logs)
            # if isinstance(outputs, list):
            #     for idx, out in enumerate(outputs):
            #         if not torch.isfinite(out).all():
            #             log_and_print(f"Non-finite values in outputs[{idx}] at epoch {epoch+1}, batch {batch_idx}")
            #             raise ValueError('Non-finite values in outputs!')
            # else:
            #     if not torch.isfinite(outputs).all():
            #         log_and_print(f"Non-finite values in outputs at epoch {epoch+1}, batch {batch_idx}")
            #         raise ValueError('Non-finite values in outputs!')
            # if not torch.isfinite(loss):
            #     log_and_print(f'Non-finite loss detected at epoch {epoch+1}, batch {batch_idx}')
            #     raise ValueError('Loss is NaN or Inf!')
            loss.backward()
            # Check for NaNs/Infs in gradients (commented out for clean logs)
            # for name, param in model.named_parameters():
            #     if param.grad is not None and not torch.isfinite(param.grad).all():
            #         log_and_print(f'Non-finite gradient in {name} at epoch {epoch+1}, batch {batch_idx}')
            #         raise ValueError(f'Non-finite gradient in {name}!')
            optimizer.step()
            train_loss += loss.item()
        scheduler.step(epoch + 1)

        # Validation
        model.eval()
        val_dice = 0
        val_iou = 0
        val_batches = 0
        sample_images, sample_masks, sample_outputs = None, None, None
        with torch.no_grad():
            for batch_idx, (imgs, masks) in enumerate(val_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                if isinstance(outputs, list):
                    outputs = outputs[0]
                # Debug: log tensor stats for validation (commented out for clean logs)
                # if epoch == 0 and batch_idx < 2:
                #     log_and_print(f"[VAL] Batch {batch_idx} imgs shape: {imgs.shape}, masks shape: {masks.shape}, outputs shape: {outputs.shape}")
                #     log_and_print(f"[VAL] outputs min/max/mean: {outputs.min().item():.4f}/{outputs.max().item():.4f}/{outputs.mean().item():.4f}")
                val_dice += 1 - dice_loss(outputs, masks).item()
                val_iou += iou_score(outputs, masks).item()
                val_batches += 1
                if batch_idx == 0:
                    sample_images, sample_masks, sample_outputs = imgs, masks, outputs
        val_dice /= val_batches
        val_iou /= val_batches
        log_and_print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            log_and_print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_and_print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}, Best Dice: {best_dice:.4f}")
                break

        # Save sample predictions
        if sample_images is not None:
            save_predictions(sample_images, sample_masks, sample_outputs, epoch)

    return model

def tta_predict(model, image, device):
    # Test-time augmentation: horizontal flip, vertical flip, original
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

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(train_loader, val_loader, device)
