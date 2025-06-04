import torch
from captum.attr import LayerGradCam
import matplotlib.pyplot as plt
from models.unet_resnet34_attention import UNetResNet34
from scripts.monuseg_dataset import NucleiSegmentationDataset, train_transform
from torch.utils.data import DataLoader
import os
from scripts.config import OUTPUT_DIR, PATCH_SIZE

# Update these paths for your workspace
test_img_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'kmms_test', 'kmms_test', 'images')
test_mask_dir = os.path.join(os.path.dirname(OUTPUT_DIR), 'kmms_test', 'kmms_test', 'masks')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists('outputs/best_model.pth'):
    raise FileNotFoundError('outputs/best_model.pth not found. Please train the model first.')

dataset = NucleiSegmentationDataset(test_img_dir, test_mask_dir, patch_size=PATCH_SIZE, transform=train_transform)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = UNetResNet34().to(device)
model.load_state_dict(torch.load('outputs/best_model.pth', map_location=device))
model.eval()

# Use the last decoder layer for Grad-CAM
layer = model.conv_last

os.makedirs(os.path.join(OUTPUT_DIR, 'gradcam'), exist_ok=True)
for i, (img, mask) in enumerate(test_loader):
    img = img.to(device)
    gradcam = LayerGradCam(model, layer)
    attributions = gradcam.attribute(img, target=0)
    attributions = torch.nn.functional.interpolate(attributions, size=img.shape[2:], mode='bilinear', align_corners=False)
    heatmap = attributions.cpu().detach().numpy()[0,0]
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(img[0].cpu().permute(1,2,0))
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(mask[0].cpu().squeeze(), cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(img[0].cpu().permute(1,2,0))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'gradcam', f'gradcam_{i}.png'))
    plt.close()
    if i >= 2:
        break
