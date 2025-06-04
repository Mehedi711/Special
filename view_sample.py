import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Paths (update these if your dataset is in a different location)
train_img_dir = '/workspaces/Special/kmms_test/kmms_test/images'
train_mask_dir = '/workspaces/Special/kmms_test/kmms_test/masks'

# Check if directories exist
if not os.path.isdir(train_img_dir):
    raise FileNotFoundError(f"Image directory not found: {train_img_dir}")
if not os.path.isdir(train_mask_dir):
    raise FileNotFoundError(f"Mask directory not found: {train_mask_dir}")

# List image and mask files (filter for common image formats)
img_files = sorted([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
mask_files = sorted([f for f in os.listdir(train_mask_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])

if not img_files:
    raise FileNotFoundError(f"No image files found in {train_img_dir}")
if not mask_files:
    raise FileNotFoundError(f"No mask files found in {train_mask_dir}")

# Try to find a mask with the same name as the image (without extension)
sample_img = img_files[0]
img_name, img_ext = os.path.splitext(sample_img)
# Find mask with same stem (ignoring extension)
sample_mask = next((m for m in mask_files if os.path.splitext(m)[0] == img_name), mask_files[0])

# Load images
img = Image.open(os.path.join(train_img_dir, sample_img))
mask = Image.open(os.path.join(train_mask_dir, sample_mask))

# Convert mask to single channel and binarize for visualization
if mask.mode != 'L':
    mask = mask.convert('L')
mask_bin = (np.array(mask) > 0).astype('uint8') * 255

# Display
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img)
axs[0].set_title(f'Image: {sample_img}')
axs[0].axis('off')
axs[1].imshow(mask_bin, cmap='gray')
axs[1].set_title(f'Mask: {sample_mask}')
axs[1].axis('off')
plt.tight_layout()
plt.show()
fig.savefig("sample_output.png")
print("Saved sample_output.png")

print("Sample image path:", os.path.join(train_img_dir, sample_img))
print("Sample mask path:", os.path.join(train_mask_dir, sample_mask))
print("Image size:", img.size)
print("Mask size:", mask.size)
