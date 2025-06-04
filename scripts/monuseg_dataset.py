import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from skimage import io
import albumentations as A

class NucleiSegmentationDataset(Dataset):
    """
    Dataset for nuclei segmentation with images and PNG masks.
    """
    def __init__(self, img_dir, mask_dir, patch_size=256, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.transform = transform
        # Build a set of mask filenames with spaces removed for robust matching
        mask_files_set = {f.replace(' ', '') for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.tif', '.tiff'))}
        self.img_files = []
        for f in os.listdir(img_dir):
            fname = f.strip()
            if fname.lower().endswith(('.png', '.tif', '.tiff')):
                mask_name = fname.replace(' ', '')  # Remove spaces for matching
                if mask_name in mask_files_set:
                    self.img_files.append(fname)
        # print(f"[DEBUG] Found {len(self.img_files)} valid image/mask pairs.")
        # if self.img_files:
        #     print(f"[DEBUG] Example image files: {self.img_files[:5]}")
        # else:
        #     print("[DEBUG] No valid image/mask pairs found!")
        self.samples = self._generate_patch_indices()
        # print(f"[DEBUG] Number of extracted patches: {len(self.samples)}")

    def _generate_patch_indices(self):
        samples = []
        for img_file in self.img_files:
            if not img_file.lower().endswith(('.png', '.tif', '.tiff')):
                continue  # Only use PNG, TIF, and TIFF images
            img_path = os.path.join(self.img_dir, img_file)
            img = io.imread(img_path)
            h, w = img.shape[:2]
            if h < self.patch_size or w < self.patch_size:
                # print(f"[DEBUG] Skipping {img_file}: shape {img.shape} too small for patch size {self.patch_size}")
                continue
            for y in range(0, h, self.patch_size):
                for x in range(0, w, self.patch_size):
                    if y + self.patch_size <= h and x + self.patch_size <= w:
                        samples.append((img_file, x, y))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, x, y = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_file = img_file.replace(' ', '')
        mask_path = os.path.join(self.mask_dir, mask_file)
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        # Ensure mask is single channel
        if mask.ndim == 3:
            mask = mask[..., 0]
        img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
        mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
        # Ensure img_patch is 3 channels (RGB)
        if img_patch.ndim == 2:
            img_patch = np.stack([img_patch]*3, axis=-1)
        elif img_patch.shape[-1] == 4:
            img_patch = img_patch[..., :3]
        elif img_patch.shape[-1] == 1:
            img_patch = np.repeat(img_patch, 3, axis=-1)
        # Ensure correct patch size
        if img_patch.shape != (self.patch_size, self.patch_size, 3) or mask_patch.shape != (self.patch_size, self.patch_size):
            raise ValueError(f"Patch size mismatch: img_patch {img_patch.shape}, mask_patch {mask_patch.shape}, file {img_file}")
        if self.transform:
            augmented = self.transform(image=img_patch, mask=mask_patch)
            img_patch = augmented['image']
            mask_patch = augmented['mask']
        img_patch = np.transpose(img_patch, (2, 0, 1)) / 255.0
        mask_patch = (mask_patch > 0).astype(np.float32)
        return torch.tensor(img_patch, dtype=torch.float32), torch.tensor(mask_patch, dtype=torch.float32).unsqueeze(0)

# Define augmentations
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.RandomBrightnessContrast(),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2),
])

if __name__ == "__main__":
    # Update these paths for your workspace
    train_img_dir = '/workspaces/Special/kmms_test/kmms_test/images'
    train_mask_dir = '/workspaces/Special/kmms_test/kmms_test/masks'
    # If you have separate test data, set test_img_dir and test_mask_dir accordingly
    # test_img_dir = '/workspaces/Special/kmms_test/kmms_test/images'  # Example
    # test_mask_dir = '/workspaces/Special/kmms_test/kmms_test/masks'  # Example

    dataset = NucleiSegmentationDataset(train_img_dir, train_mask_dir, patch_size=256, transform=train_transform)
    # print(f"[DEBUG] Dataset length: {len(dataset)}")