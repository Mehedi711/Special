import torch
import torch.nn.functional as F

def explanation_guided_loss(gradcam_map, mask):
    # gradcam_map and mask should be normalized and same shape
    gradcam_map = gradcam_map / (gradcam_map.sum() + 1e-8)
    mask = mask / (mask.sum() + 1e-8)
    kl_div = F.kl_div(gradcam_map.log(), mask, reduction='batchmean')
    return kl_div
