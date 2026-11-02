"""Identical differentiable alignment for cached references and predictions."""
import numpy as np
import torch
from torch.nn import functional as F

TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def estimate_alignment(landmarks):
    from skimage.transform import SimilarityTransform
    transform = SimilarityTransform()
    if not transform.estimate(np.asarray(landmarks), TEMPLATE):
        raise ValueError('Cannot estimate face alignment')
    # Output pixel coordinates -> source pixel coordinates.
    return torch.as_tensor(np.linalg.inv(transform.params)[:2], dtype=torch.float32)


def align_faces(images, inverse_matrices):
    """RGB [N,3,H,W], pixel-coordinate inverse transforms [N,2,3]."""
    n, _, height, width = images.shape
    if inverse_matrices.shape != (n, 2, 3):
        raise ValueError('Alignment matrices must have shape [N,2,3]')
    with torch.autocast(device_type=images.device.type, enabled=False):
        y, x = torch.meshgrid(torch.arange(112, device=images.device, dtype=torch.float32),
                              torch.arange(112, device=images.device, dtype=torch.float32), indexing='ij')
        pixels = torch.stack((x, y, torch.ones_like(x)), -1).reshape(-1, 3)
        source = torch.einsum('nij,pj->npi', inverse_matrices.float(), pixels)
        grid = torch.stack(((source[..., 0] + .5) * 2 / width - 1,
                            (source[..., 1] + .5) * 2 / height - 1), -1)
        return F.grid_sample(images.float(), grid.reshape(n, 112, 112, 2),
                             mode='bilinear', padding_mode='zeros', align_corners=False)
