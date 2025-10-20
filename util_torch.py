"""
Torch/CUDA accelerated utility functions for color computation and distance metrics.
GPU-accelerated version of util.py for significant performance improvements.
"""

import torch
import numpy as np
from typing import Tuple, Optional

try:
    from .common import (
        clamp,
        clamp_color,
        distance_to_difference,
        difference_to_distance,
        compute_valid_regions
    )
except ImportError:
    from common import (
        clamp,
        clamp_color,
        distance_to_difference,
        difference_to_distance,
        compute_valid_regions
    )


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_color_and_difference_change_torch(
    offset: dict,
    image_data: dict,
    alpha: float,
    device: torch.device
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal color and difference change (GPU-accelerated combined version).

    This is the main performance-critical function. All tensor operations happen on GPU.

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' as torch tensors
        alpha: Shape alpha/transparency value
        device: torch device to use

    Returns:
        Tuple of (rgb color array, difference change)
    """
    shape_data = image_data['shape']  # Already on GPU
    current_data = image_data['current']  # Already on GPU
    target_data = image_data['target']  # Already on GPU

    sh, sw = shape_data.shape[:2]
    fh, fw = current_data.shape[:2]

    # Calculate valid region bounds using common utility
    (shape_y_start, shape_y_end, shape_x_start, shape_x_end), \
    (frame_y_start, frame_y_end, frame_x_start, frame_x_end), \
    is_valid = compute_valid_regions(offset, sh, sw, fh, fw)

    # Handle out of bounds
    if not is_valid:
        return np.array([0, 0, 0], dtype=np.uint8), 0.0

    # Extract overlapping regions (single slice operation, stays on GPU)
    shape_region = shape_data[shape_y_start:shape_y_end, shape_x_start:shape_x_end]
    current_region = current_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
    target_region = target_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

    # Create mask (on GPU)
    alpha_channel = shape_region[:, :, 3].float() / 255.0
    mask = alpha_channel > 0

    count = torch.sum(mask).item()
    if count == 0:
        return np.array([0, 0, 0], dtype=np.uint8), 0.0

    # Convert to float (on GPU)
    target_rgb = target_region[:, :, :3].float()
    current_rgb = current_region[:, :, :3].float()

    # ========== COMPUTE OPTIMAL COLOR (GPU) ==========
    color_contribution = (target_rgb - current_rgb) / alpha + current_rgb

    mask_3d = mask.unsqueeze(2)
    color_sum = torch.sum(color_contribution * mask_3d, dim=(0, 1))

    color = color_sum / count
    rgb = torch.clamp(color, 0, 255)

    # ========== COMPUTE DIFFERENCE CHANGE (GPU) ==========
    # Reuse already-computed tensors
    alpha_3d = alpha_channel.unsqueeze(2)
    beta_3d = 1.0 - alpha_3d

    # Compute differences
    d1 = target_rgb - current_rgb
    new_pixel = rgb * alpha_3d + current_rgb * beta_3d
    d2 = target_rgb - new_pixel

    # Squared differences
    d1_squared = d1 * d1
    d2_squared = d2 * d2

    # Apply mask and sum
    difference_change = torch.sum((d2_squared - d1_squared) * mask_3d).item()

    # Only move final color back to CPU/numpy
    rgb_np = rgb.cpu().numpy().astype(np.uint8)

    return rgb_np, difference_change
