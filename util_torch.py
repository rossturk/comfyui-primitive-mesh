"""
Torch/CUDA accelerated utility functions for color computation and distance metrics.
GPU-accelerated version of util.py for significant performance improvements.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, x))


def clamp_color(x: float) -> int:
    """Clamp color value to valid range [0, 255]."""
    return int(clamp(x, 0, 255))


def distance_to_difference(distance: float, pixels: int) -> float:
    """Convert distance metric to difference sum."""
    return (distance * 255) ** 2 * (3 * pixels)


def difference_to_distance(difference: float, pixels: int) -> float:
    """Convert difference sum to distance metric."""
    if pixels == 0:
        return 0.0
    difference = max(0.0, difference)
    return np.sqrt(difference / (3 * pixels)) / 255


def difference_torch(data1: torch.Tensor, data2: torch.Tensor) -> float:
    """
    Calculate sum of squared differences between two image tensors (GPU-accelerated).

    Args:
        data1: First image tensor (H, W, C) on GPU
        data2: Second image tensor (H, W, C) on GPU

    Returns:
        Sum of squared differences across RGB channels
    """
    # Only compare RGB channels (not alpha if present)
    diff = data2[:, :, :3] - data1[:, :, :3]
    return torch.sum(diff * diff).item()


def compute_color_torch(
    offset: dict,
    image_data: dict,
    alpha: float,
    device: torch.device
) -> np.ndarray:
    """
    Compute optimal RGB color for a shape (GPU-accelerated version).

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' as torch tensors
        alpha: Shape alpha/transparency value
        device: torch device to use

    Returns:
        RGB color as numpy array [R, G, B]
    """
    shape_data = image_data['shape']  # torch tensor
    current_data = image_data['current']  # torch tensor
    target_data = image_data['target']  # torch tensor

    sh, sw = shape_data.shape[:2]
    fh, fw = current_data.shape[:2]

    # Calculate valid region bounds
    top = offset['top']
    left = offset['left']

    shape_y_start = max(0, -top)
    shape_y_end = min(sh, fh - top)
    shape_x_start = max(0, -left)
    shape_x_end = min(sw, fw - left)

    frame_y_start = max(0, top)
    frame_y_end = min(fh, top + sh)
    frame_x_start = max(0, left)
    frame_x_end = min(fw, left + sw)

    # Handle out of bounds
    if shape_y_start >= shape_y_end or shape_x_start >= shape_x_end:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Extract overlapping regions (all on GPU)
    shape_region = shape_data[shape_y_start:shape_y_end, shape_x_start:shape_x_end]
    current_region = current_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
    target_region = target_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

    # Create mask where shape is drawn (alpha > 0)
    mask = shape_region[:, :, 3] > 0

    count = torch.sum(mask).item()
    if count == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Vectorized color computation (all on GPU)
    target_rgb = target_region[:, :, :3].float()
    current_rgb = current_region[:, :, :3].float()

    # Compute optimal color for each RGB channel
    color_contribution = (target_rgb - current_rgb) / alpha + current_rgb

    # Apply mask and sum
    mask_3d = mask.unsqueeze(2)  # (H, W, 1) for broadcasting
    color_sum = torch.sum(color_contribution * mask_3d, dim=(0, 1))

    # Average and clamp
    color = color_sum / count
    color_np = torch.clamp(color, 0, 255).cpu().numpy().astype(np.uint8)

    return color_np


def compute_difference_change_torch(
    offset: dict,
    image_data: dict,
    color: np.ndarray,
    device: torch.device
) -> float:
    """
    Compute change in difference when adding a shape (GPU-accelerated version).

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' as torch tensors
        color: RGB color array [R, G, B]
        device: torch device to use

    Returns:
        Change in difference metric
    """
    shape_data = image_data['shape']
    current_data = image_data['current']
    target_data = image_data['target']

    sh, sw = shape_data.shape[:2]
    fh, fw = current_data.shape[:2]

    top = offset['top']
    left = offset['left']

    shape_y_start = max(0, -top)
    shape_y_end = min(sh, fh - top)
    shape_x_start = max(0, -left)
    shape_x_end = min(sw, fw - left)

    frame_y_start = max(0, top)
    frame_y_end = min(fh, top + sh)
    frame_x_start = max(0, left)
    frame_x_end = min(fw, left + sw)

    if shape_y_start >= shape_y_end or shape_x_start >= shape_x_end:
        return 0.0

    # Extract regions (on GPU)
    shape_region = shape_data[shape_y_start:shape_y_end, shape_x_start:shape_x_end]
    current_region = current_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
    target_region = target_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

    # Get alpha values and mask
    alpha_channel = shape_region[:, :, 3].float() / 255.0
    mask = alpha_channel > 0

    if not torch.any(mask):
        return 0.0

    # Convert to float (on GPU)
    target_rgb = target_region[:, :, :3].float()
    current_rgb = current_region[:, :, :3].float()
    color_tensor = torch.tensor(color, dtype=torch.float32, device=device)

    # Vectorized computation (all on GPU)
    alpha_3d = alpha_channel.unsqueeze(2)
    beta_3d = 1.0 - alpha_3d

    # Compute differences
    d1 = target_rgb - current_rgb
    new_pixel = color_tensor * alpha_3d + current_rgb * beta_3d
    d2 = target_rgb - new_pixel

    # Squared differences
    d1_squared = d1 * d1
    d2_squared = d2 * d2

    # Apply mask and sum
    mask_3d = mask.unsqueeze(2)
    difference_change = torch.sum((d2_squared - d1_squared) * mask_3d).item()

    return difference_change


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

    # Calculate valid region bounds
    top = offset['top']
    left = offset['left']

    shape_y_start = max(0, -top)
    shape_y_end = min(sh, fh - top)
    shape_x_start = max(0, -left)
    shape_x_end = min(sw, fw - left)

    frame_y_start = max(0, top)
    frame_y_end = min(fh, top + sh)
    frame_x_start = max(0, left)
    frame_x_end = min(fw, left + sw)

    # Handle out of bounds
    if shape_y_start >= shape_y_end or shape_x_start >= shape_x_end:
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


def numpy_to_torch(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array to torch tensor on specified device."""
    return torch.from_numpy(arr).to(device)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.cpu().numpy()
