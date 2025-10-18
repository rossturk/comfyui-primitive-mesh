"""
Utility functions for color computation and distance metrics.
Ported from util.js to Python with NumPy.
"""

import numpy as np
from typing import Tuple


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
    # Clamp difference to non-negative to avoid sqrt of negative number
    # (can occur due to floating point precision issues)
    difference = max(0.0, difference)
    return np.sqrt(difference / (3 * pixels)) / 255


def difference(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate sum of squared differences between two image arrays.

    Args:
        data1: First image array (H, W, C)
        data2: Second image array (H, W, C)

    Returns:
        Sum of squared differences across RGB channels
    """
    # Only compare RGB channels (not alpha)
    diff = data2[:, :, :3] - data1[:, :, :3]
    return np.sum(diff * diff)


def compute_color(offset: dict, image_data: dict, alpha: float) -> np.ndarray:
    """
    Compute optimal RGB color for a shape (vectorized version).

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' arrays
        alpha: Shape alpha/transparency value

    Returns:
        RGB color as numpy array [R, G, B]
    """
    shape_data = image_data['shape']
    current_data = image_data['current']
    target_data = image_data['target']

    sh, sw = shape_data.shape[:2]
    fh, fw = current_data.shape[:2]

    # Calculate valid region bounds (clip to image boundaries)
    top = offset['top']
    left = offset['left']

    # Calculate valid overlap region
    shape_y_start = max(0, -top)
    shape_y_end = min(sh, fh - top)
    shape_x_start = max(0, -left)
    shape_x_end = min(sw, fw - left)

    frame_y_start = max(0, top)
    frame_y_end = min(fh, top + sh)
    frame_x_start = max(0, left)
    frame_x_end = min(fw, left + sw)

    # Handle out of bounds cases
    if shape_y_start >= shape_y_end or shape_x_start >= shape_x_end:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Extract overlapping regions
    shape_region = shape_data[shape_y_start:shape_y_end, shape_x_start:shape_x_end]
    current_region = current_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
    target_region = target_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

    # Create mask where shape is drawn (alpha > 0)
    mask = shape_region[:, :, 3] > 0

    count = np.sum(mask)
    if count == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Vectorized color computation
    # Formula: color[c] = ((target - current) / alpha + current)
    # Convert to float64 to avoid overflow
    target_rgb = target_region[:, :, :3].astype(np.float64)
    current_rgb = current_region[:, :, :3].astype(np.float64)

    # Compute optimal color for each RGB channel
    color_contribution = (target_rgb - current_rgb) / alpha + current_rgb

    # Apply mask and sum (using broadcasting)
    mask_3d = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting
    color_sum = np.sum(color_contribution * mask_3d, axis=(0, 1))  # Sum over H and W

    # Average and clamp
    color = color_sum / count
    return np.clip(color, 0, 255).astype(np.uint8)


def compute_difference_change(offset: dict, image_data: dict, color: np.ndarray) -> float:
    """
    Compute change in difference when adding a shape with given color (vectorized version).

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' arrays
        color: RGB color array [R, G, B]

    Returns:
        Change in difference metric
    """
    shape_data = image_data['shape']
    current_data = image_data['current']
    target_data = image_data['target']

    sh, sw = shape_data.shape[:2]
    fh, fw = current_data.shape[:2]

    # Calculate valid region bounds (clip to image boundaries)
    top = offset['top']
    left = offset['left']

    # Calculate valid overlap region
    shape_y_start = max(0, -top)
    shape_y_end = min(sh, fh - top)
    shape_x_start = max(0, -left)
    shape_x_end = min(sw, fw - left)

    frame_y_start = max(0, top)
    frame_y_end = min(fh, top + sh)
    frame_x_start = max(0, left)
    frame_x_end = min(fw, left + sw)

    # Handle out of bounds cases
    if shape_y_start >= shape_y_end or shape_x_start >= shape_x_end:
        return 0.0

    # Extract overlapping regions
    shape_region = shape_data[shape_y_start:shape_y_end, shape_x_start:shape_x_end]
    current_region = current_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
    target_region = target_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

    # Get alpha values and create mask
    alpha_channel = shape_region[:, :, 3].astype(np.float64) / 255.0  # (H, W)
    mask = alpha_channel > 0

    if not np.any(mask):
        return 0.0

    # Convert to float64 for precision
    target_rgb = target_region[:, :, :3].astype(np.float64)
    current_rgb = current_region[:, :, :3].astype(np.float64)
    color_f64 = color.astype(np.float64)

    # Vectorized computation
    # a = alpha, b = 1 - alpha
    # d1 = target - current
    # d2 = target - (color * a + current * b)

    # Expand alpha to 3 channels for RGB operations (H, W, 3)
    alpha_3d = alpha_channel[:, :, np.newaxis]
    beta_3d = 1.0 - alpha_3d

    # Compute differences (only where mask is True matters, but compute everywhere for vectorization)
    d1 = target_rgb - current_rgb
    new_pixel = color_f64 * alpha_3d + current_rgb * beta_3d
    d2 = target_rgb - new_pixel

    # Compute squared differences
    d1_squared = d1 * d1  # (H, W, 3)
    d2_squared = d2 * d2  # (H, W, 3)

    # Apply mask (only sum where shape is drawn)
    mask_3d = mask[:, :, np.newaxis]
    sum_diff = np.sum((d2_squared - d1_squared) * mask_3d)

    return sum_diff


def compute_color_and_difference_change(offset: dict, image_data: dict, alpha: float) -> Tuple[np.ndarray, float]:
    """
    Compute optimal color and resulting difference change (optimized combined version).

    This function combines both computations to avoid redundant array operations
    and region extraction, providing significant performance improvements.

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' arrays
        alpha: Shape alpha/transparency value

    Returns:
        Tuple of (rgb color array, difference change)
    """
    shape_data = image_data['shape']
    current_data = image_data['current']
    target_data = image_data['target']

    sh, sw = shape_data.shape[:2]
    fh, fw = current_data.shape[:2]

    # Calculate valid region bounds (clip to image boundaries)
    top = offset['top']
    left = offset['left']

    # Calculate valid overlap region
    shape_y_start = max(0, -top)
    shape_y_end = min(sh, fh - top)
    shape_x_start = max(0, -left)
    shape_x_end = min(sw, fw - left)

    frame_y_start = max(0, top)
    frame_y_end = min(fh, top + sh)
    frame_x_start = max(0, left)
    frame_x_end = min(fw, left + sw)

    # Handle out of bounds cases
    if shape_y_start >= shape_y_end or shape_x_start >= shape_x_end:
        return np.array([0, 0, 0], dtype=np.uint8), 0.0

    # Extract overlapping regions (only done once, shared by both computations)
    shape_region = shape_data[shape_y_start:shape_y_end, shape_x_start:shape_x_end]
    current_region = current_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]
    target_region = target_data[frame_y_start:frame_y_end, frame_x_start:frame_x_end]

    # Create mask where shape is drawn (alpha > 0)
    alpha_channel = shape_region[:, :, 3].astype(np.float64) / 255.0  # (H, W)
    mask = alpha_channel > 0

    count = np.sum(mask)
    if count == 0:
        return np.array([0, 0, 0], dtype=np.uint8), 0.0

    # Convert to float64 for precision (shared arrays)
    target_rgb = target_region[:, :, :3].astype(np.float64)
    current_rgb = current_region[:, :, :3].astype(np.float64)

    # ========== COMPUTE OPTIMAL COLOR ==========
    # Formula: color[c] = ((target - current) / alpha + current)
    color_contribution = (target_rgb - current_rgb) / alpha + current_rgb

    # Apply mask and sum (using broadcasting)
    mask_3d = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting
    color_sum = np.sum(color_contribution * mask_3d, axis=(0, 1))  # Sum over H and W

    # Average and clamp
    color = color_sum / count
    rgb = np.clip(color, 0, 255).astype(np.uint8)

    # ========== COMPUTE DIFFERENCE CHANGE ==========
    # Now compute difference change using the computed color
    color_f64 = rgb.astype(np.float64)

    # Expand alpha to 3 channels for RGB operations (H, W, 3)
    # For difference computation, we need the actual alpha from the shape
    alpha_3d = alpha_channel[:, :, np.newaxis]
    beta_3d = 1.0 - alpha_3d

    # Compute differences
    # d1 = target - current (before adding shape)
    # d2 = target - (color * alpha + current * beta) (after adding shape)
    d1 = target_rgb - current_rgb
    new_pixel = color_f64 * alpha_3d + current_rgb * beta_3d
    d2 = target_rgb - new_pixel

    # Compute squared differences
    d1_squared = d1 * d1  # (H, W, 3)
    d2_squared = d2 * d2  # (H, W, 3)

    # Apply mask (only sum where shape is drawn)
    difference_change = np.sum((d2_squared - d1_squared) * mask_3d)

    return rgb, difference_change
