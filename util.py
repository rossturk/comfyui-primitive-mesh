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
    Compute optimal RGB color for a shape.

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' arrays
        alpha: Shape alpha/transparency value

    Returns:
        RGB color as numpy array [R, G, B]
    """
    color = np.zeros(3, dtype=np.float64)
    shape_data = image_data['shape']
    current_data = image_data['current']
    target_data = image_data['target']

    sh, sw = shape_data.shape[:2]
    fh, fw = current_data.shape[:2]

    count = 0

    for sy in range(sh):
        fy = sy + offset['top']
        if fy < 0 or fy >= fh:
            continue

        for sx in range(sw):
            fx = offset['left'] + sx
            if fx < 0 or fx >= fw:
                continue

            # Only where drawn (check alpha channel)
            if shape_data[sy, sx, 3] == 0:
                continue

            # Accumulate color differences (cast to float to avoid overflow)
            for c in range(3):  # RGB channels
                target_val = float(target_data[fy, fx, c])
                current_val = float(current_data[fy, fx, c])
                color[c] += (target_val - current_val) / alpha + current_val

            count += 1

    if count == 0:
        return np.array([0, 0, 0], dtype=np.uint8)

    # Average and clamp
    color = color / count
    return np.array([clamp_color(c) for c in color], dtype=np.uint8)


def compute_difference_change(offset: dict, image_data: dict, color: np.ndarray) -> float:
    """
    Compute change in difference when adding a shape with given color.

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

    sum_diff = 0.0

    for sy in range(sh):
        fy = sy + offset['top']
        if fy < 0 or fy >= fh:
            continue

        for sx in range(sw):
            fx = offset['left'] + sx
            if fx < 0 or fx >= fw:
                continue

            a = shape_data[sy, sx, 3]
            if a == 0:
                continue

            a = a / 255.0
            b = 1.0 - a

            # Calculate differences before and after (cast to float to avoid overflow)
            target_pixel = target_data[fy, fx, :3].astype(np.float64)
            current_pixel = current_data[fy, fx, :3].astype(np.float64)

            d1 = target_pixel - current_pixel
            d2 = target_pixel - (color * a + current_pixel * b)

            # Update sum
            sum_diff -= np.sum(d1 * d1)
            sum_diff += np.sum(d2 * d2)

    return sum_diff


def compute_color_and_difference_change(offset: dict, image_data: dict, alpha: float) -> Tuple[np.ndarray, float]:
    """
    Compute optimal color and resulting difference change.

    Args:
        offset: Dictionary with 'top', 'left', 'width', 'height'
        image_data: Dictionary with 'shape', 'current', 'target' arrays
        alpha: Shape alpha/transparency value

    Returns:
        Tuple of (rgb color array, difference change)
    """
    rgb = compute_color(offset, image_data, alpha)
    difference_change = compute_difference_change(offset, image_data, rgb)
    return rgb, difference_change
