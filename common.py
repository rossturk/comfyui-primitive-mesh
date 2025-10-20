"""
Common utility functions shared between CPU and GPU implementations.
"""

import numpy as np
from typing import Tuple, Dict

try:
    from .constants import COLOR_MIN, COLOR_MAX, RGB_CHANNELS
except ImportError:
    from constants import COLOR_MIN, COLOR_MAX, RGB_CHANNELS


def clamp(x: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, x))


def clamp_color(x: float) -> int:
    """Clamp color value to valid range [0, 255]."""
    return int(clamp(x, COLOR_MIN, COLOR_MAX))


def distance_to_difference(distance: float, pixels: int) -> float:
    """Convert distance metric to difference sum."""
    return (distance * COLOR_MAX) ** 2 * (RGB_CHANNELS * pixels)


def difference_to_distance(difference: float, pixels: int) -> float:
    """Convert difference sum to distance metric."""
    if pixels == 0:
        return 0.0
    # Clamp difference to non-negative to avoid sqrt of negative number
    # (can occur due to floating point precision issues)
    difference = max(0.0, difference)
    return np.sqrt(difference / (RGB_CHANNELS * pixels)) / COLOR_MAX


def compute_valid_regions(
    offset: Dict[str, int],
    shape_height: int,
    shape_width: int,
    frame_height: int,
    frame_width: int
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], bool]:
    """
    Compute valid overlapping regions between shape and frame.

    Args:
        offset: Dictionary with 'top', 'left' keys
        shape_height: Height of shape bounding box
        shape_width: Width of shape bounding box
        frame_height: Height of target/current frame
        frame_width: Width of target/current frame

    Returns:
        Tuple of:
            - shape_region: (y_start, y_end, x_start, x_end) in shape coordinates
            - frame_region: (y_start, y_end, x_start, x_end) in frame coordinates
            - is_valid: True if regions overlap, False otherwise
    """
    top = offset['top']
    left = offset['left']

    # Calculate valid overlap region
    shape_y_start = max(0, -top)
    shape_y_end = min(shape_height, frame_height - top)
    shape_x_start = max(0, -left)
    shape_x_end = min(shape_width, frame_width - left)

    frame_y_start = max(0, top)
    frame_y_end = min(frame_height, top + shape_height)
    frame_x_start = max(0, left)
    frame_x_end = min(frame_width, left + shape_width)

    # Check if valid overlap exists
    is_valid = (
        shape_y_start < shape_y_end and
        shape_x_start < shape_x_end and
        frame_y_start < frame_y_end and
        frame_x_start < frame_x_end
    )

    shape_region = (shape_y_start, shape_y_end, shape_x_start, shape_x_end)
    frame_region = (frame_y_start, frame_y_end, frame_x_start, frame_x_end)

    return shape_region, frame_region, is_valid
