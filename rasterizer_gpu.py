"""
Pure GPU polygon rasterization using torch operations.
No PIL/CPU dependency - everything runs on GPU.
"""

import torch
import math
from typing import List, Tuple


def point_in_triangle_gpu(px: torch.Tensor, py: torch.Tensor, v0: tuple, v1: tuple, v2: tuple) -> torch.Tensor:
    """
    Vectorized point-in-triangle test using barycentric coordinates.

    Args:
        px: X coordinates tensor (H, W)
        py: Y coordinates tensor (H, W)
        v0, v1, v2: Triangle vertices (x, y)

    Returns:
        Boolean mask (H, W) where True = inside triangle
    """
    # Barycentric coordinate method
    x0, y0 = v0
    x1, y1 = v1
    x2, y2 = v2

    # Compute barycentric coordinates
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)

    if abs(denom) < 1e-10:  # Degenerate triangle
        return torch.zeros_like(px, dtype=torch.bool)

    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    c = 1 - a - b

    # Point is inside if all barycentric coords are in [0, 1]
    return (a >= 0) & (b >= 0) & (c >= 0)


def rasterize_triangle_gpu(points: List[Tuple[int, int]], width: int, height: int, device: torch.device) -> torch.Tensor:
    """
    Rasterize triangle on GPU using vectorized operations.

    Args:
        points: List of 3 (x, y) tuples
        width: Output width
        height: Output height
        device: torch device

    Returns:
        Binary mask (H, W) where True = inside triangle
    """
    if len(points) < 3:
        return torch.zeros((height, width), dtype=torch.bool, device=device)

    # Create coordinate grids (H, W)
    y_coords = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1).expand(height, width)
    x_coords = torch.arange(width, device=device, dtype=torch.float32).view(1, -1).expand(height, width)

    # Point-in-triangle test (fully vectorized)
    mask = point_in_triangle_gpu(x_coords, y_coords, points[0], points[1], points[2])

    return mask


def rasterize_polygon_gpu(points: List[Tuple[int, int]], width: int, height: int, device: torch.device) -> torch.Tensor:
    """
    Rasterize convex polygon on GPU using scanline algorithm.

    Args:
        points: List of (x, y) tuples (convex polygon)
        width: Output width
        height: Output height
        device: torch device

    Returns:
        Binary mask (H, W) where True = inside polygon
    """
    if len(points) < 3:
        return torch.zeros((height, width), dtype=torch.bool, device=device)

    # For convex polygons, triangulate and rasterize each triangle
    # Fan triangulation from first vertex
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)

    for i in range(1, len(points) - 1):
        tri_mask = rasterize_triangle_gpu([points[0], points[i], points[i + 1]], width, height, device)
        mask |= tri_mask

    return mask


def rasterize_ellipse_gpu(center: Tuple[int, int], rx: int, ry: int, rotation: float,
                          width: int, height: int, device: torch.device) -> torch.Tensor:
    """
    Rasterize ellipse on GPU using vectorized distance calculation.

    Args:
        center: (cx, cy) center point
        rx: X radius
        ry: Y radius
        rotation: Rotation angle in radians
        width: Output width
        height: Output height
        device: torch device

    Returns:
        Binary mask (H, W) where True = inside ellipse
    """
    cx, cy = center

    # Create coordinate grids
    y_coords = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1).expand(height, width)
    x_coords = torch.arange(width, device=device, dtype=torch.float32).view(1, -1).expand(height, width)

    # Translate to origin
    x = x_coords - cx
    y = y_coords - cy

    # Rotate coordinates
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    x_rot = x * cos_r + y * sin_r
    y_rot = -x * sin_r + y * cos_r

    # Ellipse equation: (x/rx)^2 + (y/ry)^2 <= 1
    if rx == 0 or ry == 0:
        return torch.zeros((height, width), dtype=torch.bool, device=device)

    dist_sq = (x_rot / rx) ** 2 + (y_rot / ry) ** 2
    mask = dist_sq <= 1.0

    return mask


def create_alpha_mask_gpu(mask: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Convert binary mask to RGBA tensor with alpha channel.

    Args:
        mask: Binary mask (H, W)
        alpha: Alpha value (0-1)

    Returns:
        RGBA tensor (H, W, 4) with uint8 dtype
    """
    height, width = mask.shape
    device = mask.device

    # Create RGBA tensor
    rgba = torch.zeros((height, width, 4), dtype=torch.uint8, device=device)

    # Set alpha channel where mask is True
    alpha_val = int(alpha * 255)
    rgba[mask, 3] = alpha_val

    return rgba


def rasterize_shape_gpu(shape, alpha: float, device: torch.device) -> torch.Tensor:
    """
    Rasterize any shape type on GPU.

    Args:
        shape: Shape object (Triangle, Rectangle, etc.)
        alpha: Alpha transparency value
        device: torch device

    Returns:
        RGBA tensor (H, W, 4) with shape rasterized
    """
    bbox = shape.bbox
    width = max(int(bbox.get('width', 1)), 1)
    height = max(int(bbox.get('height', 1)), 1)

    # Offset to translate shape to bbox origin
    offset_x = bbox['left']
    offset_y = bbox['top']

    # Get shape type
    shape_type = shape.type

    if shape_type in ['Triangle', 'Rectangle', 'RotatedRectangle', 'Quadrilateral']:
        # Polygon-based shapes
        # Translate points to bbox coordinates
        translated_points = [(p[0] - offset_x, p[1] - offset_y) for p in shape.points]
        mask = rasterize_polygon_gpu(translated_points, width, height, device)

    elif shape_type == 'Ellipse':
        # Ellipse shape
        center_translated = (shape.center[0] - offset_x, shape.center[1] - offset_y)
        mask = rasterize_ellipse_gpu(center_translated, shape.rx, shape.ry, shape.rot, width, height, device)

    else:
        # Unknown shape type - fallback to empty
        mask = torch.zeros((height, width), dtype=torch.bool, device=device)

    # Convert mask to RGBA with alpha
    rgba = create_alpha_mask_gpu(mask, alpha)

    return rgba


def batch_rasterize_shapes_gpu(shapes: List, alpha: float, device: torch.device) -> List[torch.Tensor]:
    """
    Batch rasterize multiple shapes on GPU.

    This is more efficient than rasterizing one at a time because:
    1. Kernel launches are amortized
    2. GPU stays saturated
    3. Memory allocations can be batched

    Args:
        shapes: List of shape objects
        alpha: Alpha value
        device: torch device

    Returns:
        List of RGBA tensors
    """
    # For now, process sequentially but on GPU
    # Future: could parallelize across GPU streams
    results = []

    for shape in shapes:
        rgba = rasterize_shape_gpu(shape, alpha, device)
        results.append(rgba)

    return results


def benchmark_rasterization():
    """Benchmark GPU vs CPU rasterization."""
    import time
    from PIL import Image, ImageDraw
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test triangle
    points = [(10, 10), (100, 20), (50, 100)]
    width, height = 256, 256

    # GPU rasterization
    start = time.time()
    for _ in range(1000):
        mask = rasterize_triangle_gpu(points, width, height, device)
        torch.cuda.synchronize()  # Wait for GPU
    gpu_time = time.time() - start

    # CPU rasterization (PIL)
    start = time.time()
    for _ in range(1000):
        img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(img)
        draw.polygon(points, fill=255)
        arr = np.array(img)
    cpu_time = time.time() - start

    print(f"GPU rasterization: {gpu_time:.3f}s (1000 triangles)")
    print(f"CPU rasterization: {cpu_time:.3f}s (1000 triangles)")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")


if __name__ == '__main__':
    benchmark_rasterization()
