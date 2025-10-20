"""
Torch-based shape rasterization for GPU acceleration.
Provides GPU-accelerated rasterization of geometric shapes.
"""

import torch
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw


def rasterize_shape_hybrid(shape, alpha: float, device: torch.device) -> torch.Tensor:
    """
    Rasterize shape using hybrid CPU/GPU approach.

    PIL renders on CPU (fast enough for individual shapes), then we move to GPU.
    This avoids the complexity of implementing polygon rasterization on GPU.

    Args:
        shape: Shape object to rasterize
        alpha: Alpha transparency value
        device: torch device

    Returns:
        RGBA tensor on GPU (H, W, 4)
    """
    # Use PIL to rasterize (CPU) - this is fast enough for individual shapes
    width = int(shape.bbox.get('width', 1))
    height = int(shape.bbox.get('height', 1))

    # Create image and draw context
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Render with translation
    shape._render_translated(draw, -shape.bbox['left'], -shape.bbox['top'], alpha)

    # Convert to numpy then torch (single copy to GPU)
    arr = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(arr).to(device)

    return tensor


# Future optimization ideas for GPU-native rasterization:
# - Custom CUDA kernels for polygon rasterization (using nvdiffrast or similar)
# - Batch rasterization of multiple shapes in parallel on GPU
# - GPU-native ellipse/polygon rendering using compute shaders
#
# Current hybrid approach (CPU PIL -> GPU tensor) works well because:
# 1. Individual shape rasterization is fast on CPU
# 2. The bottleneck is color computation (now GPU-accelerated)
# 3. Moving the rasterized result to GPU is a single copy operation
