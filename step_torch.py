"""
GPU-accelerated step class for shape optimization.
Performs all color computation and blending on GPU.
"""

import torch
import numpy as np
from typing import TYPE_CHECKING

try:
    from . import util_torch
    from .rasterizer_torch import rasterize_shape_hybrid
    from .constants import MIN_ALPHA, MAX_ALPHA, ALPHA_MUTATION_RANGE
    from .common import clamp
except ImportError:
    import util_torch
    from rasterizer_torch import rasterize_shape_hybrid
    from constants import MIN_ALPHA, MAX_ALPHA, ALPHA_MUTATION_RANGE
    from common import clamp

if TYPE_CHECKING:
    from .shapes import Shape
    from .state_torch import StateTorch


class StepTorch:
    """
    GPU-accelerated optimization step: a shape with color and alpha.

    All heavy computation happens on GPU.
    """

    def __init__(self, shape: 'Shape', cfg: dict, device: torch.device):
        """
        Initialize step.

        Args:
            shape: Shape instance
            cfg: Configuration dictionary
            device: torch device
        """
        import random

        self.shape = shape
        self.cfg = cfg
        self.device = device

        # Randomize alpha within the configured range
        alpha_base = cfg.get('alpha', 0.5)
        alpha_range = cfg.get('alpha_range', 0.0)
        self.alpha = alpha_base + (random.random() - 0.5) * alpha_range
        self.alpha = clamp(self.alpha, MIN_ALPHA, MAX_ALPHA)

        # Computed during compute() call
        self.color = (0, 0, 0)
        self.distance = float('inf')

    def to_svg(self) -> str:
        """Generate SVG representation."""
        self.shape.color = self.color
        self.shape.alpha = self.alpha
        return self.shape.to_svg()

    def apply(self, state: 'StateTorch') -> 'StateTorch':
        """
        Apply this step to a state to create new state (GPU-accelerated).

        Args:
            state: Current state

        Returns:
            New state with this step applied
        """
        try:
            from .state_torch import StateTorch
        except ImportError:
            from state_torch import StateTorch

        # Clone current canvas (on GPU)
        new_canvas = state.current.clone()

        # Draw this step onto it (GPU blending)
        new_canvas = self._draw_step_gpu(new_canvas)

        # Create new state (distance computed on GPU)
        return StateTorch(state.target, new_canvas, self.device)

    def _draw_step_gpu(self, canvas: torch.Tensor) -> torch.Tensor:
        """
        Draw this step onto a canvas tensor using GPU alpha blending.

        Args:
            canvas: Canvas tensor on GPU (H, W, C)

        Returns:
            Modified canvas tensor
        """
        # Rasterize shape (hybrid: CPU raster -> GPU tensor)
        shape_tensor = rasterize_shape_hybrid(self.shape, self.alpha, self.device)

        # Get bounding box
        bbox = self.shape.bbox
        top = bbox['top']
        left = bbox['left']
        height = bbox['height']
        width = bbox['width']

        # Calculate valid region (clip to canvas bounds)
        canvas_h, canvas_w = canvas.shape[:2]

        # Shape region to use
        shape_y_start = max(0, -top)
        shape_y_end = min(height, canvas_h - top)
        shape_x_start = max(0, -left)
        shape_x_end = min(width, canvas_w - left)

        # Canvas region to update
        canvas_y_start = max(0, top)
        canvas_y_end = min(canvas_h, top + height)
        canvas_x_start = max(0, left)
        canvas_x_end = min(canvas_w, left + width)

        # Check for valid overlap
        if (shape_y_start >= shape_y_end or shape_x_start >= shape_x_end or
                canvas_y_start >= canvas_y_end or canvas_x_start >= canvas_x_end):
            return canvas

        # Extract regions
        shape_region = shape_tensor[shape_y_start:shape_y_end, shape_x_start:shape_x_end]
        canvas_region = canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end]

        # GPU alpha blending
        # Shape has RGBA, canvas might be RGB or RGBA
        alpha_mask = shape_region[:, :, 3].unsqueeze(2).float() / 255.0

        # Color with alpha applied
        color_tensor = torch.tensor(self.color, dtype=torch.float32, device=self.device)

        # Blend: result = shape_color * alpha + canvas * (1 - alpha)
        blended = color_tensor * alpha_mask + canvas_region[:, :, :3].float() * (1 - alpha_mask)
        blended = torch.clamp(blended, 0, 255).to(torch.uint8)

        # Update canvas region
        canvas_region_copy = canvas_region.clone()
        canvas_region_copy[:, :, :3] = blended

        # Write back to canvas
        canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end] = canvas_region_copy

        return canvas

    def compute(self, state: 'StateTorch') -> 'StepTorch':
        """
        Find optimal color and compute resulting distance (GPU-accelerated).

        Args:
            state: Current optimization state

        Returns:
            Self (for chaining)
        """
        pixels = state.current.shape[0] * state.current.shape[1]
        offset = self.shape.bbox

        # Rasterize shape (hybrid approach)
        shape_array_tensor = rasterize_shape_hybrid(self.shape, self.alpha, self.device)

        # Prepare image data (all tensors on GPU)
        image_data = {
            'shape': shape_array_tensor,
            'current': state.current,
            'target': state.target
        }

        # Compute optimal color and difference change (GPU-accelerated)
        color, difference_change = util_torch.compute_color_and_difference_change_torch(
            offset, image_data, self.alpha, self.device
        )

        self.color = tuple(color)

        # Compute new distance
        current_difference = util_torch.distance_to_difference(state.distance, pixels)
        new_difference = current_difference + difference_change
        self.distance = util_torch.difference_to_distance(new_difference, pixels)

        return self

    def mutate(self) -> 'StepTorch':
        """
        Create mutated version of this step.

        Returns:
            New mutated step
        """
        import random

        # Mutate shape
        new_shape = self.shape.mutate(self.cfg)

        # Create new step
        mutated = StepTorch(new_shape, self.cfg, self.device)

        # Optionally mutate alpha
        if self.cfg.get('mutateAlpha', False):
            mutated_alpha = self.alpha + (random.random() - 0.5) * ALPHA_MUTATION_RANGE
            mutated.alpha = clamp(mutated_alpha, MIN_ALPHA, MAX_ALPHA)
        else:
            mutated.alpha = self.alpha

        return mutated

    def scale(self, scale_factor: float) -> 'StepTorch':
        """
        Scale the shape coordinates by a factor.

        Args:
            scale_factor: Factor to scale coordinates by

        Returns:
            Self (for chaining)
        """
        self.shape.scale(scale_factor)
        return self
