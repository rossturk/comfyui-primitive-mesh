"""
Step class representing a shape with color and alpha.
Ported from step.js to Python.
"""

import numpy as np
from typing import TYPE_CHECKING
from PIL import Image, ImageDraw
import copy

try:
    from . import util
    from .constants import MIN_ALPHA, MAX_ALPHA, ALPHA_MUTATION_RANGE
    from .common import clamp
except ImportError:
    import util
    from constants import MIN_ALPHA, MAX_ALPHA, ALPHA_MUTATION_RANGE
    from common import clamp

if TYPE_CHECKING:
    from .shapes import Shape
    from .state import State


class Step:
    """
    Represents a single optimization step: a shape with color and alpha.

    Attributes:
        shape: The geometric shape
        cfg: Configuration dictionary
        alpha: Transparency value (0-1)
        color: RGB color tuple
        distance: Distance metric after applying this step
    """

    def __init__(self, shape: 'Shape', cfg: dict):
        """
        Initialize step.

        Args:
            shape: Shape instance
            cfg: Configuration dictionary
        """
        import random

        self.shape = shape
        self.cfg = cfg

        # Randomize alpha within the configured range
        alpha_base = cfg.get('alpha', 0.5)
        alpha_range = cfg.get('alpha_range', 0.0)
        self.alpha = alpha_base + (random.random() - 0.5) * alpha_range
        self.alpha = clamp(self.alpha, MIN_ALPHA, MAX_ALPHA)

        # Computed during compute() call
        self.color = (0, 0, 0)
        self.distance = float('inf')

    def to_svg(self) -> str:
        """
        Generate SVG representation.

        Returns:
            SVG string
        """
        self.shape.color = self.color
        self.shape.alpha = self.alpha
        return self.shape.to_svg()

    def apply(self, state: 'State') -> 'State':
        """
        Apply this step to a state to create new state.

        Args:
            state: Current state

        Returns:
            New state with this step applied
        """
        try:
            from .state import State
        except ImportError:
            from state import State

        # Clone current canvas
        new_canvas = state.current.copy()

        # Draw this step onto it
        new_canvas = self._draw_step(new_canvas)

        # IMPORTANT: Always recalculate distance from actual canvas data
        # Don't trust the pre-computed distance as it may have rounding errors
        return State(state.target, new_canvas)

    def _draw_step(self, canvas: np.ndarray) -> np.ndarray:
        """
        Draw this step onto a canvas array.

        Args:
            canvas: Canvas array to draw on (H, W, C)

        Returns:
            Modified canvas array
        """
        from PIL import Image

        # Convert to PIL Image (ensure RGBA mode for alpha blending)
        if canvas.shape[2] == 3:
            # RGB - add alpha channel
            alpha_channel = np.ones((canvas.shape[0], canvas.shape[1], 1), dtype=np.uint8) * 255
            canvas_rgba = np.concatenate([canvas, alpha_channel], axis=2)
        else:
            canvas_rgba = canvas

        img = Image.fromarray(canvas_rgba.astype(np.uint8), mode='RGBA')

        # Create a transparent overlay for the shape
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Set color with alpha
        color_with_alpha = self.color + (int(self.alpha * 255),)
        self.shape.color = color_with_alpha

        # Render shape on overlay
        self.shape.render(draw)

        # Composite overlay onto image
        img = Image.alpha_composite(img, overlay)

        # Convert back to numpy (maintain RGBA if input was RGBA, RGB if RGB)
        result = np.array(img)
        if canvas.shape[2] == 3:
            return result[:, :, :3]  # Return RGB if input was RGB
        else:
            return result  # Return RGBA if input was RGBA

    def compute(self, state: 'State') -> 'Step':
        """
        Find optimal color and compute resulting distance.

        Args:
            state: Current optimization state

        Returns:
            Self (for chaining)
        """
        pixels = state.current.shape[0] * state.current.shape[1]
        offset = self.shape.bbox

        # Rasterize shape
        shape_array = self.shape.rasterize(self.alpha)

        # Prepare image data
        image_data = {
            'shape': shape_array,
            'current': state.current,
            'target': state.target
        }

        # Compute optimal color and difference change
        color, difference_change = util.compute_color_and_difference_change(offset, image_data, self.alpha)

        self.color = tuple(color)

        # Compute new distance
        current_difference = util.distance_to_difference(state.distance, pixels)
        new_difference = current_difference + difference_change
        self.distance = util.difference_to_distance(new_difference, pixels)

        return self

    def mutate(self) -> 'Step':
        """
        Create mutated version of this step.

        Returns:
            New mutated step
        """
        import random

        # Mutate shape
        new_shape = self.shape.mutate(self.cfg)

        # Create new step
        mutated = Step(new_shape, self.cfg)

        # Optionally mutate alpha
        if self.cfg.get('mutateAlpha', False):
            mutated_alpha = self.alpha + (random.random() - 0.5) * ALPHA_MUTATION_RANGE
            mutated.alpha = clamp(mutated_alpha, MIN_ALPHA, MAX_ALPHA)
        else:
            mutated.alpha = self.alpha

        return mutated

    def scale(self, scale_factor: float) -> 'Step':
        """
        Scale the shape coordinates by a factor.

        Args:
            scale_factor: Factor to scale coordinates by

        Returns:
            Self (for chaining)
        """
        self.shape.scale(scale_factor)
        return self
