"""
ComfyUI node implementation of a primitive mesh.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import random
import io
import base64

from .shapes import Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral
from .optimizer import Optimizer

# ComfyUI imports for preview support
try:
    import comfy.utils
    from server import PromptServer
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    PromptServer = None


class PrimitiveMeshNode:
    """
    ComfyUI node for generating vector art from images using geometric shapes.

    This node converts input images into artistic collages composed of geometric
    shapes (triangles, rectangles, ellipses, etc.) using a greedy optimization
    algorithm.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input parameters for the node."""
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI image tensor (B, H, W, C) in 0-1 range
                "num_shapes": ("INT", {
                    "default": 50,
                    "min": 7,
                    "max": 500,
                    "step": 1,
                    "display": "number"
                }),
                "shape_type": ([
                    "mixed",
                    "rectangles",
                    "triangles",
                    "quadrilaterals",
                    "ellipses"
                ], {
                    "default": "mixed"
                }),
                "alpha": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "display": "number"
                }),
                "compute_size": ("INT", {
                    "default": 256,
                    "min": 128,
                    "max": 1024,
                    "step": 64,
                    "display": "number"
                }),
                "candidate_shapes": ("INT", {
                    "default": 200,
                    "min": 50,
                    "max": 500,
                    "step": 10,
                    "display": "number"
                }),
                "mutations": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 200,
                    "step": 10,
                    "display": "number"
                }),
            },
            "optional": {
                "fill_color": ("STRING", {
                    "default": "auto",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "svg")
    FUNCTION = "generate"
    CATEGORY = "image/artistic"
    OUTPUT_NODE = False

    def generate(
        self,
        image: torch.Tensor,
        num_shapes: int,
        shape_type: str,
        alpha: float,
        seed: int,
        compute_size: int,
        candidate_shapes: int,
        mutations: int,
        fill_color: str = "auto"
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate vector art from input image.

        Args:
            image: Input image tensor (B, H, W, C) in range 0-1
            num_shapes: Number of shapes to generate
            shape_type: Type of shapes to use
            alpha: Shape transparency (0-1)
            seed: Random seed for reproducibility
            compute_size: Maximum dimension for computation
            candidate_shapes: Number of candidate shapes per iteration
            mutations: Maximum mutation attempts per shape
            fill_color: Background fill color (hex or 'auto')

        Returns:
            Tuple of (output image tensor, SVG string)
        """
        # Set random seed (clamp to valid range for NumPy)
        # NumPy requires seed to be between 0 and 2**32 - 1
        seed_clamped = seed % (2**32)
        random.seed(seed)
        np.random.seed(seed_clamped)

        # Convert ComfyUI tensor to numpy (take first image if batch)
        img_np = image[0].cpu().numpy()  # (H, W, C)
        img_np = (img_np * 255).astype(np.uint8)  # Convert to 0-255 range

        # Get original dimensions
        original_height, original_width = img_np.shape[:2]

        # Scale down for computation
        scale = max(original_width / compute_size, original_height / compute_size, 1.0)
        compute_width = int(original_width / scale)
        compute_height = int(original_height / scale)

        # Resize for computation
        img_pil = Image.fromarray(img_np)
        img_resized = img_pil.resize((compute_width, compute_height), Image.Resampling.LANCZOS)
        target_array = np.array(img_resized)

        # Ensure RGBA
        if target_array.shape[2] == 3:
            # Add alpha channel
            alpha_channel = np.ones((target_array.shape[0], target_array.shape[1], 1), dtype=np.uint8) * 255
            target_array = np.concatenate([target_array, alpha_channel], axis=2)

        # Determine fill color
        if fill_color == "auto":
            fill_rgb = self._compute_fill_color(target_array)
        else:
            fill_rgb = self._parse_color(fill_color)

        # Create initial canvas
        initial_canvas = np.ones_like(target_array) * 0
        initial_canvas[:, :, :3] = fill_rgb
        initial_canvas[:, :, 3] = 255

        # Configure shape types
        shape_types = self._get_shape_types(shape_type)

        # Build configuration
        cfg = {
            'width': compute_width,
            'height': compute_height,
            'steps': num_shapes,
            'alpha': alpha,
            'shapeTypes': shape_types,
            'shapes': candidate_shapes,
            'mutations': mutations,
            'computeSize': compute_size,
            'mutateAlpha': False,
            'blur': 0,  # No blur for now
            'minlinewidth': 1,
            'maxlinewidth': 2,
            'fill': fill_rgb,
            'parallel': False  # Sequential is more reliable for now
        }

        # Run optimization
        print(f"Starting primitive mesh optimization: {num_shapes} shapes, {shape_type} mode")

        optimizer = Optimizer(target_array, initial_canvas, cfg)

        svg_parts = []
        step_count = [0]  # Use list for closure

        # Initialize ComfyUI progress bar if available
        pbar = None
        if COMFY_AVAILABLE:
            pbar = comfy.utils.ProgressBar(num_shapes)

        def progress_callback(step_num, total, step, state):
            """Callback for progress updates with preview support."""
            step_count[0] = step_num
            if step:
                svg_parts.append(step.to_svg())

            # Update progress bar
            if COMFY_AVAILABLE and pbar:
                pbar.update_absolute(step_num, total)

            # Send preview every N steps via PromptServer WebSocket
            preview_interval = max(1, num_shapes // 20)  # Show ~20 previews total
            if COMFY_AVAILABLE and PromptServer and (step_num % preview_interval == 0 or step_num == total):
                # Get current canvas and convert to preview format
                preview_array = state.current[:, :, :3]  # RGB only

                # Resize to original dimensions for preview
                preview_pil = Image.fromarray(preview_array.astype(np.uint8))
                preview_pil = preview_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)

                # Convert PIL image to base64 for sending via WebSocket
                buffered = io.BytesIO()
                preview_pil.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Send preview via PromptServer WebSocket
                PromptServer.instance.send_sync("primitivemesh.preview", {
                    "image": img_str,
                    "step": step_num,
                    "total": total
                })

            # Print progress periodically
            if step_num % 10 == 0:
                print(f"  Progress: {step_num}/{total} shapes")

        final_state = optimizer.start(progress_callback)

        # Convert result back to ComfyUI tensor
        result_array = final_state.current[:, :, :3]  # Drop alpha channel

        # Resize back to original dimensions
        result_pil = Image.fromarray(result_array.astype(np.uint8))
        result_pil = result_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
        result_np = np.array(result_pil).astype(np.float32) / 255.0

        # Convert to tensor
        result_tensor = torch.from_numpy(result_np).unsqueeze(0)  # (1, H, W, C)

        # Generate SVG
        svg = self._generate_svg(svg_parts, original_width, original_height, fill_rgb)

        print(f"Primitive mesh complete: generated {step_count[0]} shapes")

        return (result_tensor, svg)

    def _get_shape_types(self, shape_type: str) -> list:
        """Get shape class list based on type selection."""
        if shape_type == "mixed":
            return [Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral]
        elif shape_type == "rectangles":
            return [RotatedRectangle]
        elif shape_type == "triangles":
            return [Triangle]
        elif shape_type == "quadrilaterals":
            return [Quadrilateral]
        elif shape_type == "ellipses":
            return [Ellipse]
        else:
            return [Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral]

    def _compute_fill_color(self, image_array: np.ndarray) -> Tuple[int, int, int]:
        """Compute fill color from image border pixels."""
        h, w = image_array.shape[:2]
        border_pixels = []

        # Sample border pixels
        for x in range(w):
            for y in range(h):
                if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                    border_pixels.append(image_array[y, x, :3])

        if not border_pixels:
            return (255, 255, 255)

        # Average
        border_array = np.array(border_pixels)
        avg_color = np.mean(border_array, axis=0).astype(np.uint8)

        return tuple(avg_color)

    def _parse_color(self, color_str: str) -> Tuple[int, int, int]:
        """Parse color string (hex or rgb)."""
        if color_str.startswith('#'):
            # Hex color
            color_str = color_str.lstrip('#')
            return tuple(int(color_str[i:i+2], 16) for i in (0, 2, 4))
        else:
            # Default white
            return (255, 255, 255)

    def _generate_svg(self, svg_parts: list, width: int, height: int, fill_color: Tuple[int, int, int]) -> str:
        """Generate complete SVG document."""
        svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<defs>
  <clipPath id="clip" clipPathUnits="objectBoundingBox">
    <rect x="0" y="0" width="{width}" height="{height}"/>
  </clipPath>
  <filter id="g0.6">
    <feGaussianBlur stdDeviation="0.6"/>
  </filter>
  <filter id="g1">
    <feGaussianBlur stdDeviation="1"/>
  </filter>
  <filter id="g10">
    <feGaussianBlur stdDeviation="10"/>
  </filter>
</defs>
<rect x="0" y="0" width="{width}" height="{height}" fill="rgb({fill_color[0]},{fill_color[1]},{fill_color[2]})"/>
'''

        svg_footer = '</svg>'

        svg_body = '\n'.join(svg_parts)

        return svg_header + svg_body + svg_footer
