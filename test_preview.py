#!/usr/bin/env python3
"""
Test script to verify the preview callback functionality.
This simulates what happens in ComfyUI.
"""

import numpy as np
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shapes import Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral
from optimizer import Optimizer
import random


def create_test_image(size=256):
    """Create a simple test image."""
    img = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Draw a simple gradient circle
    draw.ellipse([size//4, size//4, 3*size//4, 3*size//4], fill=(100, 150, 200, 255))

    return np.array(img)


def main():
    print("Testing Sketchbeast preview callback...")

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Create test image
    target_array = create_test_image(256)

    # Create initial canvas (white background)
    initial_canvas = np.ones_like(target_array)
    initial_canvas[:, :, :3] = [255, 255, 255]
    initial_canvas[:, :, 3] = 255

    # Configuration
    cfg = {
        'width': 256,
        'height': 256,
        'steps': 10,  # Just 10 shapes for quick test
        'alpha': 0.5,
        'shapeTypes': [Triangle, Rectangle, Ellipse],
        'shapes': 50,  # Fewer candidates for speed
        'mutations': 20,
        'computeSize': 256,
        'mutateAlpha': False,
        'blur': 0,
        'minlinewidth': 1,
        'maxlinewidth': 2,
        'fill': (255, 255, 255),
        'parallel': False
    }

    # Create optimizer
    optimizer = Optimizer(target_array, initial_canvas, cfg)

    preview_count = [0]

    def progress_callback(step_num, total, step, state):
        """Test callback that verifies we receive the state."""
        preview_count[0] += 1

        # Verify we got the state
        assert state is not None, "State should not be None"
        assert hasattr(state, 'current'), "State should have 'current' attribute"
        assert state.current.shape == target_array.shape, "State canvas should match target shape"

        print(f"  Step {step_num}/{total}: distance={state.distance:.4f}, shape={'added' if step else 'skipped'}")

        # Simulate preview generation (like in the node)
        if step_num % 3 == 0 or step_num == total:
            preview_array = state.current[:, :, :3]
            print(f"    -> Preview generated: shape={preview_array.shape}, dtype={preview_array.dtype}")

    # Run optimization
    print(f"\nStarting optimization with {cfg['steps']} shapes...")
    final_state = optimizer.start(progress_callback)

    print(f"\n✓ Optimization complete!")
    print(f"  Callbacks received: {preview_count[0]}")
    print(f"  Final distance: {final_state.distance:.4f}")
    print(f"  Final canvas shape: {final_state.current.shape}")

    # Save result
    output_path = '/tmp/sketchbeast_test_preview.png'
    result_img = Image.fromarray(final_state.current[:, :, :3].astype(np.uint8))
    result_img.save(output_path)
    print(f"  Saved result to: {output_path}")

    print("\n✓ Preview callback test passed!")


if __name__ == '__main__':
    main()
