#!/usr/bin/env python3
"""
Standalone test script for Sketchbeast algorithm.
Tests the core functionality without ComfyUI.
"""

import numpy as np
from PIL import Image
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shapes import Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral
from optimizer import Optimizer
import random


def create_test_image(size=256):
    """Create a simple test image with colored circles."""
    img = Image.new('RGBA', (size, size), (255, 255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Red circle
    draw.ellipse([size//4 - size//7, size//2 - size//7,
                  size//4 + size//7, size//2 + size//7], fill=(255, 0, 0, 255))

    # Green circle
    draw.ellipse([size//2 - size//7, size//2 - size//7,
                  size//2 + size//7, size//2 + size//7], fill=(0, 255, 0, 255))

    # Blue circle
    draw.ellipse([3*size//4 - size//7, size//2 - size//7,
                  3*size//4 + size//7, size//2 + size//7], fill=(0, 0, 255, 255))

    return np.array(img)


def main():
    parser = argparse.ArgumentParser(description='Test Sketchbeast algorithm')
    parser.add_argument('--input', type=str, help='Input image path (optional, uses test image if not provided)')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--svg', type=str, default='output.svg', help='Output SVG path')
    parser.add_argument('--shapes', type=int, default=20, help='Number of shapes')
    parser.add_argument('--mode', type=str, default='mixed',
                       choices=['mixed', 'triangles', 'rectangles', 'ellipses', 'quadrilaterals'],
                       help='Shape type')
    parser.add_argument('--alpha', type=float, default=0.5, help='Shape alpha')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--size', type=int, default=256, help='Computation size')

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load or create image
    if args.input:
        print(f"Loading image: {args.input}")
        img = Image.open(args.input)
        target_array = np.array(img)
    else:
        print("Using test image (3 colored circles)")
        target_array = create_test_image(args.size)

    # Ensure RGBA
    if len(target_array.shape) == 2:
        # Grayscale
        target_array = np.stack([target_array] * 3, axis=-1)
    if target_array.shape[2] == 3:
        # Add alpha
        alpha_channel = np.ones((target_array.shape[0], target_array.shape[1], 1), dtype=np.uint8) * 255
        target_array = np.concatenate([target_array, alpha_channel], axis=2)

    height, width = target_array.shape[:2]

    # Create initial canvas (white background)
    initial_canvas = np.ones_like(target_array)
    initial_canvas[:, :, :3] = 255
    initial_canvas[:, :, 3] = 255

    # Configure shapes
    shape_map = {
        'mixed': [Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral],
        'triangles': [Triangle],
        'rectangles': [RotatedRectangle],
        'ellipses': [Ellipse],
        'quadrilaterals': [Quadrilateral]
    }

    cfg = {
        'width': width,
        'height': height,
        'steps': args.shapes,
        'alpha': args.alpha,
        'shapeTypes': shape_map[args.mode],
        'shapes': 200,  # candidates per step
        'mutations': 50,  # max mutations
        'computeSize': args.size,
        'mutateAlpha': False,
        'blur': 0,
        'minlinewidth': 1,
        'maxlinewidth': 2,
        'fill': (255, 255, 255),
        'parallel': False
    }

    print(f"\nConfiguration:")
    print(f"  Size: {width}x{height}")
    print(f"  Shapes: {args.shapes}")
    print(f"  Mode: {args.mode}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Seed: {args.seed}\n")

    # Run optimization
    optimizer = Optimizer(target_array, initial_canvas, cfg)

    svg_parts = []

    def progress_callback(step_num, total, step, state):
        if step:
            svg_parts.append(step.to_svg())
        if step_num % 5 == 0 or step_num == total:
            print(f"Progress: {step_num}/{total} shapes (distance: {state.distance:.6f})")

    print("Starting optimization...")
    final_state = optimizer.start(progress_callback)

    # Save raster output
    result_array = final_state.current[:, :, :3]  # Drop alpha
    result_img = Image.fromarray(result_array.astype(np.uint8))
    result_img.save(args.output)
    print(f"\nSaved raster output: {args.output}")

    # Save SVG output
    svg_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<defs>
  <clipPath id="clip">
    <rect x="0" y="0" width="{width}" height="{height}"/>
  </clipPath>
</defs>
<rect x="0" y="0" width="{width}" height="{height}" fill="rgb(255,255,255)"/>
'''
    svg_footer = '</svg>'
    svg_content = svg_header + '\n'.join(svg_parts) + svg_footer

    with open(args.svg, 'w') as f:
        f.write(svg_content)
    print(f"Saved SVG output: {args.svg}")

    print(f"\nFinal distance: {final_state.distance:.6f}")
    print("Done!")


if __name__ == '__main__':
    main()
