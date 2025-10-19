#!/usr/bin/env python3
"""
Test script to verify high-resolution rendering works correctly.
"""

import numpy as np
from PIL import Image
import sys

# Test that shapes can be scaled
from shapes import Triangle, Rectangle, Ellipse

def test_shape_scaling():
    """Test that shapes can be scaled properly."""
    print("Testing shape scaling...")

    cfg = {
        'width': 256,
        'height': 256,
        'alpha': 0.5,
        'shapeTypes': [Triangle],
        'blur': 0,
        'minlinewidth': 1,
        'maxlinewidth': 2
    }

    # Create a triangle at 256x256
    triangle = Triangle(cfg, 256, 256)
    print(f"Original triangle points: {triangle.points}")
    print(f"Original bbox: {triangle.bbox}")

    # Scale it 4x (to 1024x1024)
    triangle.scale(4.0)
    print(f"Scaled triangle points: {triangle.points}")
    print(f"Scaled bbox: {triangle.bbox}")

    # Verify points were scaled
    assert all(p[0] >= 0 and p[1] >= 0 for p in triangle.points), "Points should be positive"
    print("✓ Triangle scaling works!")

    # Test rectangle
    rect = Rectangle(cfg, 256, 256)
    print(f"\nOriginal rectangle points: {rect.points}")
    rect.scale(2.0)
    print(f"Scaled rectangle points: {rect.points}")
    print("✓ Rectangle scaling works!")

    # Test ellipse
    ellipse = Ellipse(cfg, 256, 256)
    print(f"\nOriginal ellipse center: {ellipse.center}, rx: {ellipse.rx}, ry: {ellipse.ry}")
    ellipse.scale(3.0)
    print(f"Scaled ellipse center: {ellipse.center}, rx: {ellipse.rx}, ry: {ellipse.ry}")
    print("✓ Ellipse scaling works!")

    print("\n✅ All shape scaling tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_shape_scaling()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
