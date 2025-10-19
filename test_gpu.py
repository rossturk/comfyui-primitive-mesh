"""
Test script to verify GPU acceleration is working and compare performance.
"""

import torch
import numpy as np
from PIL import Image
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import with absolute imports (no relative imports for standalone script)
import optimizer
import optimizer_torch
import shapes
import util_torch

# Aliases for convenience
Optimizer = optimizer.Optimizer
OptimizerTorch = optimizer_torch.OptimizerTorch
get_device = util_torch.get_device
Triangle = shapes.Triangle
Rectangle = shapes.Rectangle
RotatedRectangle = shapes.RotatedRectangle
Ellipse = shapes.Ellipse
Quadrilateral = shapes.Quadrilateral


def create_test_image(size=256):
    """Create a simple test image."""
    # Create gradient image
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            img[y, x, 0] = int(255 * x / size)  # Red gradient
            img[y, x, 1] = int(255 * y / size)  # Green gradient
            img[y, x, 2] = 128  # Constant blue

    # Add alpha channel
    alpha = np.ones((size, size, 1), dtype=np.uint8) * 255
    img_rgba = np.concatenate([img, alpha], axis=2)

    return img_rgba


def create_config(num_shapes=50):
    """Create test configuration."""
    return {
        'width': 256,
        'height': 256,
        'steps': num_shapes,
        'alpha': 0.5,
        'shapeTypes': [Triangle, Rectangle, RotatedRectangle, Ellipse, Quadrilateral],
        'shapes': 200,  # Candidate shapes per iteration
        'mutations': 50,
        'computeSize': 256,
        'mutateAlpha': False,
        'blur': 0,
        'minlinewidth': 1,
        'maxlinewidth': 2,
        'fill': (255, 255, 255),
        'parallel': False
    }


def test_cpu_optimizer(target, initial_canvas, cfg):
    """Test CPU optimizer."""
    print("\n" + "="*60)
    print("Testing CPU Optimizer")
    print("="*60)

    start_time = time.time()
    optimizer = Optimizer(target, initial_canvas, cfg)

    step_count = [0]

    def progress_cb(step_num, total, step, state):
        step_count[0] = step_num
        if step_num % 10 == 0:
            print(f"  CPU Progress: {step_num}/{total}")

    final_state = optimizer.start(progress_cb)
    elapsed = time.time() - start_time

    print(f"\nCPU Optimizer Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Shapes/sec: {step_count[0] / elapsed:.2f}")
    print(f"  Final distance: {final_state.distance:.6f}")

    return final_state, elapsed


def test_gpu_optimizer(target, initial_canvas, cfg, device):
    """Test GPU optimizer."""
    print("\n" + "="*60)
    print("Testing GPU Optimizer")
    print("="*60)
    print(f"Device: {device}")

    start_time = time.time()
    optimizer = OptimizerTorch(target, initial_canvas, cfg, device)

    step_count = [0]

    def progress_cb(step_num, total, step, state):
        step_count[0] = step_num
        if step_num % 10 == 0:
            print(f"  GPU Progress: {step_num}/{total}")

    final_state = optimizer.start(progress_cb)
    elapsed = time.time() - start_time

    print(f"\nGPU Optimizer Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Shapes/sec: {step_count[0] / elapsed:.2f}")
    print(f"  Final distance: {final_state.distance:.6f}")

    return final_state, elapsed


def main():
    """Run performance comparison."""
    print("Primitive Mesh GPU Acceleration Test")
    print("="*60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
    else:
        print("✗ CUDA is not available - will compare CPU vs CPU (no speedup expected)")

    device = get_device()
    print(f"  Selected device: {device}")

    # Create test data
    print("\nCreating test image...")
    target = create_test_image(256)
    initial_canvas = np.ones_like(target) * 255
    initial_canvas[:, :, 3] = 255

    # Test with different shape counts
    test_configs = [
        ("Quick test", 10),
        ("Medium test", 30),
    ]

    results = []

    for test_name, num_shapes in test_configs:
        print(f"\n{'='*60}")
        print(f"{test_name}: {num_shapes} shapes")
        print(f"{'='*60}")

        cfg = create_config(num_shapes)

        # Test CPU
        cpu_state, cpu_time = test_cpu_optimizer(target.copy(), initial_canvas.copy(), cfg)

        # Test GPU
        gpu_state, gpu_time = test_gpu_optimizer(target.copy(), initial_canvas.copy(), cfg, device)

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0

        results.append({
            'name': test_name,
            'shapes': num_shapes,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })

        print(f"\n{test_name} Comparison:")
        print(f"  CPU time: {cpu_time:.2f}s")
        print(f"  GPU time: {gpu_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['name']:20s}: {r['speedup']:.2f}x speedup ({r['cpu_time']:.2f}s → {r['gpu_time']:.2f}s)")

    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

    if avg_speedup > 1.5:
        print("\n✓ GPU acceleration is working well!")
    elif avg_speedup > 1.0:
        print("\n⚠ GPU acceleration is working but speedup is modest")
    else:
        print("\n✗ GPU acceleration may not be working properly")


if __name__ == "__main__":
    main()
