"""
Advanced benchmarking script for ultra-optimized GPU acceleration.
Tests CPU vs Standard GPU vs Ultra GPU modes.
"""

import torch
import numpy as np
from PIL import Image
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import optimizers
import optimizer
import optimizer_torch
import optimizer_ultra
import shapes
import util_torch

# Aliases
Optimizer = optimizer.Optimizer
OptimizerTorch = optimizer_torch.OptimizerTorch
OptimizerUltra = optimizer_ultra.OptimizerUltra
get_device = util_torch.get_device
Triangle = shapes.Triangle
Rectangle = shapes.Rectangle
RotatedRectangle = shapes.RotatedRectangle
Ellipse = shapes.Ellipse
Quadrilateral = shapes.Quadrilateral


def create_test_image(size=256):
    """Create a test image with interesting features."""
    # Create gradient image with shapes
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Radial gradient
    center = size // 2
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            intensity = int(255 * (1 - min(dist / center, 1.0)))
            img[y, x, 0] = intensity  # Red
            img[y, x, 1] = int(255 * x / size)  # Green
            img[y, x, 2] = int(255 * y / size)  # Blue

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


def test_optimizer(name, optimizer_class, target, initial_canvas, cfg, device=None, **kwargs):
    """Test a specific optimizer."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")

    start_time = time.time()

    if device:
        opt = optimizer_class(target, initial_canvas, cfg, device, **kwargs)
    else:
        opt = optimizer_class(target, initial_canvas, cfg)

    step_count = [0]

    def progress_cb(step_num, total, step, state):
        step_count[0] = step_num
        if step_num % 10 == 0:
            print(f"  Progress: {step_num}/{total}")

    final_state = opt.start(progress_cb)
    elapsed = time.time() - start_time

    # Get distance
    if hasattr(final_state, 'distance'):
        distance = final_state.distance
    else:
        distance = 0.0

    print(f"\n{name} Results:")
    print(f"  Total time:   {elapsed:.2f}s")
    print(f"  Shapes/sec:   {step_count[0] / elapsed:.2f}")
    print(f"  Final distance: {distance:.6f}")

    return {
        'name': name,
        'time': elapsed,
        'shapes_per_sec': step_count[0] / elapsed,
        'distance': distance
    }


def main():
    """Run comprehensive performance comparison."""
    print("="*60)
    print("ULTRA GPU ACCELERATION BENCHMARK")
    print("="*60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")

        # Memory info
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU Memory: {mem_total:.1f} GB")
    else:
        print("✗ CUDA is not available")
        return

    device = get_device()
    print(f"  Selected device: {device}")

    # Create test data
    print("\nCreating test image...")
    target = create_test_image(256)
    initial_canvas = np.ones_like(target) * 255
    initial_canvas[:, :, 3] = 255

    # Test configurations
    test_configs = [
        ("Quick test", 20),
        ("Medium test", 50),
        ("Large test", 100),
    ]

    all_results = []

    for test_name, num_shapes in test_configs:
        print(f"\n{'#'*60}")
        print(f"# {test_name}: {num_shapes} shapes")
        print(f"{'#'*60}")

        cfg = create_config(num_shapes)

        results = []

        # Test CPU
        result = test_optimizer(
            "CPU Baseline",
            Optimizer,
            target.copy(),
            initial_canvas.copy(),
            cfg
        )
        results.append(result)
        cpu_time = result['time']

        # Test Standard GPU
        result = test_optimizer(
            "Standard GPU",
            OptimizerTorch,
            target.copy(),
            initial_canvas.copy(),
            cfg,
            device=device
        )
        results.append(result)
        standard_speedup = cpu_time / result['time']

        # Test Ultra GPU
        result = test_optimizer(
            "Ultra GPU",
            OptimizerUltra,
            target.copy(),
            initial_canvas.copy(),
            cfg,
            device=device
        )
        results.append(result)
        ultra_speedup = cpu_time / result['time']

        # Summary for this test
        print(f"\n{test_name} Comparison:")
        print(f"  CPU time:      {cpu_time:.2f}s (baseline)")
        print(f"  Standard GPU:  {results[1]['time']:.2f}s ({standard_speedup:.2f}x speedup)")
        print(f"  Ultra GPU:     {results[2]['time']:.2f}s ({ultra_speedup:.2f}x speedup)")
        print(f"  Ultra vs Standard: {results[1]['time'] / results[2]['time']:.2f}x faster")

        all_results.append({
            'test_name': test_name,
            'num_shapes': num_shapes,
            'cpu_time': cpu_time,
            'standard_time': results[1]['time'],
            'ultra_time': results[2]['time'],
            'standard_speedup': standard_speedup,
            'ultra_speedup': ultra_speedup
        })

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Test':<20} {'Shapes':<8} {'CPU':>8} {'Standard':>10} {'Ultra':>10}")
    print(f"{'':<20} {'':<8} {'(sec)':>8} {'speedup':>10} {'speedup':>10}")
    print("-" * 60)

    for r in all_results:
        print(f"{r['test_name']:<20} {r['num_shapes']:<8} "
              f"{r['cpu_time']:>8.2f} {r['standard_speedup']:>9.2f}x {r['ultra_speedup']:>9.2f}x")

    # Calculate averages
    avg_standard = sum(r['standard_speedup'] for r in all_results) / len(all_results)
    avg_ultra = sum(r['ultra_speedup'] for r in all_results) / len(all_results)

    print("-" * 60)
    print(f"{'AVERAGE':<20} {'':<8} {'':<8} {avg_standard:>9.2f}x {avg_ultra:>9.2f}x")

    print(f"\n{'='*60}")
    if avg_ultra >= 10.0:
        print("🚀 OUTSTANDING! Ultra mode achieves 10x+ speedup!")
    elif avg_ultra >= 5.0:
        print("✓ EXCELLENT! Ultra mode achieves 5x+ speedup!")
    elif avg_ultra >= 2.0:
        print("✓ GOOD! Ultra mode achieves 2x+ speedup!")
    else:
        print("⚠ Ultra mode needs further optimization")

    print(f"{'='*60}")

    # Efficiency rating
    print(f"\nGPU Efficiency:")
    print(f"  Standard mode: {avg_standard:.2f}x faster than CPU")
    print(f"  Ultra mode:    {avg_ultra:.2f}x faster than CPU")
    print(f"  Ultra vs Standard: {avg_ultra / avg_standard:.2f}x improvement")


if __name__ == "__main__":
    main()
