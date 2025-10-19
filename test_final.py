"""
Final comprehensive benchmark: CPU vs GPU (Standard/Hybrid approach)
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimizer
import optimizer_torch
import shapes

def create_test_image(size=256):
    """Create gradient test image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            intensity = int(255 * (1 - min(dist / center, 1.0)))
            img[y, x, 0] = intensity
            img[y, x, 1] = int(255 * x / size)
            img[y, x, 2] = int(255 * y / size)
    alpha = np.ones((size, size, 1), dtype=np.uint8) * 255
    return np.concatenate([img, alpha], axis=2)

def test_mode(name, opt_class, target, canvas, cfg, device=None):
    """Test a specific optimizer mode."""
    print(f"\n{'-'*60}")
    print(f"{name}")
    print(f"{'-'*60}")

    start = time.time()
    if device:
        opt = opt_class(target, canvas, cfg, device)
    else:
        opt = opt_class(target, canvas, cfg)

    state = opt.start(lambda *args: None)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s")
    return elapsed, state.distance

def main():
    print("="*60)
    print("FINAL GPU ACCELERATION BENCHMARK")
    print("ComfyUI Primitive Mesh - Production Performance Test")
    print("="*60)

    if not torch.cuda.is_available():
        print("\n❌ CUDA not available - cannot run GPU tests")
        return

    print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ CUDA {torch.version.cuda} / PyTorch {torch.__version__}")

    device = torch.device('cuda')
    target = create_test_image(256)
    canvas = np.ones_like(target) * 255

    # Test configurations
    tests = [
        ("Quick (20 shapes)", 20),
        ("Medium (50 shapes)", 50),
        ("Large (100 shapes)", 100),
    ]

    all_results = []

    for test_name, num_shapes in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")

        cfg = {
            'width': 256, 'height': 256, 'steps': num_shapes,
            'alpha': 0.5,
            'shapeTypes': [shapes.Triangle, shapes.Rectangle, shapes.RotatedRectangle,
                          shapes.Ellipse, shapes.Quadrilateral],
            'shapes': 200, 'mutations': 50,
            'computeSize': 256, 'mutateAlpha': False, 'blur': 0,
            'minlinewidth': 1, 'maxlinewidth': 2,
            'fill': (255, 255, 255), 'parallel': False
        }

        # Test CPU
        cpu_time, cpu_dist = test_mode(
            "CPU Baseline",
            optimizer.Optimizer,
            target.copy(),
            canvas.copy(),
            cfg
        )

        # Test GPU
        gpu_time, gpu_dist = test_mode(
            "GPU Accelerated",
            optimizer_torch.OptimizerTorch,
            target.copy(),
            canvas.copy(),
            cfg,
            device
        )

        speedup = cpu_time / gpu_time
        all_results.append({
            'name': test_name,
            'shapes': num_shapes,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'cpu_dist': cpu_dist,
            'gpu_dist': gpu_dist
        })

        print(f"\n{test_name} Results:")
        print(f"  CPU: {cpu_time:.2f}s")
        print(f"  GPU: {gpu_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Quality (CPU): {cpu_dist:.6f}")
        print(f"  Quality (GPU): {gpu_dist:.6f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")

    print(f"{'Test':<25} {'Shapes':<8} {'CPU (s)':<10} {'GPU (s)':<10} {'Speedup':<10}")
    print("-"*60)
    for r in all_results:
        print(f"{r['name']:<25} {r['shapes']:<8} {r['cpu_time']:<10.2f} "
              f"{r['gpu_time']:<10.2f} {r['speedup']:<9.2f}x")

    avg_speedup = sum(r['speedup'] for r in all_results) / len(all_results)
    print("-"*60)
    print(f"{'AVERAGE':<25} {'':<8} {'':<10} {'':<10} {avg_speedup:<9.2f}x")

    print(f"\n{'='*60}")
    if avg_speedup >= 1.5:
        print(f"🚀 SUCCESS! Average {avg_speedup:.2f}x speedup achieved!")
    elif avg_speedup >= 1.2:
        print(f"✓ GOOD! Average {avg_speedup:.2f}x speedup")
    elif avg_speedup > 1.0:
        print(f"✓ Modest improvement: {avg_speedup:.2f}x faster")
    else:
        print(f"⚠ GPU slower than CPU ({avg_speedup:.2f}x)")

    print(f"{'='*60}")

    # Additional stats
    print(f"\nPerformance Improvement:")
    best = max(all_results, key=lambda x: x['speedup'])
    print(f"  Best case: {best['speedup']:.2f}x speedup ({best['name']})")
    worst = min(all_results, key=lambda x: x['speedup'])
    print(f"  Worst case: {worst['speedup']:.2f}x speedup ({worst['name']})")

    total_cpu = sum(r['cpu_time'] for r in all_results)
    total_gpu = sum(r['gpu_time'] for r in all_results)
    print(f"\nTime saved: {total_cpu - total_gpu:.2f}s ({(1 - total_gpu/total_cpu)*100:.1f}% faster)")

if __name__ == "__main__":
    main()
