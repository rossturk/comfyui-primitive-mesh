"""Test TURBO optimizer with parallel rasterization."""
import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimizer
import optimizer_turbo
import shapes

def create_test_image(size=256):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            intensity = int(255 * (1 - min(dist / center, 1.0)))
            img[y, x, :] = intensity
    alpha = np.ones((size, size, 1), dtype=np.uint8) * 255
    return np.concatenate([img, alpha], axis=2)

target = create_test_image(256)
canvas = np.ones_like(target) * 255

cfg = {
    'width': 256, 'height': 256, 'steps': 30,
    'alpha': 0.5,
    'shapeTypes': [shapes.Triangle, shapes.Rectangle, shapes.Ellipse],
    'shapes': 200, 'mutations': 50,
    'computeSize': 256, 'mutateAlpha': False, 'blur': 0,
    'minlinewidth': 1, 'maxlinewidth': 2,
    'fill': (255, 255, 255), 'parallel': False
}

device = torch.device('cuda')

print("CPU Baseline:")
start = time.time()
cpu_opt = optimizer.Optimizer(target.copy(), canvas.copy(), cfg)
cpu_state = cpu_opt.start(lambda *args: None)
cpu_time = time.time() - start
print(f"  Time: {cpu_time:.2f}s\n")

print("Turbo GPU:")
start = time.time()
turbo_opt = optimizer_turbo.OptimizerTurbo(target.copy(), canvas.copy(), cfg, device)
turbo_state = turbo_opt.start(lambda *args: None)
turbo_time = time.time() - start
print(f"  Time: {turbo_time:.2f}s\n")

speedup = cpu_time / turbo_time
print(f"{'='*60}")
print(f"Speedup: {speedup:.2f}x")
if speedup >= 2.0:
    print("🚀 EXCELLENT! 2x+ speedup achieved!")
elif speedup >= 1.5:
    print("✓ GOOD! 1.5x+ speedup!")
elif speedup > 1.0:
    print("✓ Faster than CPU")
else:
    print("⚠ Needs more optimization")
print(f"{'='*60}")
