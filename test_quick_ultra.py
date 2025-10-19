"""Quick test of ultra optimizer."""
import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import optimizer
import optimizer_ultra
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

def test(name, opt_class, target, canvas, cfg, device=None):
    print(f"\n{name}:")
    start = time.time()
    if device:
        opt = opt_class(target, canvas, cfg, device)
    else:
        opt = opt_class(target, canvas, cfg)
    state = opt.start(lambda *args: None)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.2f}s")
    return elapsed

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

cpu_time = test("CPU", optimizer.Optimizer, target.copy(), canvas.copy(), cfg)
ultra_time = test("Ultra GPU", optimizer_ultra.OptimizerUltra, target.copy(), canvas.copy(), cfg, device)

print(f"\nSpeedup: {cpu_time / ultra_time:.2f}x")
