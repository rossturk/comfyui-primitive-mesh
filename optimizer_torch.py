"""
GPU-accelerated optimizer for shape optimization.
Keeps all image data on GPU and performs parallel candidate evaluation.
"""

import torch
import numpy as np
from typing import Callable, Optional
import time

try:
    from .shapes import Shape
    from .step_torch import StepTorch
    from .state_torch import StateTorch
except ImportError:
    from shapes import Shape
    from step_torch import StepTorch
    from state_torch import StateTorch


class OptimizerTorch:
    """
    GPU-accelerated optimization engine.

    Key improvements over CPU version:
    1. All image data stays on GPU (no CPU<->GPU transfers in inner loop)
    2. Color computation uses GPU tensor operations
    3. Distance calculations on GPU
    4. Potential for parallel candidate evaluation (future)
    """

    def __init__(
        self,
        target: np.ndarray,
        current: np.ndarray,
        cfg: dict,
        device: Optional[torch.device] = None
    ):
        """
        Initialize GPU-accelerated optimizer.

        Args:
            target: Target image array (H, W, C)
            current: Starting canvas array (H, W, C)
            cfg: Configuration dictionary
            device: torch device (auto-detected if None)
        """
        self.cfg = cfg
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Convert to torch tensors and move to GPU
        self.state = StateTorch.from_numpy(target, current, self.device)

        self._steps = 0
        self.on_step: Optional[Callable] = None

        print(f"OptimizerTorch initialized on device: {self.device}")

    def start(self, progress_callback: Optional[Callable] = None) -> StateTorch:
        """
        Start optimization process.

        Args:
            progress_callback: Optional callback function(step_num, total_steps, step, state)

        Returns:
            Final state
        """
        self.on_step = progress_callback
        start_time = time.time()

        total_steps = self.cfg.get('steps', 100)

        for i in range(total_steps):
            step = self._add_shape()

            if step and step.distance < self.state.distance:
                # Better than current state
                self.state = step.apply(self.state)

                if self.on_step:
                    self.on_step(i + 1, total_steps, step, self.state)
            else:
                # No improvement
                if self.on_step:
                    self.on_step(i + 1, total_steps, None, self.state)

        elapsed = time.time() - start_time
        print(f"GPU Optimization finished in {elapsed:.2f}s ({total_steps / elapsed:.2f} shapes/sec)")

        return self.state

    def _add_shape(self) -> Optional[StepTorch]:
        """
        Find and optimize a single shape.

        Returns:
            Best optimized step, or None if no improvement
        """
        # Find best initial shape
        best_step = self._find_best_step()

        if best_step is None:
            return None

        # Optimize it through mutations
        optimized_step = self._optimize_step(best_step)

        self._steps += 1
        return optimized_step

    def _find_best_step(self) -> Optional[StepTorch]:
        """
        Generate N random shapes and find the best one (GPU-accelerated).

        Returns:
            Best step from random candidates
        """
        num_shapes = self.cfg.get('shapes', 200)
        best_step = None

        # Future optimization: Batch evaluate candidates on GPU in parallel
        # For now, sequential evaluation with GPU-accelerated color computation
        for i in range(num_shapes):
            shape = Shape.create(self.cfg)
            step = StepTorch(shape, self.cfg, self.device)
            step.compute(self.state)  # GPU-accelerated

            if best_step is None or step.distance < best_step.distance:
                best_step = step

        return best_step

    def _optimize_step(self, step: StepTorch) -> StepTorch:
        """
        Optimize a step through mutations (GPU-accelerated).

        Args:
            step: Initial step to optimize

        Returns:
            Best mutated version
        """
        max_mutations = self.cfg.get('mutations', 50)
        failed_attempts = 0
        best_step = step

        while failed_attempts < max_mutations:
            # Try mutation
            mutated = best_step.mutate()
            mutated.compute(self.state)  # GPU-accelerated

            if mutated.distance < best_step.distance:
                # Success - accept mutation
                best_step = mutated
                failed_attempts = 0
            else:
                # Failure - increment counter
                failed_attempts += 1

        return best_step


# Future optimizations:
# 1. Batch candidate evaluation - evaluate multiple shapes in parallel on GPU
# 2. Custom CUDA kernels for shape rasterization
# 3. Mixed precision (FP16) for faster computation
# 4. Pre-allocate GPU memory for common operations
