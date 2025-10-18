"""
Optimizer class implementing the greedy shape optimization algorithm.
Ported from optimizer.js to Python.
"""

import numpy as np
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time

from .shapes import Shape
from .step import Step
from .state import State


class Optimizer:
    """
    Core optimization engine that iteratively finds and applies shapes.

    The algorithm:
    1. Generate N random candidate shapes
    2. Compute optimal color for each in parallel
    3. Select best shape (lowest distance)
    4. Mutate shape M times to refine it
    5. Apply if it improves overall state
    6. Repeat until target number of shapes reached
    """

    def __init__(self, target: np.ndarray, current: np.ndarray, cfg: dict):
        """
        Initialize optimizer.

        Args:
            target: Target image array (H, W, C)
            current: Starting canvas array (H, W, C)
            cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.state = State(target, current)
        self._steps = 0
        self.on_step: Optional[Callable] = None

    def start(self, progress_callback: Optional[Callable] = None) -> State:
        """
        Start optimization process.

        Args:
            progress_callback: Optional callback function(step_num, total_steps, step, state)
                - step_num: current step number (1-indexed)
                - total_steps: total number of steps
                - step: Step object (or None if no improvement)
                - state: current State object with updated canvas

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
        print(f"Optimization finished in {elapsed:.2f}s")

        return self.state

    def _add_shape(self) -> Optional[Step]:
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

    def _find_best_step(self) -> Optional[Step]:
        """
        Generate N random shapes and find the best one.

        Returns:
            Best step from random candidates
        """
        num_shapes = self.cfg.get('shapes', 200)
        best_step = None

        # Option to use multiprocessing
        use_parallel = self.cfg.get('parallel', False)

        if use_parallel:
            best_step = self._find_best_step_parallel(num_shapes)
        else:
            best_step = self._find_best_step_sequential(num_shapes)

        return best_step

    def _find_best_step_sequential(self, num_shapes: int) -> Optional[Step]:
        """Find best step sequentially (simpler, no multiprocessing overhead)."""
        best_step = None

        for i in range(num_shapes):
            shape = Shape.create(self.cfg)
            step = Step(shape, self.cfg)
            step.compute(self.state)

            if best_step is None or step.distance < best_step.distance:
                best_step = step

        return best_step

    def _find_best_step_parallel(self, num_shapes: int) -> Optional[Step]:
        """Find best step using parallel evaluation (for large workloads)."""
        # Note: Due to Python's GIL, ThreadPoolExecutor might be better for I/O
        # ProcessPoolExecutor is better for CPU-bound tasks but has serialization overhead

        best_step = None
        batch_size = 10  # Process in batches to reduce overhead

        for batch_start in range(0, num_shapes, batch_size):
            batch_end = min(batch_start + batch_size, num_shapes)
            batch_count = batch_end - batch_start

            # Process batch sequentially (multiprocessing overhead not worth it for this)
            for i in range(batch_count):
                shape = Shape.create(self.cfg)
                step = Step(shape, self.cfg)
                step.compute(self.state)

                if best_step is None or step.distance < best_step.distance:
                    best_step = step

        return best_step

    def _optimize_step(self, step: Step) -> Step:
        """
        Optimize a step through mutations.

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
            mutated.compute(self.state)

            if mutated.distance < best_step.distance:
                # Success - accept mutation
                best_step = mutated
                failed_attempts = 0
            else:
                # Failure - increment counter
                failed_attempts += 1

        return best_step
