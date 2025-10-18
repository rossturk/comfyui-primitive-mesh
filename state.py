"""
State class representing the current optimization state.
Ported from state.js to Python.
"""

import numpy as np
from typing import TYPE_CHECKING
from . import util

if TYPE_CHECKING:
    from PIL import Image


class State:
    """
    Represents the state of optimization.

    Attributes:
        target: Target image as numpy array (H, W, C)
        current: Current approximation as numpy array (H, W, C)
        distance: Distance metric between target and current
    """

    def __init__(self, target: np.ndarray, current: np.ndarray, distance: float = float('inf')):
        """
        Initialize state.

        Args:
            target: Target image array
            current: Current approximation array
            distance: Pre-computed distance (if available)
        """
        self.target = target
        self.current = current

        if distance == float('inf'):
            # Compute distance
            pixels = target.shape[0] * target.shape[1]
            difference = util.difference(current, target)
            self.distance = util.difference_to_distance(difference, pixels)
        else:
            self.distance = distance
