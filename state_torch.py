"""
GPU-accelerated state management for optimization.
Keeps all image data on GPU to avoid CPU<->GPU transfers.
"""

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class StateTorch:
    """
    Optimization state with GPU-accelerated distance computation.

    Keeps target and current canvas as torch tensors on GPU.
    """

    def __init__(self, target: torch.Tensor, current: torch.Tensor, device: torch.device):
        """
        Initialize state.

        Args:
            target: Target image tensor (H, W, C) on GPU
            current: Current canvas tensor (H, W, C) on GPU
            device: torch device
        """
        self.device = device

        # Ensure tensors are on the correct device
        self.target = target.to(device) if not target.is_cuda else target
        self.current = current.to(device) if not current.is_cuda else current

        # Compute initial distance
        self.distance = self._compute_distance()

    def _compute_distance(self) -> float:
        """
        Compute RMS distance between current and target (GPU-accelerated).

        Returns:
            Distance metric
        """
        # Only compare RGB channels
        diff = self.current[:, :, :3].float() - self.target[:, :, :3].float()
        squared_diff = diff * diff
        sum_diff = torch.sum(squared_diff).item()

        pixels = self.current.shape[0] * self.current.shape[1]

        # RMS distance normalized
        if pixels == 0:
            return 0.0

        # sqrt(sum / (3 * pixels)) / 255
        distance = (sum_diff / (3 * pixels)) ** 0.5 / 255.0
        return distance

    def copy(self) -> 'StateTorch':
        """Create a copy of this state."""
        return StateTorch(
            self.target.clone(),
            self.current.clone(),
            self.device
        )

    def to_numpy(self) -> tuple:
        """
        Convert state tensors to numpy arrays (for compatibility).

        Returns:
            (target_numpy, current_numpy)
        """
        return (
            self.target.cpu().numpy(),
            self.current.cpu().numpy()
        )

    @staticmethod
    def from_numpy(
        target: np.ndarray,
        current: np.ndarray,
        device: torch.device
    ) -> 'StateTorch':
        """
        Create StateTorch from numpy arrays.

        Args:
            target: Target image numpy array
            current: Current canvas numpy array
            device: torch device

        Returns:
            StateTorch instance
        """
        target_tensor = torch.from_numpy(target).to(device)
        current_tensor = torch.from_numpy(current).to(device)

        return StateTorch(target_tensor, current_tensor, device)
