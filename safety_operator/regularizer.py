"""regularizer.py: Contains the Regularizer class for L2 regularization.

This module defines a simple L2 regularization operator that can be used to 
encourage AI system states to remain close to a reference state. The regularizer
operates element-wise on a vector and returns a damping term proportional to the 
difference between the current state and the reference.

The regularizer is optional: if `l2_scale` is zero or negative, it returns a zero 
vector of the same shape as the input.
"""

import numpy as np
from typing import Optional


class Regularizer:
    """A simple L2 regularizer.

    Args:
        l2_scale: A non-negative scalar controlling the strength of the regularization.
            A larger value increases the damping of deviations from the reference.
        reference: Optional reference state. If provided, regularization is applied toward
            this reference. If None, the zero vector of the same shape as the input is
            used as the reference.

    Raises:
        ValueError: If l2_scale is negative.
    """

    def __init__(self, l2_scale: float = 0.0, reference: Optional[np.ndarray] = None) -> None:
        if l2_scale < 0:
            raise ValueError("l2_scale must be non-negative")
        self.l2_scale = l2_scale
        self.reference = reference

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Compute the regularization term for a given state.

        Args:
            x: A numpy array representing the current system state.

        Returns:
            A numpy array representing the adjustment term. If the regularizer is disabled
            (l2_scale <= 0), this will be a zero array of the same shape as `x`.
        """
        if self.l2_scale <= 0:
            return np.zeros_like(x)

        ref = np.zeros_like(x) if self.reference is None else self.reference
        if ref.shape != x.shape:
            raise ValueError(
                f"Reference shape {ref.shape} does not match state shape {x.shape}."
            )

        # L2 regularization encourages the system state to move toward the reference.
        return -self.l2_scale * (x - ref)
