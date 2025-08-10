"""wrapper.py: Provides a high-level safety wrapper for applying the safety operator.

This module defines a SafetyWrapper class that coordinates the projection and 
regularization components to filter AI system outputs. It supports both 1D and
2D numpy arrays (e.g., logits vectors or batches of logits) and applies the
SafetyOperator iteratively to drive the state toward a safe region.
"""

import numpy as np
from typing import Optional, Union

from .operator import SafetyOperator
from .projector import SafeProjector


class SafetyWrapper:
    """A high-level wrapper for applying the safety operator to model outputs.

    Args:
        operator: An instance of :class:`SafetyOperator` containing the projection,
            regularization, and configuration parameters.
        projector: An instance of :class:`SafeProjector` used to project individual
            vectors onto the safe subspace. While the operator also holds a
            projector, this reference allows direct access when needed.

    This wrapper simplifies applying the safety transformation to arbitrary shapes
    by handling both single vectors and batches of vectors. It calls the
    underlying :meth:`SafetyOperator.apply` method iteratively.
    """

    def __init__(self, operator: SafetyOperator, projector: Optional[SafeProjector] = None) -> None:
        self.operator = operator
        # Default to the operator's projector if not provided separately
        self.projector = projector if projector is not None else operator.projector

    def filter_logits(self, logits: np.ndarray, steps: int = 1) -> np.ndarray:
        """Filter a vector or batch of logits through the safety operator.

        Args:
            logits: A 1D or 2D numpy array of raw model outputs. If 2D, the first
                dimension is treated as the batch dimension.
            steps: Number of iterations to apply the operator. More steps may
                increase convergence toward the safe region.

        Returns:
            A numpy array of the same shape as ``logits`` after applying the
            safety transformation.
        """
        x = np.asarray(logits)
        if x.ndim == 1:
            return self.operator.apply(x, steps)
        elif x.ndim == 2:
            # Apply safety operator to each row independently
            return np.stack([self.operator.apply(row, steps) for row in x], axis=0)
        else:
            raise ValueError(
                f"filter_logits only supports 1D or 2D arrays, got array with shape {x.shape}"
            )
