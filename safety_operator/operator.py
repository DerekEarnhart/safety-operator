"""Core safety operator implementation."""

from __future__ import annotations

import numpy as np

from .config import SafetyConfig
from .projector import SafeProjector
from .regularizer import Regularizer


class SafetyOperator:
    """Safety operator for transforming model outputs.

    This operator maps an input vector into a safer version by separating
    safe and unsafe components, damping the unsafe components and optionally
    the safe components, and applying an alignment regularizer.

    Parameters
    ----------
    projector : SafeProjector
        Projector used to decompose vectors into safe and unsafe parts.
    regularizer : Regularizer
        Regularizer used to compute alignment contributions.
    config : SafetyConfig
        Configuration specifying damping and regularization parameters.
    """

    def __init__(self, *, projector: SafeProjector, regularizer: Regularizer, config: SafetyConfig):
        self.projector = projector
        self.regularizer = regularizer
        self.config = config

    def apply_once(self, x: np.ndarray) -> np.ndarray:
        """Apply the safety operator to `x` once.

        Parameters
        ----------
        x : array‑like
            Input vector.

        Returns
        -------
        ndarray
            Transformed vector after one application of the safety operator.
        """
        # Decompose into safe and unsafe parts
        safe_comp, unsafe_comp = self.projector.project(x)

        # Retrieve parameters
        alpha = self.config.alpha
        beta = self.config.beta
        gamma = self.config.gamma

        # Apply damping/regularization
        # Safe part: damp by (1 - gamma), add regularization
        safe_update = (1.0 - gamma) * safe_comp + beta * self.regularizer.apply(safe_comp)

        # Unsafe part: damp by alpha, add regularization
        unsafe_update = alpha * unsafe_comp + beta * self.regularizer.apply(unsafe_comp)

        return safe_update + unsafe_update

    def apply(self, x: np.ndarray, *, steps: int = 1) -> np.ndarray:
        """Iteratively apply the safety operator `steps` times."""
        result = np.asarray(x, dtype=float)
        # Guard against non‑positive iteration counts
        n = max(1, int(steps))
        for _ in range(n):
            result = self.apply_once(result)
        return result
