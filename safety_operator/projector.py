"""Projection utilities for separating safe and unsafe components."""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple


class SafeProjector:
    """Projector that separates safe and unsafe components of a vector.

    The projector can operate in two modes:

    1. **Mask‑based**: Provide a `safe_mask` with values in [0, 1]. A value
       of 1 indicates a fully safe dimension; 0 indicates fully unsafe.
       Intermediate values partially weight the component.
    2. **Classifier‑based**: Provide a callable `classifier` that, given
       an input vector, returns an array of weights in [0, 1] of the same
       shape. Higher weights indicate safer components.

    Parameters
    ----------
    safe_mask : array‑like of float, optional
        Mask indicating safe components. Values outside [0, 1] will be
        clipped.
    classifier : Callable[[np.ndarray], np.ndarray], optional
        Function that returns safety weights for a given input vector.
        Should output values in [0, 1] of the same shape as the input.
    """

    def __init__(self, *, safe_mask=None, classifier: Callable[[np.ndarray], np.ndarray] | None = None):
        if safe_mask is None and classifier is None:
            # If nothing provided, default to identity projection (everything safe)
            self.safe_mask = None
            self.classifier = None
        else:
            self.safe_mask = None if safe_mask is None else np.asarray(safe_mask, dtype=float)
            self.classifier = classifier

    def _get_weights(self, x: np.ndarray) -> np.ndarray:
        """Compute safety weights for the given vector `x`."""
        if self.safe_mask is not None:
            # Broadcast mask to match the shape of x
            try:
                weights = np.broadcast_to(self.safe_mask, x.shape).astype(float)
            except Exception as exc:  # pragma: no cover
                raise ValueError(f"Cannot broadcast safe_mask to input shape {x.shape}: {exc}")
        elif self.classifier is not None:
            weights = self.classifier(x)
            if weights.shape != x.shape:
                raise ValueError('Classifier must return an array of the same shape as its input')
        else:
            # Identity: everything safe
            weights = np.ones_like(x, dtype=float)

        # Clip to [0, 1]
        return np.clip(weights, 0.0, 1.0)

    def project(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the safe and unsafe components of `x`.

        Parameters
        ----------
        x : array‑like
            Input vector (can be multi‑dimensional).

        Returns
        -------
        (safe, unsafe) : Tuple[np.ndarray, np.ndarray]
            The safe component (weighted by safety weights) and the unsafe
            component (complement of safety weights) of the input.
        """
        x_arr = np.asarray(x, dtype=float)
        weights = self._get_weights(x_arr)
        safe_component = weights * x_arr
        unsafe_component = (1.0 - weights) * x_arr
        return safe_component, unsafe_component
