"""Safety Operator package.

This package provides a simple, model‑agnostic safety mechanism for AI
systems. It includes a safety operator that projects outputs into a
safe region, dampens unsafe components, and applies an optional
regularization towards aligned behaviors. See the README.md for
installation and usage examples.
"""

from .config import SafetyConfig  # noqa: F401
from .projector import SafeProjector  # noqa: F401
from .regularizer import Regularizer  # noqa: F401
from .operator import SafetyOperator  # noqa: F401
from .wrapper import SafetyWrapper  # noqa: F401

__all__ = [
    'SafetyConfig',
    'SafeProjector',
    'Regularizer',
    'SafetyOperator',
    'SafetyWrapper',
]
