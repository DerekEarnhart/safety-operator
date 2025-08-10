"""Configuration for the safety operator."""


class SafetyConfig:
    """Simple configuration for the safety operator.

    Parameters
    ----------
    alpha : float
        Damping factor for the unsafe component. Must satisfy 0 <= alpha < 1.
    beta : float
        Regularization strength. Non‑negative.
    gamma : float, optional
        Optional damping of the safe component. 0 <= gamma < 1. Defaults to 0.
    """

    def __init__(self, alpha=0.8, beta=0.0, gamma=0.0):
        if not (0 <= alpha < 1):
            raise ValueError('alpha must satisfy 0 <= alpha < 1')
        if beta < 0:
            raise ValueError('beta must be non‑negative')
        if not (0 <= gamma < 1):
            raise ValueError('gamma must satisfy 0 <= gamma < 1')
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def __repr__(self) -> str:
        return f"SafetyConfig(alpha={self.alpha}, beta={self.beta}, gamma={self.gamma})"
