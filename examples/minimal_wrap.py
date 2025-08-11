"""
Minimal example for the safety operator.

This example demonstrates basic usage of the safety operator to filter
model logits and ensure they stay within a safe region.
"""

import numpy as np
from safety_operator import SafetyOperator, SafetyConfig, SafeProjector, Regularizer, SafetyWrapper


def main():
    # Define a safe mask - first 3 dimensions are safe, last 2 are unsafe
    safe_mask = np.array([1, 1, 1, 0, 0], dtype=float)
    
    # Create the projector
    proj = SafeProjector(safe_mask=safe_mask)
    
    # Create the safety operator with configuration
    op = SafetyOperator(
        projector=proj,
        regularizer=Regularizer(l2_scale=0.05),
        config=SafetyConfig(alpha=0.8, beta=0.5, gamma=0.1),
    )
    
    # Create the wrapper
    wrapper = SafetyWrapper(operator=op, projector=proj)
    
    # Example logits (some potentially unsafe values)
    logits = np.array([3.2, -1.0, 0.5, 8.0, -4.0])
    
    print("Original logits:", logits)
    print("Safe mask:", safe_mask)
    
    # Apply safety filtering
    filtered = wrapper.filter_logits(logits, steps=2)
    print("Filtered logits:", filtered)
    
    # Show the difference
    diff = filtered - logits
    print("Difference:", diff)
    print("Magnitude of change:", np.linalg.norm(diff))


if __name__ == "__main__":
    main()

