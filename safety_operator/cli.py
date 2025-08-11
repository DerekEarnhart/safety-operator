#!/usr/bin/env python3
"""
Command-line interface for the safety operator.

Provides a convenient way to apply safety filtering to model outputs
from the command line.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import SafetyOperator, SafetyConfig, SafeProjector, Regularizer, SafetyWrapper


def load_array_from_file(file_path: str) -> np.ndarray:
    """Load numpy array from various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return np.array(data)
    elif file_path.suffix == '.txt':
        return np.loadtxt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_array_to_file(array: np.ndarray, file_path: str) -> None:
    """Save numpy array to various file formats."""
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        np.save(file_path, array)
    elif file_path.suffix == '.json':
        with open(file_path, 'w') as f:
            json.dump(array.tolist(), f, indent=2)
    elif file_path.suffix == '.txt':
        np.savetxt(file_path, array)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def create_safety_operator(
    safe_mask: np.ndarray,
    alpha: float = 0.8,
    beta: float = 0.1,
    gamma: float = 0.05,
    l2_scale: float = 0.05,
) -> SafetyWrapper:
    """Create a safety operator with the given parameters."""
    proj = SafeProjector(safe_mask=safe_mask)
    reg = Regularizer(l2_scale=l2_scale)
    config = SafetyConfig(alpha=alpha, beta=beta, gamma=gamma)
    op = SafetyOperator(projector=proj, regularizer=reg, config=config)
    return SafetyWrapper(operator=op, projector=proj)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Apply safety filtering to AI model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Filter a single vector with default parameters
  safety-operator-demo --input "1.0,2.0,3.0" --safe-mask "1,1,0"
  
  # Filter from file with custom parameters
  safety-operator-demo --input data.npy --output filtered.npy \\
    --safe-mask "1,1,0" --alpha 0.5 --beta 0.2 --steps 5
  
  # Filter batch from JSON file
  safety-operator-demo --input batch.json --output filtered.json \\
    --safe-mask "1,0,1,0" --steps 3
        """,
    )
    
    # Input/output arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input array (comma-separated values or file path)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: print to stdout)"
    )
    
    # Safety mask (required)
    parser.add_argument(
        "--safe-mask", "-m",
        required=True,
        help="Safe mask as comma-separated values (1=safe, 0=unsafe)"
    )
    
    # Safety parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.8,
        help="Unsafe component damping factor (default: 0.8)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Regularization strength (default: 0.1)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.05,
        help="Safe component damping factor (default: 0.05)"
    )
    parser.add_argument(
        "--l2-scale",
        type=float,
        default=0.05,
        help="L2 regularization scale (default: 0.05)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of iteration steps (default: 3)"
    )
    
    # Output format
    parser.add_argument(
        "--format",
        choices=["json", "npy", "txt"],
        help="Output format (inferred from file extension if not specified)"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places for output (default: 6)"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse safe mask
        safe_mask = np.array([float(x.strip()) for x in args.safe_mask.split(",")])
        
        # Load input data
        if Path(args.input).exists():
            # Input is a file
            input_array = load_array_from_file(args.input)
        else:
            # Input is comma-separated values
            input_array = np.array([float(x.strip()) for x in args.input.split(",")])
        
        if args.verbose:
            print(f"Input shape: {input_array.shape}")
            print(f"Safe mask: {safe_mask}")
            print(f"Safety parameters: α={args.alpha}, β={args.beta}, γ={args.gamma}")
            print(f"Iteration steps: {args.steps}")
        
        # Validate dimensions
        if input_array.ndim == 1:
            if len(input_array) != len(safe_mask):
                raise ValueError(f"Input length ({len(input_array)}) must match safe mask length ({len(safe_mask)})")
        elif input_array.ndim == 2:
            if input_array.shape[1] != len(safe_mask):
                raise ValueError(f"Input width ({input_array.shape[1]}) must match safe mask length ({len(safe_mask)})")
        else:
            raise ValueError("Input must be 1D or 2D array")
        
        # Create safety operator
        wrapper = create_safety_operator(
            safe_mask=safe_mask,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            l2_scale=args.l2_scale,
        )
        
        # Apply filtering
        filtered_array = wrapper.filter_logits(input_array, steps=args.steps)
        
        if args.verbose:
            change = np.linalg.norm(filtered_array - input_array)
            print(f"Change magnitude: {change:.6f}")
        
        # Output result
        if args.output:
            # Save to file
            save_array_to_file(filtered_array, args.output)
            if args.verbose:
                print(f"Result saved to: {args.output}")
        else:
            # Print to stdout
            if args.format == "json":
                print(json.dumps(filtered_array.tolist(), indent=2))
            else:
                np.set_printoptions(precision=args.precision, suppress=True)
                print(filtered_array)
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
