#!/usr/bin/env python3
"""
Performance benchmarks for the safety operator.

This script benchmarks the safety operator against various scenarios
to demonstrate its efficiency and scalability.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from safety_operator import SafetyOperator, SafetyConfig, SafeProjector, Regularizer, SafetyWrapper


def benchmark_vector_sizes():
    """Benchmark performance across different vector sizes."""
    print("🔍 Benchmarking vector size performance...")
    
    sizes = [10, 50, 100, 500, 1000, 5000]
    times = []
    
    for size in sizes:
        # Create random safe mask
        safe_mask = np.random.choice([0, 1], size=size, p=[0.3, 0.7]).astype(float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.8, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        wrapper = SafetyWrapper(operator=op, projector=proj)
        
        # Random input
        logits = np.random.randn(size)
        
        # Time the operation
        start_time = time.time()
        for _ in range(100):  # Multiple iterations for better timing
            result = wrapper.filter_logits(logits, steps=3)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        times.append(avg_time * 1000)  # Convert to milliseconds
        
        print(f"  Size {size:4d}: {avg_time*1000:6.2f} ms per operation")
    
    return sizes, times


def benchmark_batch_sizes():
    """Benchmark performance across different batch sizes."""
    print("\n🔍 Benchmarking batch size performance...")
    
    vector_size = 100
    batch_sizes = [1, 10, 50, 100, 500, 1000]
    times = []
    
    # Create fixed components
    safe_mask = np.random.choice([0, 1], size=vector_size, p=[0.3, 0.7]).astype(float)
    proj = SafeProjector(safe_mask=safe_mask)
    reg = Regularizer(l2_scale=0.05)
    config = SafetyConfig(alpha=0.8, beta=0.1, gamma=0.05)
    
    op = SafetyOperator(projector=proj, regularizer=reg, config=config)
    wrapper = SafetyWrapper(operator=op, projector=proj)
    
    for batch_size in batch_sizes:
        # Random batch input
        batch_logits = np.random.randn(batch_size, vector_size)
        
        # Time the operation
        start_time = time.time()
        for _ in range(max(1, 100 // batch_size)):  # Adjust iterations based on batch size
            result = wrapper.filter_logits(batch_logits, steps=3)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / max(1, 100 // batch_size)
        times.append(avg_time * 1000)  # Convert to milliseconds
        
        print(f"  Batch {batch_size:4d}: {avg_time*1000:6.2f} ms per batch")
    
    return batch_sizes, times


def benchmark_iteration_steps():
    """Benchmark performance across different iteration steps."""
    print("\n🔍 Benchmarking iteration steps performance...")
    
    steps_list = [1, 2, 5, 10, 20, 50]
    times = []
    
    # Create fixed components
    size = 100
    safe_mask = np.random.choice([0, 1], size=size, p=[0.3, 0.7]).astype(float)
    proj = SafeProjector(safe_mask=safe_mask)
    reg = Regularizer(l2_scale=0.05)
    config = SafetyConfig(alpha=0.8, beta=0.1, gamma=0.05)
    
    op = SafetyOperator(projector=proj, regularizer=reg, config=config)
    wrapper = SafetyWrapper(operator=op, projector=proj)
    
    logits = np.random.randn(size)
    
    for steps in steps_list:
        # Time the operation
        start_time = time.time()
        for _ in range(100):
            result = wrapper.filter_logits(logits, steps=steps)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        times.append(avg_time * 1000)  # Convert to milliseconds
        
        print(f"  Steps {steps:2d}: {avg_time*1000:6.2f} ms per operation")
    
    return steps_list, times


def benchmark_convergence():
    """Benchmark convergence behavior."""
    print("\n🔍 Benchmarking convergence behavior...")
    
    # Create test setup
    size = 50
    safe_mask = np.random.choice([0, 1], size=size, p=[0.3, 0.7]).astype(float)
    proj = SafeProjector(safe_mask=safe_mask)
    reg = Regularizer(l2_scale=0.05)
    config = SafetyConfig(alpha=0.8, beta=0.1, gamma=0.05)
    
    op = SafetyOperator(projector=proj, regularizer=reg, config=config)
    wrapper = SafetyWrapper(operator=op, projector=proj)
    
    # Test input
    logits = np.random.randn(size)
    
    # Track convergence
    results = []
    changes = []
    
    prev_result = logits
    for step in range(1, 21):
        result = wrapper.filter_logits(logits, steps=step)
        results.append(result)
        
        change = np.linalg.norm(result - prev_result)
        changes.append(change)
        prev_result = result
        
        if step % 5 == 0:
            print(f"  Step {step:2d}: Change magnitude = {change:.6f}")
    
    return list(range(1, 21)), changes


def plot_benchmarks():
    """Create plots of benchmark results."""
    print("\n📊 Generating benchmark plots...")
    
    # Run all benchmarks
    sizes, size_times = benchmark_vector_sizes()
    batch_sizes, batch_times = benchmark_batch_sizes()
    steps_list, step_times = benchmark_iteration_steps()
    convergence_steps, convergence_changes = benchmark_convergence()
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vector size performance
    ax1.plot(sizes, size_times, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Vector Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Performance vs Vector Size')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Batch size performance
    ax2.plot(batch_sizes, batch_times, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Performance vs Batch Size')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Iteration steps performance
    ax3.plot(steps_list, step_times, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Iteration Steps')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Performance vs Iteration Steps')
    ax3.grid(True, alpha=0.3)
    
    # Convergence behavior
    ax4.plot(convergence_steps, convergence_changes, 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Iteration Steps')
    ax4.set_ylabel('Change Magnitude')
    ax4.set_title('Convergence Behavior')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300, bbox_inches='tight')
    print("  📈 Benchmark plots saved to 'benchmark_results.png'")
    
    return fig


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark suite."""
    print("🚀 Running comprehensive safety operator benchmarks...")
    print("=" * 60)
    
    # Run benchmarks
    sizes, size_times = benchmark_vector_sizes()
    batch_sizes, batch_times = benchmark_batch_sizes()
    steps_list, step_times = benchmark_iteration_steps()
    convergence_steps, convergence_changes = benchmark_convergence()
    
    # Summary statistics
    print("\n📋 Benchmark Summary:")
    print(f"  • Vector size range: {min(sizes)} - {max(sizes)}")
    print(f"  • Batch size range: {min(batch_sizes)} - {max(batch_sizes)}")
    print(f"  • Max vector processing time: {max(size_times):.2f} ms")
    print(f"  • Max batch processing time: {max(batch_times):.2f} ms")
    print(f"  • Convergence achieved by step: {np.argmin(convergence_changes[5:]) + 6}")
    
    # Performance insights
    print("\n💡 Performance Insights:")
    print(f"  • Linear scaling with vector size: {np.corrcoef(sizes, size_times)[0,1]:.3f}")
    print(f"  • Batch efficiency: {batch_times[0] / batch_times[-1] * len(batch_sizes):.1f}x speedup")
    print(f"  • Step efficiency: {step_times[0] / step_times[-1] * len(steps_list):.1f}x slowdown")
    
    return {
        'vector_sizes': sizes,
        'vector_times': size_times,
        'batch_sizes': batch_sizes,
        'batch_times': batch_times,
        'steps_list': steps_list,
        'step_times': step_times,
        'convergence_steps': convergence_steps,
        'convergence_changes': convergence_changes
    }


if __name__ == "__main__":
    try:
        results = run_comprehensive_benchmark()
        print("\n✅ All benchmarks completed successfully!")
        
        # Try to create plots if matplotlib is available
        try:
            plot_benchmarks()
            print("✅ Benchmark plots generated!")
        except ImportError:
            print("⚠️  matplotlib not available - skipping plots")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise
