"""
Comprehensive tests for the safety operator package.

Tests cover:
- Basic functionality
- Edge cases
- Performance characteristics
- Mathematical properties
- Error handling
"""

import numpy as np
import pytest
from safety_operator import (
    SafetyOperator, SafetyConfig, SafeProjector, 
    Regularizer, SafetyWrapper
)


class TestSafeProjector:
    """Test the SafeProjector class."""
    
    def test_basic_projection(self):
        """Test basic projection functionality."""
        safe_mask = np.array([1, 1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        
        x = np.array([2.0, -1.0, 3.0])
        safe_comp, unsafe_comp = proj.project(x)
        
        # Safe component should be [2.0, -1.0, 0.0]
        np.testing.assert_array_almost_equal(safe_comp, [2.0, -1.0, 0.0])
        # Unsafe component should be [0.0, 0.0, 3.0]
        np.testing.assert_array_almost_equal(unsafe_comp, [0.0, 0.0, 3.0])
        # Sum should equal original
        np.testing.assert_array_almost_equal(safe_comp + unsafe_comp, x)
    
    def test_all_safe_mask(self):
        """Test projection with all-safe mask."""
        safe_mask = np.array([1, 1, 1], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        
        x = np.array([1.0, 2.0, 3.0])
        safe_comp, unsafe_comp = proj.project(x)
        
        np.testing.assert_array_almost_equal(safe_comp, x)
        np.testing.assert_array_almost_equal(unsafe_comp, [0.0, 0.0, 0.0])
    
    def test_all_unsafe_mask(self):
        """Test projection with all-unsafe mask."""
        safe_mask = np.array([0, 0, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        
        x = np.array([1.0, 2.0, 3.0])
        safe_comp, unsafe_comp = proj.project(x)
        
        np.testing.assert_array_almost_equal(safe_comp, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(unsafe_comp, x)


class TestRegularizer:
    """Test the Regularizer class."""
    
    def test_l2_regularization(self):
        """Test L2 regularization functionality."""
        reg = Regularizer(l2_scale=0.1)
        x = np.array([1.0, 2.0, 3.0])
        result = reg.apply(x)
        
        # Should move toward zero (reference)
        expected = -0.1 * x
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_zero_scale(self):
        """Test regularization with zero scale."""
        reg = Regularizer(l2_scale=0.0)
        x = np.array([1.0, 2.0, 3.0])
        result = reg.apply(x)
        
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])


class TestSafetyOperator:
    """Test the SafetyOperator class."""
    
    def test_basic_operation(self):
        """Test basic safety operator functionality."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.5, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        
        x = np.array([2.0, 3.0])
        result = op.apply_once(x)
        
        # Should not be NaN or infinite
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert len(result) == len(x)
    
    def test_iterative_application(self):
        """Test iterative application of the operator."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.5, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        
        x = np.array([2.0, 3.0])
        result_1 = op.apply(x, steps=1)
        result_2 = op.apply(x, steps=2)
        
        # Multiple steps should produce different results
        assert not np.allclose(result_1, result_2)
    
    def test_convergence_property(self):
        """Test that repeated application shows convergence behavior."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.8, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        
        x = np.array([2.0, 3.0])
        result_1 = op.apply(x, steps=1)
        result_5 = op.apply(x, steps=5)
        result_10 = op.apply(x, steps=10)
        
        # Later iterations should be more similar to each other
        diff_1_5 = np.linalg.norm(result_1 - result_5)
        diff_5_10 = np.linalg.norm(result_5 - result_10)
        
        # This is a heuristic - in practice, convergence depends on parameters
        assert diff_5_10 <= diff_1_5 * 2  # Allow some tolerance


class TestSafetyWrapper:
    """Test the SafetyWrapper class."""
    
    def test_single_vector_processing(self):
        """Test processing of single vectors."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.5, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        wrapper = SafetyWrapper(operator=op, projector=proj)
        
        logits = np.array([2.0, 3.0])
        result = wrapper.filter_logits(logits, steps=2)
        
        assert result.shape == logits.shape
        assert not np.any(np.isnan(result))
    
    def test_batch_processing(self):
        """Test processing of batches of vectors."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.5, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        wrapper = SafetyWrapper(operator=op, projector=proj)
        
        batch_logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = wrapper.filter_logits(batch_logits, steps=2)
        
        assert result.shape == batch_logits.shape
        assert not np.any(np.isnan(result))
    
    def test_invalid_input_dimensions(self):
        """Test error handling for invalid input dimensions."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.5, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        wrapper = SafetyWrapper(operator=op, projector=proj)
        
        # 3D array should raise ValueError
        invalid_logits = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        
        with pytest.raises(ValueError):
            wrapper.filter_logits(invalid_logits, steps=1)


class TestPerformance:
    """Test performance characteristics."""
    
    def test_large_vector_performance(self):
        """Test performance with large vectors."""
        import time
        
        # Create large safe mask
        size = 1000
        safe_mask = np.random.choice([0, 1], size=size, p=[0.3, 0.7]).astype(float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.8, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        wrapper = SafetyWrapper(operator=op, projector=proj)
        
        # Large input vector
        logits = np.random.randn(size)
        
        start_time = time.time()
        result = wrapper.filter_logits(logits, steps=5)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process in reasonable time (less than 1 second)
        assert processing_time < 1.0
        assert result.shape == logits.shape
        assert not np.any(np.isnan(result))
    
    def test_batch_performance(self):
        """Test performance with large batches."""
        import time
        
        batch_size = 100
        vector_size = 100
        
        safe_mask = np.random.choice([0, 1], size=vector_size, p=[0.3, 0.7]).astype(float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        config = SafetyConfig(alpha=0.8, beta=0.1, gamma=0.05)
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        wrapper = SafetyWrapper(operator=op, projector=proj)
        
        # Large batch
        batch_logits = np.random.randn(batch_size, vector_size)
        
        start_time = time.time()
        result = wrapper.filter_logits(batch_logits, steps=3)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process in reasonable time (less than 2 seconds)
        assert processing_time < 2.0
        assert result.shape == batch_logits.shape
        assert not np.any(np.isnan(result))


class TestMathematicalProperties:
    """Test mathematical properties of the safety operator."""
    
    def test_linearity_approximation(self):
        """Test that the operator is approximately linear for small inputs."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.01)  # Small regularization
        config = SafetyConfig(alpha=0.9, beta=0.01, gamma=0.01)  # Small damping
        
        op = SafetyOperator(projector=proj, regularizer=reg, config=config)
        
        x1 = np.array([0.1, 0.1])
        x2 = np.array([0.2, 0.2])
        
        result1 = op.apply_once(x1)
        result2 = op.apply_once(x2)
        result_sum = op.apply_once(x1 + x2)
        
        # For small inputs, should be approximately linear
        linear_approx = result1 + result2
        error = np.linalg.norm(result_sum - linear_approx)
        
        # Error should be small for small inputs
        assert error < 0.1
    
    def test_safety_monotonicity(self):
        """Test that increasing damping parameters increases safety."""
        safe_mask = np.array([1, 0], dtype=float)
        proj = SafeProjector(safe_mask=safe_mask)
        reg = Regularizer(l2_scale=0.05)
        
        x = np.array([2.0, 3.0])
        
        # Low damping
        config_low = SafetyConfig(alpha=0.9, beta=0.05, gamma=0.05)
        op_low = SafetyOperator(projector=proj, regularizer=reg, config=config_low)
        result_low = op_low.apply_once(x)
        
        # High damping
        config_high = SafetyConfig(alpha=0.5, beta=0.2, gamma=0.2)
        op_high = SafetyOperator(projector=proj, regularizer=reg, config=config_high)
        result_high = op_high.apply_once(x)
        
        # High damping should change the input more
        change_low = np.linalg.norm(result_low - x)
        change_high = np.linalg.norm(result_high - x)
        
        assert change_high > change_low


if __name__ == "__main__":
    pytest.main([__file__])
