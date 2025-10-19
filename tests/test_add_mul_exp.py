"""
Unit tests for add_mul_exp operation.
Tests correctness, not performance.

Run with: pytest tests/
"""

import pytest
import torch

from ops.cuda import add_mul_exp


class TestAddMulExp:
    """Test suite for add_mul_exp fused kernel"""

    @pytest.fixture
    def device(self):
        """Ensure CUDA is available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    @pytest.mark.cuda
    def test_basic_correctness(self, device):
        """Test basic operation correctness"""
        x = torch.randn(1000, device=device)
        y = torch.randn(1000, device=device)

        # Compute both ways
        result_fused = add_mul_exp(x, y)
        result_pytorch = torch.exp((x + y) * 2)

        # Should match within tolerance
        assert torch.allclose(result_fused, result_pytorch, rtol=1e-5)

    @pytest.mark.cuda
    def test_known_values(self, device):
        """Test with known input/output pairs"""
        x = torch.tensor([0.0, 1.0, -1.0], device=device)
        y = torch.tensor([0.0, 0.0, 1.0], device=device)

        result = add_mul_exp(x, y)
        expected = torch.tensor(
            [
                torch.exp(torch.tensor(0.0)),  # exp((0+0)*2) = exp(0) = 1
                torch.exp(torch.tensor(2.0)),  # exp((1+0)*2) = exp(2) = 7.389
                torch.exp(torch.tensor(0.0)),  # exp((-1+1)*2) = exp(0) = 1
            ],
            device=device,
        )

        assert torch.allclose(result, expected, rtol=1e-5)

    @pytest.mark.cuda
    def test_shape_preservation(self, device):
        """Test that output shape matches input shape"""
        shapes = [(100,), (10, 10), (5, 5, 5), (2, 3, 4, 5)]

        for shape in shapes:
            x = torch.randn(shape, device=device)
            y = torch.randn(shape, device=device)
            result = add_mul_exp(x, y)

            assert result.shape == x.shape

    @pytest.mark.cuda
    def test_dtype_consistency(self, device):
        """Test that output dtype matches input dtype"""
        x = torch.randn(1000, device=device, dtype=torch.float32)
        y = torch.randn(1000, device=device, dtype=torch.float32)

        result = add_mul_exp(x, y)
        assert result.dtype == torch.float32

    @pytest.mark.cuda
    def test_input_validation_cpu_tensor(self, device):
        """Test that CPU tensors are rejected"""
        x = torch.randn(100, device="cpu")
        y = torch.randn(100, device=device)

        with pytest.raises(ValueError, match="must be a CUDA tensor"):
            add_mul_exp(x, y)

    @pytest.mark.cuda
    def test_input_validation_shape_mismatch(self, device):
        """Test that mismatched shapes are rejected"""
        x = torch.randn(100, device=device)
        y = torch.randn(200, device=device)

        with pytest.raises(ValueError, match="Shape mismatch"):
            add_mul_exp(x, y)

    @pytest.mark.cuda
    def test_input_validation_dtype(self, device):
        """Test that non-float32 tensors are rejected"""
        x = torch.randn(100, device=device, dtype=torch.float16)
        y = torch.randn(100, device=device, dtype=torch.float16)

        with pytest.raises(ValueError, match="float32"):
            add_mul_exp(x, y)

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_large_tensors(self, device):
        """Test with larger tensors (stress test)"""
        x = torch.randn(10_000_000, device=device)
        y = torch.randn(10_000_000, device=device)

        result = add_mul_exp(x, y)
        expected = torch.exp((x + y) * 2)

        # Looser tolerance for large tensors
        assert torch.allclose(result, expected, rtol=1e-4)

    @pytest.mark.cuda
    def test_edge_cases_zeros(self, device):
        """Test with zero inputs"""
        x = torch.zeros(1000, device=device)
        y = torch.zeros(1000, device=device)

        result = add_mul_exp(x, y)
        expected = torch.ones(1000, device=device)  # exp(0) = 1

        assert torch.allclose(result, expected, rtol=1e-6)

    @pytest.mark.cuda
    def test_edge_cases_large_values(self, device):
        """Test with large values (avoid overflow)"""
        # exp(15*2) = exp(30) ≈ 1e13 (still within float32 range)
        x = torch.full((1000,), 15.0, device=device)
        y = torch.zeros(1000, device=device)

        result = add_mul_exp(x, y)
        expected = torch.exp(torch.tensor(30.0))

        assert torch.allclose(result, torch.full_like(result, expected), rtol=1e-4)

    @pytest.mark.cuda
    @pytest.mark.numerical
    def test_numerical_stability(self, device):
        """Test numerical stability with mixed magnitudes"""
        # Use values that won't overflow: exp((5+5)*2) = exp(20) ≈ 4.85e8 (safe for float32)
        # Max safe value for exp in float32 is ~88, so (x+y)*2 should be < 88
        x = torch.tensor([1e-7, 1e0, 5.0], device=device)
        y = torch.tensor([1e-7, 1e0, 5.0], device=device)

        result = add_mul_exp(x, y)
        expected = torch.exp((x + y) * 2)

        # Check no NaNs or Infs
        assert not torch.isnan(result).any(), "Result contains NaN values"
        assert not torch.isinf(result).any(), "Result contains Inf values"

        # Check reasonable accuracy
        assert torch.allclose(result, expected, rtol=1e-5)

    @pytest.mark.cuda
    def test_no_nan_inf_output(self, device):
        """Test that output contains no NaN or Inf values"""
        x = torch.randn(10000, device=device)
        y = torch.randn(10000, device=device)

        result = add_mul_exp(x, y)

        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

    @pytest.mark.cuda
    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_various_sizes(self, device, size):
        """Test with various tensor sizes"""
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)

        result = add_mul_exp(x, y)
        expected = torch.exp((x + y) * 2)

        assert torch.allclose(result, expected, rtol=1e-5)


class TestNumericalAccuracy:
    """Detailed numerical accuracy tests"""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    @pytest.mark.cuda
    @pytest.mark.numerical
    @pytest.mark.slow
    def test_max_relative_error(self, device):
        """Test maximum relative error is acceptable"""
        x = torch.randn(100000, device=device)
        y = torch.randn(100000, device=device)

        result_fused = add_mul_exp(x, y)
        result_pytorch = torch.exp((x + y) * 2)

        # Calculate relative error
        abs_diff = torch.abs(result_fused - result_pytorch)
        denominator = torch.maximum(torch.abs(result_pytorch), torch.abs(result_fused))
        rel_error = abs_diff / (denominator + 1e-10)

        max_rel_error = rel_error.max().item()

        # Should be less than 1e-5 (0.001%)
        assert (
            max_rel_error < 1e-5
        ), f"Max relative error {max_rel_error:.2e} exceeds threshold"

    @pytest.mark.cuda
    @pytest.mark.numerical
    @pytest.mark.slow
    def test_mean_relative_error(self, device):
        """Test mean relative error is very small"""
        x = torch.randn(100000, device=device)
        y = torch.randn(100000, device=device)

        result_fused = add_mul_exp(x, y)
        result_pytorch = torch.exp((x + y) * 2)

        abs_diff = torch.abs(result_fused - result_pytorch)
        denominator = torch.maximum(torch.abs(result_pytorch), torch.abs(result_fused))
        rel_error = abs_diff / (denominator + 1e-10)

        mean_rel_error = rel_error.mean().item()

        # Mean should be much smaller
        assert (
            mean_rel_error < 1e-6
        ), f"Mean relative error {mean_rel_error:.2e} too large"


# Run tests with:
# pytest tests/test_add_mul_exp.py -v
# pytest tests/test_add_mul_exp.py::TestAddMulExp::test_basic_correctness -v
