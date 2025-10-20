"""
Unit tests for add_mul_exp PyTorch baseline implementation.
Tests correctness of the reference implementation only.

Run with: pytest tests/test_add_mul_exp_pytorch.py -v
"""

import pytest
import torch

from ops.torch.add_mul_exp import add_mul_exp_pytorch


class TestAddMulExpPyTorch:
    """Test suite for PyTorch baseline add_mul_exp implementation"""

    @pytest.fixture
    def device(self):
        """Ensure CUDA is available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    @pytest.mark.cuda
    def test_basic_correctness(self, device):
        """Test basic operation correctness against manual computation"""
        x = torch.randn(1000, device=device)
        y = torch.randn(1000, device=device)

        result = add_mul_exp_pytorch(x, y)
        expected = torch.exp((x + y) * 2)

        assert torch.allclose(result, expected, rtol=1e-6)

    @pytest.mark.cuda
    def test_known_values(self, device):
        """Test with known input/output pairs"""
        x = torch.tensor([0.0, 1.0, -1.0], device=device)
        y = torch.tensor([0.0, 0.0, 1.0], device=device)

        result = add_mul_exp_pytorch(x, y)

        # exp((0+0)*2) = exp(0) = 1
        # exp((1+0)*2) = exp(2) ≈ 7.389
        # exp((-1+1)*2) = exp(0) = 1
        expected = torch.tensor([
            1.0,
            torch.exp(torch.tensor(2.0)).item(),
            1.0
        ], device=device)

        assert torch.allclose(result, expected, rtol=1e-6)

    @pytest.mark.cuda
    def test_shape_preservation(self, device):
        """Test that output shape matches input shape"""
        shapes = [(100,), (10, 10), (5, 5, 5), (2, 3, 4, 5)]

        for shape in shapes:
            x = torch.randn(shape, device=device)
            y = torch.randn(shape, device=device)
            result = add_mul_exp_pytorch(x, y)

            assert result.shape == x.shape

    @pytest.mark.cuda
    def test_dtype_consistency(self, device):
        """Test that output dtype matches input dtype"""
        x = torch.randn(1000, device=device, dtype=torch.float32)
        y = torch.randn(1000, device=device, dtype=torch.float32)

        result = add_mul_exp_pytorch(x, y)
        assert result.dtype == torch.float32

    @pytest.mark.cuda
    def test_cpu_tensors(self):
        """Test that function works with CPU tensors"""
        x = torch.randn(100, device="cpu")
        y = torch.randn(100, device="cpu")

        result = add_mul_exp_pytorch(x, y)
        expected = torch.exp((x + y) * 2)

        assert torch.allclose(result, expected, rtol=1e-6)
        assert result.device.type == "cpu"

    @pytest.mark.cuda
    def test_edge_case_zeros(self, device):
        """Test with zero inputs"""
        x = torch.zeros(1000, device=device)
        y = torch.zeros(1000, device=device)

        result = add_mul_exp_pytorch(x, y)
        expected = torch.ones(1000, device=device)  # exp(0) = 1

        assert torch.allclose(result, expected, rtol=1e-6)

    @pytest.mark.cuda
    def test_edge_case_large_values(self, device):
        """Test with large values (but avoid overflow)"""
        # exp((15+0)*2) = exp(30) ≈ 1.07e13 (within float32 range)
        x = torch.full((1000,), 15.0, device=device)
        y = torch.zeros(1000, device=device)

        result = add_mul_exp_pytorch(x, y)
        expected = torch.full_like(result, torch.exp(torch.tensor(30.0)))

        assert torch.allclose(result, expected, rtol=1e-4)

    @pytest.mark.cuda
    def test_no_nan_inf_output(self, device):
        """Test that output contains no NaN or Inf values for reasonable inputs"""
        x = torch.randn(10000, device=device)
        y = torch.randn(10000, device=device)

        result = add_mul_exp_pytorch(x, y)

        assert not torch.isnan(result).any(), "Output contains NaN"
        assert not torch.isinf(result).any(), "Output contains Inf"

    @pytest.mark.cuda
    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_various_sizes(self, device, size):
        """Test with various tensor sizes"""
        x = torch.randn(size, device=device)
        y = torch.randn(size, device=device)

        result = add_mul_exp_pytorch(x, y)
        expected = torch.exp((x + y) * 2)

        assert torch.allclose(result, expected, rtol=1e-6)

    @pytest.mark.cuda
    def test_negative_values(self, device):
        """Test with negative input values"""
        x = torch.tensor([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0], device=device)
        y = torch.tensor([-1.0, -1.0, -1.0, 0.0, 1.0, 1.0, 1.0], device=device)

        result = add_mul_exp_pytorch(x, y)
        expected = torch.exp((x + y) * 2)

        assert torch.allclose(result, expected, rtol=1e-6)

    @pytest.mark.cuda
    def test_mixed_signs(self, device):
        """Test with mixed positive and negative values"""
        x = torch.randn(1000, device=device)
        y = -x  # Opposite signs

        result = add_mul_exp_pytorch(x, y)
        # x + y = 0, so exp(0 * 2) = exp(0) = 1
        expected = torch.ones(1000, device=device)

        assert torch.allclose(result, expected, rtol=1e-6)


# Run tests with:
# pytest tests/test_add_mul_exp_pytorch.py -v
# pytest tests/test_add_mul_exp_pytorch.py -m "not slow"
