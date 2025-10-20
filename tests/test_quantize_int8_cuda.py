"""
Unit tests for quantize_int8 CUDA implementation.
Compares CUDA kernel results against PyTorch baseline.

Run with: pytest tests/test_quantize_int8_cuda.py -v
"""

import pytest
import torch

from ops.cuda import quantize_int8_cuda
from ops.torch.quantize_int8 import quantize_int8_pytorch


class TestQuantizeInt8CUDA:
    """Test suite comparing CUDA implementation against PyTorch baseline"""

    @pytest.fixture
    def device(self):
        """Ensure CUDA is available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    @pytest.mark.cuda
    def test_basic_correctness(self, device):
        """Test basic operation correctness against PyTorch baseline"""
        x = torch.randn(1000, device=device)
        scale = 2.0
        zero_point = 0.0

        # Compute both ways
        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        # Should match exactly for int8
        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_known_values(self, device):
        """Test with known input/output pairs"""
        x = torch.tensor([0.0, 128.0, -128.0, 64.0, -64.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

        # Also check expected values
        expected = torch.tensor([0, 127, -128, 64, -64], dtype=torch.int8, device=device)
        assert torch.equal(result_cuda, expected)

    @pytest.mark.cuda
    def test_with_zero_point(self, device):
        """Test with non-zero zero_point"""
        x = torch.tensor([0.0, 10.0, -10.0], device=device)
        scale = 1.0
        zero_point = 10.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_clamping_upper_bound(self, device):
        """Test that values > 127 are clamped to 127"""
        x = torch.tensor([200.0, 150.0, 127.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)
        assert torch.all(result_cuda == 127)

    @pytest.mark.cuda
    def test_clamping_lower_bound(self, device):
        """Test that values < -128 are clamped to -128"""
        x = torch.tensor([-200.0, -150.0, -128.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)
        assert torch.all(result_cuda == -128)

    @pytest.mark.cuda
    def test_rounding(self, device):
        """Test that rounding works correctly"""
        x = torch.tensor([0.4, 0.5, 0.6, -0.4, -0.5, -0.6], device=device)
        scale = 1.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_scale_effect(self, device):
        """Test that scale properly affects quantization"""
        x = torch.tensor([10.0, 20.0, 30.0], device=device)
        zero_point = 0.0

        # Test with different scales
        for scale in [1.0, 2.0, 10.0]:
            result_cuda = quantize_int8_cuda(x, scale, zero_point)
            result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

            assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_zero_point_effect(self, device):
        """Test that zero_point properly shifts quantization"""
        x = torch.tensor([10.0, 20.0, 30.0], device=device)
        scale = 1.0

        # Test with different zero points
        for zero_point in [0.0, 5.0, 10.0, -10.0]:
            result_cuda = quantize_int8_cuda(x, scale, zero_point)
            result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

            assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_shape_preservation(self, device):
        """Test that output shape matches input shape"""
        shapes = [(100,), (10, 10), (5, 5, 5), (2, 3, 4, 5)]

        for shape in shapes:
            x = torch.randn(shape, device=device)
            result_cuda = quantize_int8_cuda(x, 2.0, 0.0)
            result_pytorch = quantize_int8_pytorch(x, 2.0, 0.0)

            assert result_cuda.shape == result_pytorch.shape
            assert result_cuda.shape == x.shape

    @pytest.mark.cuda
    def test_output_dtype(self, device):
        """Test that output dtype is int8"""
        x = torch.randn(1000, device=device, dtype=torch.float32)

        result = quantize_int8_cuda(x, 1.5, 0.0)
        assert result.dtype == torch.int8

    @pytest.mark.cuda
    def test_input_validation_cpu_tensor(self):
        """Test that CPU tensors are rejected"""
        x = torch.randn(100, device="cpu")

        with pytest.raises(ValueError, match="must be a CUDA tensor"):
            quantize_int8_cuda(x, 2.0, 0.0)

    @pytest.mark.cuda
    def test_input_validation_dtype(self, device):
        """Test that non-float32 tensors are rejected"""
        x = torch.randn(100, device=device, dtype=torch.float16)

        with pytest.raises(ValueError, match="float32"):
            quantize_int8_cuda(x, 2.0, 0.0)

    @pytest.mark.cuda
    def test_edge_case_zeros(self, device):
        """Test with zero inputs"""
        x = torch.zeros(1000, device=device)

        result_cuda = quantize_int8_cuda(x, 1.0, 0.0)
        result_pytorch = quantize_int8_pytorch(x, 1.0, 0.0)

        assert torch.equal(result_cuda, result_pytorch)
        assert torch.all(result_cuda == 0)

    @pytest.mark.cuda
    def test_edge_case_zeros_with_zero_point(self, device):
        """Test with zero inputs and non-zero zero_point"""
        x = torch.zeros(1000, device=device)
        zero_point = 5.0

        result_cuda = quantize_int8_cuda(x, 1.0, zero_point)
        result_pytorch = quantize_int8_pytorch(x, 1.0, zero_point)

        assert torch.equal(result_cuda, result_pytorch)
        assert torch.all(result_cuda == 5)

    @pytest.mark.cuda
    def test_edge_case_small_scale(self, device):
        """Test with very small scale (large magnification)"""
        x = torch.tensor([0.01, 0.02, 0.03], device=device)
        scale = 0.001
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_edge_case_large_scale(self, device):
        """Test with very large scale (strong compression)"""
        x = torch.tensor([100.0, 200.0, 300.0], device=device)
        scale = 100.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_various_sizes(self, device, size):
        """Test with various tensor sizes"""
        x = torch.randn(size, device=device)
        scale = 2.5
        zero_point = 1.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_large_tensors(self, device):
        """Test with larger tensors (stress test)"""
        x = torch.randn(10_000_000, device=device)
        scale = 3.14
        zero_point = 0.5

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        # Should match exactly
        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_negative_values(self, device):
        """Test with negative input values"""
        x = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_range_utilization(self, device):
        """Test that full int8 range can be utilized"""
        x = torch.linspace(-128.0, 127.0, 256, device=device)
        scale = 1.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)
        assert result_cuda.min().item() == -128
        assert result_cuda.max().item() == 127

    @pytest.mark.cuda
    def test_asymmetric_range_with_zero_point(self, device):
        """Test asymmetric quantization with zero_point"""
        x = torch.linspace(0.0, 255.0, 256, device=device)
        scale = 1.0
        zero_point = -128.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)
        assert result_cuda.min().item() == -128
        assert result_cuda.max().item() == 127

    @pytest.mark.cuda
    def test_random_distributions(self, device):
        """Test with various random distributions"""
        torch.manual_seed(42)

        # Test with different distributions
        distributions = [
            torch.randn(10000, device=device),                    # Normal
            torch.randn(10000, device=device) * 10,               # Wider normal
            torch.rand(10000, device=device) * 100 - 50,          # Uniform [-50, 50]
            torch.randn(10000, device=device).abs() * 10,         # Half-normal
        ]

        for x in distributions:
            result_cuda = quantize_int8_cuda(x, 1.0, 0.0)
            result_pytorch = quantize_int8_pytorch(x, 1.0, 0.0)

            assert torch.equal(result_cuda, result_pytorch)

    @pytest.mark.cuda
    def test_monotonicity(self, device):
        """Test that larger inputs map to larger outputs (when not clamped)"""
        x = torch.linspace(-100.0, 100.0, 200, device=device)
        scale = 1.0
        zero_point = 0.0

        result_cuda = quantize_int8_cuda(x, scale, zero_point)
        result_pytorch = quantize_int8_pytorch(x, scale, zero_point)

        assert torch.equal(result_cuda, result_pytorch)

        # Check monotonicity
        diffs = result_cuda[1:].int() - result_cuda[:-1].int()
        assert torch.all(diffs >= 0), "Quantization should be monotonic"


class TestNumericalAccuracy:
    """Detailed numerical accuracy tests comparing CUDA to PyTorch"""

    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device("cuda")

    @pytest.mark.cuda
    @pytest.mark.numerical
    @pytest.mark.slow
    def test_exact_match_large_dataset(self, device):
        """Test that CUDA matches PyTorch exactly on large dataset"""
        torch.manual_seed(123)
        x = torch.randn(1_000_000, device=device) * 50

        result_cuda = quantize_int8_cuda(x, 1.0, 0.0)
        result_pytorch = quantize_int8_pytorch(x, 1.0, 0.0)

        # Should match exactly (no tolerance needed for int8)
        assert torch.equal(result_cuda, result_pytorch)

        # Check that differences are zero
        differences = (result_cuda.int() - result_pytorch.int()).abs()
        assert differences.sum().item() == 0

    @pytest.mark.cuda
    @pytest.mark.numerical
    def test_no_off_by_one_errors(self, device):
        """Test for off-by-one errors in rounding/clamping"""
        # Test boundary values that are prone to off-by-one errors
        boundary_values = torch.tensor([
            -128.5, -128.0, -127.5,
            -0.5, 0.0, 0.5,
            126.5, 127.0, 127.5
        ], device=device)

        result_cuda = quantize_int8_cuda(boundary_values, 1.0, 0.0)
        result_pytorch = quantize_int8_pytorch(boundary_values, 1.0, 0.0)

        assert torch.equal(result_cuda, result_pytorch)


# Run tests with:
# pytest tests/test_quantize_int8_cuda.py -v
# pytest tests/test_quantize_int8_cuda.py::TestQuantizeInt8CUDA::test_basic_correctness -v
# pytest tests/test_quantize_int8_cuda.py -m "not slow"  # Skip slow tests
