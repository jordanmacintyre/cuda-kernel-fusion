"""
Unit tests for quantize_int8 PyTorch baseline implementation.
Tests correctness of the reference implementation only.

Run with: pytest tests/test_quantize_int8_pytorch.py -v
"""

import pytest
import torch

from ops.torch.quantize_int8 import quantize_int8_pytorch


class TestQuantizeInt8PyTorch:
    """Test suite for PyTorch baseline quantize_int8 implementation"""

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
        scale = 2.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)
        expected = ((x / scale) + zero_point).round().clamp(-128, 127).to(torch.int8)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    def test_known_values_zero_point(self, device):
        """Test with known input/output pairs"""
        x = torch.tensor([0.0, 128.0, -128.0, 64.0, -64.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)
        # 0.0/1.0 + 0 = 0, 128.0/1.0 + 0 = 128 -> clamped to 127
        # -128.0/1.0 + 0 = -128, 64.0/1.0 + 0 = 64, -64.0/1.0 + 0 = -64
        expected = torch.tensor([0, 127, -128, 64, -64], dtype=torch.int8, device=device)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    def test_known_values_with_zero_point(self, device):
        """Test with known input/output pairs including zero_point offset"""
        x = torch.tensor([0.0, 10.0, -10.0], device=device)
        scale = 1.0
        zero_point = 10.0

        result = quantize_int8_pytorch(x, scale, zero_point)
        # 0.0/1.0 + 10 = 10, 10.0/1.0 + 10 = 20, -10.0/1.0 + 10 = 0
        expected = torch.tensor([10, 20, 0], dtype=torch.int8, device=device)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    def test_clamping_upper_bound(self, device):
        """Test that values > 127 are clamped to 127"""
        x = torch.tensor([200.0, 150.0, 127.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # All should be clamped to 127
        assert torch.all(result <= 127)
        assert result[0].item() == 127
        assert result[1].item() == 127
        assert result[2].item() == 127

    @pytest.mark.cuda
    def test_clamping_lower_bound(self, device):
        """Test that values < -128 are clamped to -128"""
        x = torch.tensor([-200.0, -150.0, -128.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # All should be clamped to -128
        assert torch.all(result >= -128)
        assert result[0].item() == -128
        assert result[1].item() == -128
        assert result[2].item() == -128

    @pytest.mark.cuda
    def test_clamping_with_zero_point(self, device):
        """Test clamping behavior with non-zero zero_point"""
        x = torch.tensor([100.0, 200.0], device=device)
        scale = 1.0
        zero_point = 50.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # 100/1 + 50 = 150 -> clamped to 127
        # 200/1 + 50 = 250 -> clamped to 127
        assert torch.all(result == 127)

    @pytest.mark.cuda
    def test_rounding(self, device):
        """Test that rounding works correctly"""
        x = torch.tensor([0.4, 0.5, 0.6, -0.4, -0.5, -0.6], device=device)
        scale = 1.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # torch.round() uses banker's rounding (round half to even)
        expected_float = (torch.tensor([0.4, 0.5, 0.6, -0.4, -0.5, -0.6]) / 1.0 + 0.0)
        expected_rounded = expected_float.round().to(torch.int8)

        assert torch.equal(result, expected_rounded.to(device))

    @pytest.mark.cuda
    def test_scale_effect(self, device):
        """Test that scale properly affects quantization"""
        x = torch.tensor([10.0, 20.0, 30.0], device=device)
        zero_point = 0.0

        # With scale = 1.0
        result_scale1 = quantize_int8_pytorch(x, 1.0, zero_point)

        # With scale = 10.0 (should reduce values by 10x before quantization)
        result_scale10 = quantize_int8_pytorch(x, 10.0, zero_point)

        # Values should be different (scale10 should have smaller magnitudes)
        assert not torch.equal(result_scale1, result_scale10)

        # Check specific expected values
        # x=10.0, scale=1.0: 10/1 + 0 = 10
        assert result_scale1[0].item() == 10
        # x=10.0, scale=10.0: 10/10 + 0 = 1
        assert result_scale10[0].item() == 1

    @pytest.mark.cuda
    def test_zero_point_effect(self, device):
        """Test that zero_point properly shifts quantization"""
        x = torch.tensor([10.0, 20.0, 30.0], device=device)
        scale = 1.0

        # With zero_point = 0
        result_zp0 = quantize_int8_pytorch(x, scale, 0.0)

        # With zero_point = 10 (should shift all values up by 10)
        result_zp10 = quantize_int8_pytorch(x, scale, 10.0)

        # Values should be shifted by 10
        assert result_zp10[0].item() == result_zp0[0].item() + 10
        assert result_zp10[1].item() == result_zp0[1].item() + 10
        assert result_zp10[2].item() == result_zp0[2].item() + 10

    @pytest.mark.cuda
    def test_shape_preservation(self, device):
        """Test that output shape matches input shape"""
        shapes = [(100,), (10, 10), (5, 5, 5), (2, 3, 4, 5)]

        for shape in shapes:
            x = torch.randn(shape, device=device)
            result = quantize_int8_pytorch(x, 2.0, 0.0)

            assert result.shape == x.shape

    @pytest.mark.cuda
    def test_output_dtype(self, device):
        """Test that output dtype is int8"""
        x = torch.randn(1000, device=device, dtype=torch.float32)

        result = quantize_int8_pytorch(x, 1.5, 0.0)
        assert result.dtype == torch.int8

    @pytest.mark.cuda
    def test_cpu_tensors(self):
        """Test that function works with CPU tensors"""
        x = torch.randn(100, device="cpu")

        result = quantize_int8_pytorch(x, 2.0, 0.0)

        # Should work on CPU too
        assert result.dtype == torch.int8
        assert result.device.type == "cpu"

    @pytest.mark.cuda
    def test_edge_case_zeros(self, device):
        """Test with zero inputs"""
        x = torch.zeros(1000, device=device)

        result = quantize_int8_pytorch(x, 1.0, 0.0)
        expected = torch.zeros(1000, dtype=torch.int8, device=device)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    def test_edge_case_zeros_with_zero_point(self, device):
        """Test with zero inputs and non-zero zero_point"""
        x = torch.zeros(1000, device=device)
        zero_point = 5.0

        result = quantize_int8_pytorch(x, 1.0, zero_point)
        expected = torch.full((1000,), 5, dtype=torch.int8, device=device)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    def test_edge_case_small_scale(self, device):
        """Test with very small scale (large magnification)"""
        x = torch.tensor([0.01, 0.02, 0.03], device=device)
        scale = 0.001  # Small scale = large magnification
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # 0.01 / 0.001 = 10, 0.02 / 0.001 = 20, 0.03 / 0.001 = 30
        expected = torch.tensor([10, 20, 30], dtype=torch.int8, device=device)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    def test_edge_case_large_scale(self, device):
        """Test with very large scale (strong compression)"""
        x = torch.tensor([100.0, 200.0, 300.0], device=device)
        scale = 100.0  # Large scale = strong compression
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # 100 / 100 = 1, 200 / 100 = 2, 300 / 100 = 3
        expected = torch.tensor([1, 2, 3], dtype=torch.int8, device=device)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    @pytest.mark.parametrize("size", [1, 10, 100, 1000, 10000])
    def test_various_sizes(self, device, size):
        """Test with various tensor sizes"""
        x = torch.randn(size, device=device)
        scale = 2.5
        zero_point = 1.0

        result = quantize_int8_pytorch(x, scale, zero_point)
        expected = ((x / scale) + zero_point).round().clamp(-128, 127).to(torch.int8)

        assert torch.equal(result, expected)

    @pytest.mark.cuda
    @pytest.mark.slow
    def test_large_tensors(self, device):
        """Test with larger tensors (stress test)"""
        x = torch.randn(10_000_000, device=device)
        scale = 3.14
        zero_point = 0.5

        result = quantize_int8_pytorch(x, scale, zero_point)

        # Check dtype and shape
        assert result.dtype == torch.int8
        assert result.shape == x.shape

        # Check values are in valid range
        assert torch.all(result >= -128)
        assert torch.all(result <= 127)

    @pytest.mark.cuda
    def test_negative_values(self, device):
        """Test with negative input values"""
        x = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0], device=device)
        scale = 1.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # Check that negative values are preserved (as negative)
        assert result[0].item() == -10
        assert result[1].item() == -5
        assert result[2].item() == -1
        assert result[3].item() == 0
        assert result[4].item() == 1
        assert result[5].item() == 5
        assert result[6].item() == 10

    @pytest.mark.cuda
    def test_range_utilization(self, device):
        """Test that full int8 range can be utilized"""
        # Create values that should map to full range
        x = torch.linspace(-128.0, 127.0, 256, device=device)
        scale = 1.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # Should have values spanning the full range
        assert result.min().item() == -128
        assert result.max().item() == 127

    @pytest.mark.cuda
    def test_asymmetric_range_with_zero_point(self, device):
        """Test asymmetric quantization with zero_point"""
        # Simulate quantizing a [0, 255] range to [-128, 127]
        x = torch.linspace(0.0, 255.0, 256, device=device)
        scale = 1.0
        zero_point = -128.0  # Shift so 0 maps to -128, 255 maps to 127

        result = quantize_int8_pytorch(x, scale, zero_point)

        # 0/1 + (-128) = -128, 255/1 + (-128) = 127
        assert result.min().item() == -128
        assert result.max().item() == 127

    @pytest.mark.cuda
    def test_distribution_preservation(self, device):
        """Test that distribution shape is roughly preserved"""
        # Create a normal distribution
        torch.manual_seed(42)
        x = torch.randn(100000, device=device) * 10.0  # Scale to avoid excessive clamping

        # Use scale that won't cause excessive clamping
        scale = 0.1
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # Mean should be close to zero
        assert abs(result.float().mean().item()) < 5

        # Should have both positive and negative values
        assert (result > 0).sum() > 10000
        assert (result < 0).sum() > 10000

    @pytest.mark.cuda
    @pytest.mark.numerical
    def test_monotonicity(self, device):
        """Test that larger inputs map to larger outputs (when not clamped)"""
        # Use sorted values that won't get clamped
        x = torch.linspace(-100.0, 100.0, 200, device=device)
        scale = 1.0
        zero_point = 0.0

        result = quantize_int8_pytorch(x, scale, zero_point)

        # Result should be monotonically increasing
        diffs = result[1:].int() - result[:-1].int()
        assert torch.all(diffs >= 0), "Quantization should be monotonic"


# Run tests with:
# pytest tests/test_quantize_int8_pytorch.py -v
# pytest tests/test_quantize_int8_pytorch.py -m "not slow"
