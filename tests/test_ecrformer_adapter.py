"""Automated tests for the ECRformer model adapter.

Tests checkpoint loading, input channel validation, spatial dimension checks,
preprocessing, finite output assertions, CPU inference, and error handling.
Synthetic tensors are used to verify software execution and tensor mechanics.
"""

import os
import unittest

import numpy as np
import torch

from geoclear.adapters.ecrformer_adapter import ECRformerAdapter, ECRformerInferenceResult
from geoclear.types import S2_BANDS_13


class TestECRformerAdapter(unittest.TestCase):
    """Test suite for ECRformerAdapter mechanics and inference."""

    @classmethod
    def setUpClass(cls):
        """Initialize adapter once for CPU inference tests to avoid repeated weight loading."""
        cls.adapter_light = ECRformerAdapter(variant="light", device="cpu")

    def test_checkpoint_discovery_and_loading(self):
        """Adapter must successfully discover and load the default light checkpoint."""
        self.assertIsNotNone(self.adapter_light.model)
        self.assertEqual(self.adapter_light.variant, "light")
        self.assertEqual(str(self.adapter_light.device), "cpu")
        self.assertTrue(os.path.isfile(self.adapter_light.checkpoint_path))

    def test_full_variant_loading(self):
        """Adapter must also support loading the full model variant."""
        adapter_full = ECRformerAdapter(variant="full", device="cpu")
        self.assertIsNotNone(adapter_full.model)
        self.assertEqual(adapter_full.variant, "full")

    def test_invalid_variant_raises_value_error(self):
        """Specifying an unknown variant must raise a ValueError."""
        with self.assertRaises(ValueError):
            ECRformerAdapter(variant="ultra")

    def test_missing_checkpoint_raises_file_not_found(self):
        """Specifying a non-existent checkpoint path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            ECRformerAdapter(checkpoint_path="/nonexistent/path/weights.ckpt")

    def test_device_selection_and_validation(self):
        """Requesting 'cuda' when unavailable must raise a descriptive ValueError."""
        if not torch.cuda.is_available():
            with self.assertRaises(ValueError):
                ECRformerAdapter(variant="light", device="cuda")
        else:
            adapter_cuda = ECRformerAdapter(variant="light", device="cuda")
            self.assertEqual(str(adapter_cuda.device), "cuda")

    def test_correct_input_channels_and_successful_cpu_inference(self):
        """Valid 2-channel SAR and 13-channel Optical synthetic inputs must produce valid outputs."""
        h, w = 64, 64
        # Synthetic SAR: values in typical dB range [-25.0, 0.0]
        synthetic_sar = np.random.uniform(-25.0, 0.0, (2, h, w)).astype(np.float32)
        # Synthetic Optical: values in typical DN range [0.0, 10000.0]
        synthetic_optical = np.random.uniform(0.0, 10000.0, (13, h, w)).astype(np.float32)

        result = self.adapter_light.predict(synthetic_sar, synthetic_optical)

        self.assertIsInstance(result, ECRformerInferenceResult)
        # 13 Sentinel-2 bands output
        self.assertEqual(result.reconstructed_multispectral.shape, (13, h, w))
        # Expected bands list
        self.assertEqual(result.bands, list(S2_BANDS_13))
        # RGB preview
        self.assertEqual(result.reconstructed_rgb_preview.shape, (h, w, 3))
        self.assertEqual(result.reconstructed_rgb_preview.dtype, np.uint8)
        self.assertEqual(result.input_cloudy_rgb_preview.shape, (h, w, 3))
        self.assertEqual(result.input_sar_preview.shape, (h, w, 3))

    def test_finite_output_values_and_output_scaling(self):
        """Output reflectance must contain only finite numbers bounded in [0.0, 1.0]."""
        h, w = 64, 64
        synthetic_sar = np.random.uniform(-25.0, 0.0, (2, h, w)).astype(np.float32)
        synthetic_optical = np.random.uniform(0.0, 10000.0, (13, h, w)).astype(np.float32)

        result = self.adapter_light.predict(synthetic_sar, synthetic_optical)
        recon = result.reconstructed_multispectral

        self.assertFalse(np.isnan(recon).any(), "Reconstructed output contains NaNs.")
        self.assertFalse(np.isinf(recon).any(), "Reconstructed output contains Infs.")
        self.assertGreaterEqual(float(recon.min()), 0.0)
        self.assertLessEqual(float(recon.max()), 1.0)
        self.assertTrue(result.metadata["finite_values_verified"])

    def test_nan_and_inf_handling_in_preprocessing(self):
        """Inputs with NaNs and Infs must be sanitized by preprocessing without crashing."""
        h, w = 64, 64
        synthetic_sar = np.random.uniform(-25.0, 0.0, (2, h, w)).astype(np.float32)
        synthetic_optical = np.random.uniform(0.0, 10000.0, (13, h, w)).astype(np.float32)

        # Introduce NaN and Inf into synthetic inputs
        synthetic_sar[0, 10, 10] = np.nan
        synthetic_sar[1, 20, 20] = np.inf
        synthetic_optical[3, 5, 5] = np.nan
        synthetic_optical[7, 15, 15] = -np.inf

        result = self.adapter_light.predict(synthetic_sar, synthetic_optical)
        self.assertFalse(np.isnan(result.reconstructed_multispectral).any())
        self.assertFalse(np.isinf(result.reconstructed_multispectral).any())

    def test_invalid_sar_channel_count_raises_value_error(self):
        """SAR with channel count != 2 must be rejected with descriptive error."""
        h, w = 64, 64
        optical = np.random.uniform(0.0, 10000.0, (13, h, w)).astype(np.float32)

        # 1-channel SAR
        with self.assertRaises(ValueError) as ctx:
            self.adapter_light.predict(np.zeros((1, h, w), dtype=np.float32), optical)
        self.assertIn("exactly 2 channels", str(ctx.exception))

        # 3-channel SAR
        with self.assertRaises(ValueError) as ctx:
            self.adapter_light.predict(np.zeros((3, h, w), dtype=np.float32), optical)
        self.assertIn("exactly 2 channels", str(ctx.exception))

    def test_invalid_optical_channel_count_raises_value_error(self):
        """Optical with channel count != 13 must be rejected with descriptive error."""
        h, w = 64, 64
        sar = np.random.uniform(-25.0, 0.0, (2, h, w)).astype(np.float32)

        # 3-channel RGB optical
        with self.assertRaises(ValueError) as ctx:
            self.adapter_light.predict(sar, np.zeros((3, h, w), dtype=np.float32))
        self.assertIn("exactly 13 Sentinel-2 channels", str(ctx.exception))

        # 12-channel optical
        with self.assertRaises(ValueError) as ctx:
            self.adapter_light.predict(sar, np.zeros((12, h, w), dtype=np.float32))
        self.assertIn("exactly 13 Sentinel-2 channels", str(ctx.exception))

    def test_mismatched_spatial_dimensions_raise_value_error(self):
        """Mismatched spatial heights/widths between SAR and Optical must be rejected."""
        sar = np.zeros((2, 64, 64), dtype=np.float32)
        optical = np.zeros((13, 64, 128), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            self.adapter_light.predict(sar, optical)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_dimensions_not_divisible_by_32_raise_value_error(self):
        """Dimensions not divisible by 32 must raise actionable error."""
        sar = np.zeros((2, 50, 50), dtype=np.float32)
        optical = np.zeros((13, 50, 50), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            self.adapter_light.predict(sar, optical)
        self.assertIn("divisible by 32", str(ctx.exception))

    def test_dimensions_below_minimum_raise_value_error(self):
        """Dimensions below 32x32 must be rejected."""
        sar = np.zeros((2, 16, 16), dtype=np.float32)
        optical = np.zeros((13, 16, 16), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            self.adapter_light.predict(sar, optical)
        self.assertIn("at least 32x32", str(ctx.exception))

    def test_torch_tensor_inputs_supported(self):
        """Adapter must accept PyTorch Tensors in addition to NumPy arrays."""
        h, w = 64, 64
        sar_tensor = torch.randn(2, h, w)
        opt_tensor = torch.rand(13, h, w) * 10000.0

        result = self.adapter_light.predict(sar_tensor, opt_tensor)
        self.assertEqual(result.reconstructed_multispectral.shape, (13, h, w))


if __name__ == "__main__":
    unittest.main()
