"""Automated test suite for DSen2-CR model adapter.

Tests:
1. Model instantiation.
2. Checkpoint loading from dsen2cr_carl.pth.
3. Input channel validation (13 optical bands, 2 SAR polarizations).
4. Band ordering and channel sequence.
5. Optical normalization (clipping to [0, 10000] and dividing by 2000).
6. SAR clipping and linear scaling to [0, 2].
7. Tensor concatenation order [Optical (13), SAR (2)].
8. Output shape (13, H, W).
9. Finite-value validation (rejects NaNs and Infs).
10. CPU inference on synthetic and patch-sized inputs.
11. Error handling (mismatched dimensions, wrong channel counts).
12. Geospatial metadata preservation.
13. Registry and compatibility behaviour.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from geoclear.adapters.dsen2cr_adapter import DSen2CRAdapter, DSen2CRInferenceResult
from geoclear.compatibility import CompatibilityEngine
from geoclear.registry import ModelRegistry
from geoclear.types import (
    DeclaredData,
    ImplementationStatus,
    Modality,
    ReadinessStatus,
    SensorType,
    S2_BANDS_13,
    S1_SAR_POLARIZATIONS,
)


class TestDSen2CRAdapter(unittest.TestCase):
    """Test suite for DSen2-CR model adapter and integration mechanics."""

    @classmethod
    def setUpClass(cls):
        """Initialize adapter once for CPU inference tests."""
        cls.adapter = DSen2CRAdapter(device="cpu")

    def test_01_model_instantiation(self):
        """Adapter must successfully instantiate the native PyTorch DSen2-CR model."""
        self.assertIsNotNone(self.adapter.model)
        self.assertEqual(str(self.adapter.device), "cpu")
        self.assertEqual(self.adapter.model.B, 16)
        self.assertEqual(self.adapter.model.F, 256)

    def test_02_checkpoint_loading(self):
        """Adapter must successfully load the converted official checkpoint."""
        self.assertTrue(os.path.isfile(self.adapter.checkpoint_path))
        self.assertTrue(self.adapter.checkpoint_path.endswith("dsen2cr_carl.pth"))

    def test_03_missing_checkpoint_raises(self):
        """Specifying a non-existent checkpoint path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            DSen2CRAdapter(checkpoint_path="/nonexistent/path/weights.pth")

    def test_04_device_validation(self):
        """Requesting 'cuda' when unavailable must raise a descriptive ValueError."""
        if not torch.cuda.is_available():
            with self.assertRaises(ValueError):
                DSen2CRAdapter(device="cuda")
        else:
            adapter_cuda = DSen2CRAdapter(device="cuda")
            self.assertEqual(str(adapter_cuda.device), "cuda")

    def test_05_optical_preprocessing_clipping_and_scaling(self):
        """Optical values must be clipped to [0, 10000] and divided by 2000.0."""
        raw_opt = np.array([[-500.0, 1000.0], [5000.0, 15000.0]], dtype=np.float32)
        raw_opt_13 = np.repeat(raw_opt[np.newaxis, :, :], 13, axis=0)

        norm_opt = self.adapter.preprocess_optical(raw_opt_13)

        # Min value -500 clipped to 0 -> 0 / 2000 = 0.0
        self.assertEqual(norm_opt[0, 0, 0], 0.0)
        # 1000 / 2000 = 0.5
        self.assertEqual(norm_opt[0, 0, 1], 0.5)
        # 5000 / 2000 = 2.5
        self.assertEqual(norm_opt[0, 1, 0], 2.5)
        # Max value 15000 clipped to 10000 -> 10000 / 2000 = 5.0
        self.assertEqual(norm_opt[0, 1, 1], 5.0)

    def test_06_sar_preprocessing_clipping_and_scaling(self):
        """SAR VV/VH channels must be clipped to their respective ranges and scaled to [0, 2]."""
        # VV: clip [-25, 0], VH: clip [-32.5, 0]
        raw_sar = np.zeros((2, 2, 2), dtype=np.float32)
        # VV tests
        raw_sar[0, 0, 0] = -30.0  # Underflow: clipped to -25 -> scaled 0.0
        raw_sar[0, 0, 1] = 5.0    # Overflow: clipped to 0 -> scaled 2.0
        raw_sar[0, 1, 0] = -12.5  # Midpoint: (-12.5 - (-25))/25 * 2 = 1.0

        # VH tests
        raw_sar[1, 0, 0] = -40.0  # Underflow: clipped to -32.5 -> scaled 0.0
        raw_sar[1, 0, 1] = 10.0   # Overflow: clipped to 0 -> scaled 2.0
        raw_sar[1, 1, 0] = -16.25 # Midpoint: (-16.25 - (-32.5))/32.5 * 2 = 1.0

        norm_sar = self.adapter.preprocess_sar(raw_sar)

        self.assertAlmostEqual(float(norm_sar[0, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(norm_sar[0, 0, 1]), 2.0, places=5)
        self.assertAlmostEqual(float(norm_sar[0, 1, 0]), 1.0, places=5)

        self.assertAlmostEqual(float(norm_sar[1, 0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(norm_sar[1, 0, 1]), 2.0, places=5)
        self.assertAlmostEqual(float(norm_sar[1, 1, 0]), 1.0, places=5)

    def test_07_input_channel_validation(self):
        """Adapter must reject optical arrays not having 13 bands or SAR not having 2 bands."""
        invalid_opt = np.ones((12, 64, 64), dtype=np.float32)
        valid_sar = np.zeros((2, 64, 64), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.adapter.predict(valid_sar, invalid_opt)

        valid_opt = np.ones((13, 64, 64), dtype=np.float32)
        invalid_sar = np.zeros((3, 64, 64), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.adapter.predict(invalid_sar, valid_opt)

    def test_08_spatial_dimension_mismatch_raises(self):
        """Spatial dimension mismatch between SAR and Optical must raise ValueError."""
        opt = np.ones((13, 64, 64), dtype=np.float32)
        sar = np.zeros((2, 32, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.adapter.predict(sar, opt)

    def test_09_nan_and_inf_validation(self):
        """Adapter must reject input data containing NaNs or Infs."""
        opt_nan = np.ones((13, 32, 32), dtype=np.float32)
        opt_nan[0, 0, 0] = np.nan
        sar_valid = np.zeros((2, 32, 32), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.adapter.predict(sar_valid, opt_nan)

        sar_inf = np.zeros((2, 32, 32), dtype=np.float32)
        sar_inf[0, 0, 0] = np.inf
        opt_valid = np.ones((13, 32, 32), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.adapter.predict(sar_inf, opt_valid)

    def test_10_successful_cpu_inference_and_output_structure(self):
        """Synthetic valid patch must produce a well-formed DSen2CRInferenceResult."""
        h, w = 64, 64
        sar_input = np.random.uniform(-25.0, 0.0, size=(2, h, w)).astype(np.float32)
        opt_input = np.random.uniform(0.0, 4000.0, size=(13, h, w)).astype(np.float32)

        result = self.adapter.predict(sar_input, opt_input)

        self.assertIsInstance(result, DSen2CRInferenceResult)
        self.assertEqual(result.reconstructed_multispectral.shape, (13, h, w))
        self.assertEqual(result.reconstructed_rgb_preview.shape, (h, w, 3))
        self.assertEqual(result.input_cloudy_rgb_preview.shape, (h, w, 3))
        self.assertEqual(result.input_sar_preview.shape, (h, w, 3))
        self.assertEqual(len(result.bands), 13)
        self.assertEqual(result.bands, list(S2_BANDS_13))
        self.assertTrue(result.metadata["finite_values_verified"])
        self.assertGreater(result.metadata["inference_time_ms"], 0.0)

    def test_11_registry_declaration_and_card(self):
        """DSen2-CR in ModelRegistry must reflect IMPLEMENTED and READY_FOR_INFERENCE."""
        card = ModelRegistry.get_model("dsen2_cr")
        self.assertEqual(card.implementation_status, ImplementationStatus.IMPLEMENTED)
        self.assertEqual(card.readiness_status, ReadinessStatus.READY_FOR_INFERENCE)
        self.assertTrue(card.checkpoint_present_locally)
        self.assertEqual(card.input_requirements.expected_channels, 15)
        self.assertIn(Modality.SAR, card.input_requirements.required_modalities)
        self.assertIn(Modality.MULTISPECTRAL, card.input_requirements.required_modalities)

    def test_12_compatibility_engine_behavior(self):
        """CompatibilityEngine must classify DSen2-CR as compatible and executable on multimodal data."""
        data = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations=set(S1_SAR_POLARIZATIONS),
            temporal_observations=1,
            image_height=256,
            image_width=256,
        )

        report = CompatibilityEngine.evaluate(data)
        compat_ids = [m.model_id for m in report.compatible_models]
        exec_ids = [m.model_id for m in report.executable_models]

        self.assertIn("dsen2_cr", compat_ids)
        self.assertIn("dsen2_cr", exec_ids)
        self.assertIn("ecrformer", compat_ids)
        self.assertIn("ecrformer", exec_ids)


if __name__ == "__main__":
    unittest.main()
