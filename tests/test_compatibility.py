"""Unit tests for the GeoClear strict compatibility engine."""

import unittest

from geoclear.compatibility import CompatibilityEngine
from geoclear.types import (
    DeclaredData,
    Modality,
    SensorType,
    S2_BANDS_13,
)


class TestCompatibilityEngine(unittest.TestCase):
    """Test suite verifying strict, rule-based compatibility filtering."""

    def test_ecrformer_excluded_when_sar_missing(self):
        """ECRformer must be excluded if Sentinel-1 SAR data is missing, even with 13 S2 bands."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL},
            sensors={SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations=set(),
            temporal_observations=1,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("ecrformer")

        self.assertIsNotNone(res)
        self.assertFalse(res.is_compatible)
        self.assertFalse(res.is_executable)
        self.assertTrue(any("SAR" in reason for reason in res.exclusion_reasons))
        self.assertTrue(any("SAR" in inp for inp in res.missing_inputs))

    def test_ecrformer_excluded_when_bands_missing(self):
        """ECRformer must be excluded if only RGB bands are provided (10 MS bands missing)."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands={"B4", "B3", "B2"},  # Only RGB
            sar_polarizations={"VV", "VH"},
            temporal_observations=1,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("ecrformer")

        self.assertIsNotNone(res)
        self.assertFalse(res.is_compatible)
        self.assertFalse(res.is_executable)
        # Reason must explain missing spectral bands and that RGB cannot synthesize NIR/SWIR
        self.assertTrue(any("missing 10 bands" in reason for reason in res.exclusion_reasons))
        self.assertTrue(any("Missing optical bands" in inp for inp in res.missing_inputs))

    def test_ecrformer_input_compatible_with_full_multimodal_data(self):
        """ECRformer must be input-compatible and executable when S1 SAR and all 13 S2 bands are present."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations={"VV", "VH"},
            temporal_observations=1,
            image_height=256,
            image_width=256,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("ecrformer")

        self.assertIsNotNone(res)
        self.assertTrue(res.is_compatible)
        # ECRformer adapter is implemented and verified on synthetic tensors
        self.assertTrue(res.is_executable)
        self.assertEqual(len(res.execution_blockers), 0)
        self.assertEqual(len(res.exclusion_reasons), 0)
        self.assertEqual(len(res.missing_inputs), 0)

    def test_diffcr_excluded_when_temporal_count_insufficient(self):
        """DiffCR must be excluded if fewer than 3 temporal observations are available."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL},
            sensors={SensorType.SENTINEL_2},
            optical_bands={"B4", "B3", "B2"},
            temporal_observations=1,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("diffcr")

        self.assertIsNotNone(res)
        self.assertFalse(res.is_compatible)
        self.assertTrue(any("3" in reason for reason in res.exclusion_reasons))
        self.assertTrue(any("temporal" in inp.lower() for inp in res.missing_inputs))

    def test_diffcr_input_compatible_when_three_temporal_rgb_declared(self):
        """DiffCR must be input-compatible when 3 temporal RGB observations are declared."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTI_TEMPORAL},
            sensors={SensorType.SENTINEL_2},
            optical_bands={"B4", "B3", "B2"},
            temporal_observations=3,
            image_height=256,
            image_width=256,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("diffcr")

        self.assertIsNotNone(res)
        self.assertTrue(res.is_compatible)
        # Execution is blocked by upstream code bug
        self.assertFalse(res.is_executable)
        self.assertTrue(any("bug" in b.lower() or "import" in b.lower() for b in res.execution_blockers))

    def test_dsen2_cr_input_compatibility_and_executable(self):
        """DSen2-CR is input-compatible with SAR+MS data and executable with converted weights."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations={"VV", "VH"},
            temporal_observations=1,
            image_height=256,
            image_width=256,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("dsen2_cr")

        self.assertIsNotNone(res)
        self.assertTrue(res.is_compatible)
        self.assertTrue(res.is_executable)
        self.assertEqual(len(res.execution_blockers), 0)

    def test_cloudgan_input_compatibility_vs_execution_blocker(self):
        """CloudGAN is input-compatible with 256x256 RGB data, but blocked by missing weights & TF1."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL},
            sensors={SensorType.GENERIC_RGB},
            optical_bands={"R", "G", "B"},
            temporal_observations=1,
            image_height=256,
            image_width=256,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("cloudgan")

        self.assertIsNotNone(res)
        self.assertTrue(res.is_compatible)
        # Executable must be False due to missing weights and TF1
        self.assertFalse(res.is_executable)
        self.assertTrue(any("missing" in b.lower() or "weights" in b.lower() for b in res.execution_blockers))
        self.assertTrue(any("tensorflow" in b.lower() or "legacy" in b.lower() for b in res.execution_blockers))

    def test_cloudgan_excluded_on_non_256_dimensions(self):
        """CloudGAN must be excluded if dimensions are not 256x256."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL},
            sensors={SensorType.GENERIC_RGB},
            optical_bands={"R", "G", "B"},
            temporal_observations=1,
            image_height=512,
            image_width=512,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("cloudgan")

        self.assertIsNotNone(res)
        self.assertFalse(res.is_compatible)
        self.assertTrue(any("256x256" in reason for reason in res.exclusion_reasons))

    def test_invalid_and_contradictory_declarations_raise_error(self):
        """Contradictory or invalid user declarations must be rejected by validation."""
        # 0 temporal observations
        with self.assertRaises(ValueError):
            DeclaredData(
                modalities={Modality.OPTICAL},
                optical_bands={"R", "G", "B"},
                temporal_observations=0,
            ).validate()

        # Negative image height
        with self.assertRaises(ValueError):
            DeclaredData(
                modalities={Modality.OPTICAL},
                optical_bands={"R", "G", "B"},
                image_height=-10,
            ).validate()

        # SAR modality declared for Sentinel-1 without polarizations
        with self.assertRaises(ValueError):
            DeclaredData(
                modalities={Modality.SAR},
                sensors={SensorType.SENTINEL_1},
                sar_polarizations=set(),
            ).validate()

        # Optical modality declared with empty bands set
        with self.assertRaises(ValueError):
            DeclaredData(
                modalities={Modality.OPTICAL},
                sensors={SensorType.GENERIC_RGB},
                optical_bands=set(),
            ).validate()

        # Multi-temporal modality declared with 1 observation
        with self.assertRaises(ValueError):
            DeclaredData(
                modalities={Modality.OPTICAL, Modality.MULTI_TEMPORAL},
                sensors={SensorType.SENTINEL_2},
                optical_bands={"B4", "B3", "B2"},
                temporal_observations=1,
            ).validate()


if __name__ == "__main__":
    unittest.main()
