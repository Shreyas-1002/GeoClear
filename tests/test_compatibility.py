"""Unit tests for the GeoClear strict compatibility engine."""

import unittest

from geoclear.compatibility import CompatibilityEngine
from geoclear.types import (
    DeclaredData,
    Modality,
    SensorType,
    S2_BANDS_13,
    S1_SAR_POLARIZATIONS,
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
        self.assertTrue(res.is_executable)
        self.assertEqual(len(res.execution_blockers), 0)
        self.assertEqual(len(res.exclusion_reasons), 0)
        self.assertEqual(len(res.missing_inputs), 0)

    def test_dsen2cr_input_compatible_with_full_multimodal_data(self):
        """DSen2-CR must be input-compatible and executable when S1 SAR and all 13 S2 bands are present."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations=set(S1_SAR_POLARIZATIONS),
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
        self.assertEqual(len(res.exclusion_reasons), 0)

    def test_dsen2cr_excluded_when_sar_missing(self):
        """DSen2-CR must be excluded if SAR data is missing."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL},
            sensors={SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations=set(),
            temporal_observations=1,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("dsen2_cr")

        self.assertIsNotNone(res)
        self.assertFalse(res.is_compatible)
        self.assertTrue(any("SAR" in reason for reason in res.exclusion_reasons))

    def test_ecrformer_spatial_dimension_multiple_of_32(self):
        """ECRformer requires spatial dimensions divisible by 32."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations={"VV", "VH"},
            image_height=250,  # Not divisible by 32
            image_width=250,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("ecrformer")
        self.assertFalse(res.is_compatible)
        self.assertTrue(any("32" in reason for reason in res.exclusion_reasons))

    def test_reference_note_when_reference_missing(self):
        """When no clear optical reference is declared, informational note must be added."""
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations={"VV", "VH"},
            has_ground_truth_reference=False,
            has_clear_optical_reference=False,
        )

        report = CompatibilityEngine.evaluate(declared)
        res = report.get_result("ecrformer")
        self.assertTrue(any("clear optical reference" in note.lower() for note in res.notes))

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
