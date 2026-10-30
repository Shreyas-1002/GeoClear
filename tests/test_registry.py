"""Unit tests for the GeoClear model registry."""

import unittest

from geoclear.registry import ModelRegistry
from geoclear.types import (
    AuditStatus,
    ConfigurationStatus,
    ImplementationStatus,
    Modality,
    ReadinessStatus,
    SensorType,
    VerificationStatus,
    S2_BANDS_13,
)


class TestModelRegistry(unittest.TestCase):
    """Test suite verifying ModelRegistry integrity and candidate specifications."""

    def test_all_four_models_registered(self):
        """All four external candidate models must be registered."""
        model_ids = ModelRegistry.list_model_ids()
        expected_ids = {"ecrformer", "diffcr", "dsen2_cr", "cloudgan"}
        self.assertEqual(set(model_ids), expected_ids)

    def test_model_metadata_completeness(self):
        """Each model card must provide non-empty essential metadata."""
        for model in ModelRegistry.list_models():
            with self.subTest(model_id=model.model_id):
                self.assertTrue(len(model.display_name) > 0)
                self.assertTrue(len(model.description) > 0)
                self.assertTrue(len(model.paper_citation) > 0)
                self.assertTrue(len(model.paper_url) > 0)
                self.assertTrue(len(model.output_description) > 0)
                self.assertTrue(len(model.output_bands) > 0)

    def test_status_separation(self):
        """Audit, configuration, implementation, and verification statuses must be tracked separately."""
        for model in ModelRegistry.list_models():
            with self.subTest(model_id=model.model_id):
                # All 4 models were inspected in Phase 1
                self.assertEqual(model.audit_status, AuditStatus.AUDITED)
                # All 4 models are fully configured in the registry
                self.assertEqual(model.configuration_status, ConfigurationStatus.CONFIGURED)
                if model.model_id in {"ecrformer", "dsen2_cr"}:
                    self.assertEqual(model.implementation_status, ImplementationStatus.IMPLEMENTED)
                    self.assertEqual(model.verification_status, VerificationStatus.SYNTHETIC_INFERENCE_VERIFIED)
                else:
                    self.assertEqual(model.implementation_status, ImplementationStatus.NOT_IMPLEMENTED)
                    self.assertEqual(model.verification_status, VerificationStatus.UNVERIFIED)

    def test_ecrformer_license_flagging(self):
        """ECRformer's unstated repository license must be explicitly flagged."""
        ecrformer = ModelRegistry.get_model("ecrformer")
        self.assertIsNotNone(ecrformer)
        self.assertFalse(ecrformer.has_explicit_repo_license)
        self.assertEqual(ecrformer.license_type, "Unspecified")
        self.assertIn("No explicit LICENSE", ecrformer.license_notes)
        self.assertIn("copyright remains fully with the authors", ecrformer.license_notes)

    def test_diffcr_license_flagging(self):
        """DiffCR's unstated repository license must be explicitly flagged."""
        diffcr = ModelRegistry.get_model("diffcr")
        self.assertIsNotNone(diffcr)
        self.assertFalse(diffcr.has_explicit_repo_license)
        self.assertEqual(diffcr.license_type, "Unspecified")

    def test_checkpoint_presence_accuracy(self):
        """Checkpoint presence flags must strictly reflect actual disk contents."""
        ecrformer = ModelRegistry.get_model("ecrformer")
        diffcr = ModelRegistry.get_model("diffcr")
        dsen2_cr = ModelRegistry.get_model("dsen2_cr")
        cloudgan = ModelRegistry.get_model("cloudgan")

        # ECRformer, DiffCR, and DSen2-CR have verified weights on disk
        self.assertTrue(ecrformer.checkpoint_present_locally)
        self.assertEqual(len(ecrformer.checkpoint_files), 2)
        self.assertTrue(diffcr.checkpoint_present_locally)
        self.assertEqual(len(diffcr.checkpoint_files), 2)
        self.assertTrue(dsen2_cr.checkpoint_present_locally)
        self.assertGreaterEqual(len(dsen2_cr.checkpoint_files), 1)

        # CloudGAN does NOT have weights on disk
        self.assertFalse(cloudgan.checkpoint_present_locally)
        self.assertEqual(len(cloudgan.checkpoint_files), 0)

    def test_blockers_accuracy(self):
        """Known blockers must be properly populated for each candidate."""
        dsen2_cr = ModelRegistry.get_model("dsen2_cr")
        cloudgan = ModelRegistry.get_model("cloudgan")
        diffcr = ModelRegistry.get_model("diffcr")
        ecrformer = ModelRegistry.get_model("ecrformer")

        self.assertEqual(dsen2_cr.readiness_status, ReadinessStatus.READY_FOR_INFERENCE)
        self.assertEqual(len(dsen2_cr.blockers), 0)

        self.assertEqual(cloudgan.readiness_status, ReadinessStatus.BLOCKED_MISSING_WEIGHTS)
        self.assertTrue(any("tf.contrib" in b.lower() or "missing" in b.lower() for b in cloudgan.blockers))

        self.assertEqual(diffcr.readiness_status, ReadinessStatus.BLOCKED_CODE_BUG)
        self.assertTrue(any("bug" in b.lower() for b in diffcr.blockers))

        self.assertEqual(ecrformer.readiness_status, ReadinessStatus.READY_FOR_INFERENCE)
        self.assertEqual(len(ecrformer.blockers), 0)

    def test_ecrformer_input_requirements(self):
        """ECRformer must require co-registered Sentinel-1 SAR and 13 Sentinel-2 bands."""
        ecrformer = ModelRegistry.get_model("ecrformer")
        req = ecrformer.input_requirements

        self.assertIn(Modality.SAR, req.required_modalities)
        self.assertIn(Modality.MULTISPECTRAL, req.required_modalities)
        self.assertIn(SensorType.SENTINEL_1, req.required_sensors)
        self.assertIn(SensorType.SENTINEL_2, req.required_sensors)
        self.assertEqual(req.required_optical_bands, set(S2_BANDS_13))
        self.assertEqual(req.required_sar_polarizations, {"VV", "VH"})
        self.assertEqual(req.expected_channels, 15)
        self.assertTrue(req.requires_coregistration)

    def test_diffcr_input_requirements(self):
        """DiffCR must require 3 temporal observations and optical RGB bands."""
        diffcr = ModelRegistry.get_model("diffcr")
        req = diffcr.input_requirements

        self.assertIn(Modality.OPTICAL, req.required_modalities)
        self.assertIn(Modality.MULTI_TEMPORAL, req.required_modalities)
        self.assertNotIn(Modality.SAR, req.required_modalities)
        self.assertEqual(req.min_temporal_observations, 3)
        self.assertEqual(req.expected_channels, 9)
        self.assertTrue(req.requires_coregistration)

    def test_registry_immutability(self):
        """Mutating a returned ModelCard must not alter the internal registry state."""
        ecrformer = ModelRegistry.get_model("ecrformer")
        ecrformer.display_name = "Tampered Name"

        fresh = ModelRegistry.get_model("ecrformer")
        self.assertEqual(fresh.display_name, "ECRformer")

    def test_nonexistent_model_lookup(self):
        """Looking up a non-existent model ID must return None."""
        self.assertIsNone(ModelRegistry.get_model("nonexistent_model"))


if __name__ == "__main__":
    unittest.main()
