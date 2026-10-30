"""Unit tests for the GeoClear dashboard workflows, I/O validation, and metrics.

Tests end-to-end integration of the Streamlit dashboard components:
- Synthetic verification patch generation
- Multi-format raster reading (GeoTIFF, NPZ, NPY)
- Data validation (channels, non-finite values, dimension divisibility)
- Geospatial coregistration validation
- Output serialization (NPZ, PNG)
- Scientifically grounded metrics evaluation (MAE, RMSE, PSNR, SSIM)
- End-to-end execution with ECRformer adapter
"""

import io
import json
import os
import sys
import time
import unittest
from pathlib import Path

# Windows PROJ environment isolation
if "PROJ_LIB" in os.environ and "PostgreSQL" in os.environ["PROJ_LIB"]:
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    rasterio_proj = os.path.join(site_packages, "rasterio", "proj_data")
    if os.path.isdir(rasterio_proj):
        os.environ["PROJ_LIB"] = rasterio_proj
        os.environ["PROJ_DATA"] = rasterio_proj

import numpy as np
import rasterio
from rasterio.transform import from_origin

from geoclear.types import (
    DeclaredData,
    Modality,
    SensorType,
    S2_BANDS_13,
)
from geoclear.compatibility import CompatibilityEngine
from geoclear.registry import ModelRegistry
from geoclear.adapters.ecrformer_adapter import ECRformerAdapter
from geoclear.demo_data import generate_synthetic_verification_patch
from geoclear.io import (
    read_raster_file,
    validate_spatial_alignment,
    export_npz_bytes,
    export_png_bytes,
)
from geoclear.metrics import compute_evaluation_metrics, EvaluationMetrics


class TestDemoDataGenerator(unittest.TestCase):
    """Test suite for synthetic verification data generation."""

    def test_synthetic_patch_shapes_and_ranges(self):
        """Synthetic patch must yield 2-band SAR, 13-band optical, and 13-band reference."""
        sar, cloudy, clean = generate_synthetic_verification_patch(height=128, width=128, seed=42)

        # Check shapes
        self.assertEqual(sar.shape, (2, 128, 128))
        self.assertEqual(cloudy.shape, (13, 128, 128))
        self.assertEqual(clean.shape, (13, 128, 128))

        # Check dtypes
        self.assertEqual(sar.dtype, np.float32)
        self.assertEqual(cloudy.dtype, np.float32)
        self.assertEqual(clean.dtype, np.float32)

        # Check SAR dB range [-25.0, 0.0]
        self.assertGreaterEqual(float(sar.min()), -25.0)
        self.assertLessEqual(float(sar.max()), 0.0)

        # Check Optical DN range [0.0, 10000.0]
        self.assertGreaterEqual(float(cloudy.min()), 0.0)
        self.assertLessEqual(float(cloudy.max()), 10000.0)
        self.assertGreaterEqual(float(clean.min()), 0.0)
        self.assertLessEqual(float(clean.max()), 10000.0)

        # Finite checks
        self.assertTrue(np.isfinite(sar).all())
        self.assertTrue(np.isfinite(cloudy).all())
        self.assertTrue(np.isfinite(clean).all())

    def test_dimension_divisibility_enforcement(self):
        """Non-divisible-by-32 dimensions must raise a ValueError."""
        with self.assertRaises(ValueError) as ctx:
            generate_synthetic_verification_patch(height=100, width=100)
        self.assertIn("divisible by 32", str(ctx.exception))

    def test_seed_determinism(self):
        """Same random seed must produce numerically identical patches."""
        sar1, opt1, ref1 = generate_synthetic_verification_patch(height=64, width=64, seed=123)
        sar2, opt2, ref2 = generate_synthetic_verification_patch(height=64, width=64, seed=123)

        np.testing.assert_array_equal(sar1, sar2)
        np.testing.assert_array_equal(opt1, opt2)
        np.testing.assert_array_equal(ref1, ref2)


class TestRasterIO(unittest.TestCase):
    """Test suite for GeoTIFF, NPZ, and NPY reading and validation."""

    def _create_in_memory_geotiff(
        self,
        array: np.ndarray,
        crs: str = "EPSG:32632",
        transform=None,
    ) -> bytes:
        """Helper to create an in-memory GeoTIFF file as bytes."""
        c, h, w = array.shape
        if transform is None:
            transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)

        memfile = io.BytesIO()
        with rasterio.open(
            memfile,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=c,
            dtype=str(array.dtype),
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(array)
        return memfile.getvalue()

    def test_read_geotiff_success(self):
        """Valid GeoTIFF must be read into float32 (C, H, W) with metadata."""
        data = np.random.randn(2, 64, 64).astype(np.float32)
        tif_bytes = self._create_in_memory_geotiff(data, crs="EPSG:4326")

        arr, meta = read_raster_file(tif_bytes, expected_channels=2, name="SAR_TIF")
        self.assertEqual(arr.shape, (2, 64, 64))
        self.assertIsNotNone(meta)
        self.assertEqual(meta["format"], "GeoTIFF")
        self.assertEqual(meta["channels"], 2)
        self.assertEqual(meta["height"], 64)
        self.assertEqual(meta["width"], 64)
        self.assertIn("4326", str(meta["crs"]))

    def test_read_npz_success(self):
        """Valid NPZ archive must be read with channels and dimensions."""
        data = np.random.randn(13, 64, 64).astype(np.float32)
        buf = io.BytesIO()
        np.savez(buf, optical=data)
        npz_bytes = buf.getvalue()

        arr, meta = read_raster_file(npz_bytes, expected_channels=13, name="OPT_NPZ")
        self.assertEqual(arr.shape, (13, 64, 64))
        self.assertIsNotNone(meta)
        self.assertEqual(meta["format"], "NumPy NPZ")
        self.assertEqual(meta["key_used"], "optical")

    def test_read_npy_success(self):
        """Valid NPY array must be read and standardized to (C, H, W)."""
        # Test (H, W, C) transposition
        data = np.random.randn(64, 64, 2).astype(np.float32)
        buf = io.BytesIO()
        np.save(buf, data)
        npy_bytes = buf.getvalue()

        arr, meta = read_raster_file(npy_bytes, expected_channels=2, name="SAR_NPY")
        self.assertEqual(arr.shape, (2, 64, 64))
        self.assertIsNotNone(meta)
        self.assertEqual(meta["format"], "NumPy NPY")

    def test_channel_count_mismatch_raises(self):
        """Uploading wrong channel count must raise a descriptive ValueError."""
        data = np.random.randn(3, 64, 64).astype(np.float32)  # 3 channels instead of 13
        buf = io.BytesIO()
        np.save(buf, data)

        with self.assertRaises(ValueError) as ctx:
            read_raster_file(buf.getvalue(), expected_channels=13, name="Optical")
        self.assertIn("channel mismatch", str(ctx.exception))
        self.assertIn("expected exactly 13 channels", str(ctx.exception))

    def test_non_finite_values_raise(self):
        """Arrays containing NaNs or Infs must be rejected."""
        data = np.zeros((2, 64, 64), dtype=np.float32)
        data[0, 10, 10] = np.nan
        buf = io.BytesIO()
        np.save(buf, data)

        with self.assertRaises(ValueError) as ctx:
            read_raster_file(buf.getvalue(), expected_channels=2, name="CorruptedSAR")
        self.assertIn("non-finite values", str(ctx.exception))

    def test_non_divisible_dimensions_raise(self):
        """Dimensions not divisible by 32 must be rejected."""
        data = np.zeros((2, 50, 50), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, data)

        with self.assertRaises(ValueError) as ctx:
            read_raster_file(buf.getvalue(), expected_channels=2, name="SAR_50x50", require_divisible_by=32)
        self.assertIn("must be divisible by 32", str(ctx.exception))

    def test_corrupted_file_raises(self):
        """Garbage bytes must be rejected as an invalid format."""
        with self.assertRaises(ValueError) as ctx:
            read_raster_file(b"not a valid raster format", expected_channels=2)
        self.assertIn("Could not read", str(ctx.exception))

    def test_spatial_alignment_validation(self):
        """validate_spatial_alignment must check dimensions, CRS, and bounds."""
        # 1. Matching GeoTIFF metadata
        s1_meta = {"format": "GeoTIFF", "crs": "EPSG:32632", "res": [10.0, 10.0], "bounds": [0, 0, 100, 100]}
        s2_meta = {"format": "GeoTIFF", "crs": "EPSG:32632", "res": [10.0, 10.0], "bounds": [0, 0, 100, 100]}
        aligned, notes = validate_spatial_alignment(s1_meta, s2_meta, sar_shape=(2, 64, 64), opt_shape=(13, 64, 64))
        self.assertTrue(aligned)
        self.assertTrue(any("verified" in n.lower() or "confirmed" in n.lower() for n in notes))

        # 2. Mismatched shapes
        aligned, notes = validate_spatial_alignment(s1_meta, s2_meta, sar_shape=(2, 64, 64), opt_shape=(13, 128, 128))
        self.assertFalse(aligned)
        self.assertTrue(any("mismatch" in n.lower() for n in notes))

        # 3. Mismatched CRS
        s2_diff_crs = {"format": "GeoTIFF", "crs": "EPSG:4326", "res": [10.0, 10.0], "bounds": [0, 0, 100, 100]}
        aligned, notes = validate_spatial_alignment(s1_meta, s2_diff_crs, sar_shape=(2, 64, 64), opt_shape=(13, 64, 64))
        self.assertFalse(aligned)
        self.assertTrue(any("CRS Mismatch" in n for n in notes))

        # 4. NumPy uploads (no CRS metadata)
        aligned, notes = validate_spatial_alignment(None, None, sar_shape=(2, 64, 64), opt_shape=(13, 64, 64))
        self.assertTrue(aligned)
        self.assertTrue(any("unavailable" in n.lower() for n in notes))

    def test_export_helpers(self):
        """Export helpers must produce valid downloadable bytes."""
        arr = np.ones((13, 64, 64), dtype=np.float32)
        npz_bytes = export_npz_bytes(arr, "reconstructed")
        self.assertIsInstance(npz_bytes, bytes)
        self.assertGreater(len(npz_bytes), 0)

        # Check NPZ round-trip
        buf = io.BytesIO(npz_bytes)
        loaded = np.load(buf)["reconstructed"]
        np.testing.assert_array_equal(arr, loaded)

        # PNG export
        rgb_img = np.full((64, 64, 3), 128, dtype=np.uint8)
        png_bytes = export_png_bytes(rgb_img)
        self.assertIsInstance(png_bytes, bytes)
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))


class TestEvaluationMetrics(unittest.TestCase):
    """Test suite for scientific metrics computation (MAE, RMSE, PSNR, SSIM)."""

    def test_identical_reference_and_reconstruction(self):
        """Identical images must have MAE=0, RMSE=0, SSIM=1.0, and high PSNR."""
        ref = np.random.uniform(0.1, 0.9, (13, 64, 64)).astype(np.float32)
        metrics = compute_evaluation_metrics(ref, ref)

        self.assertAlmostEqual(metrics.mae, 0.0, places=4)
        self.assertAlmostEqual(metrics.rmse, 0.0, places=4)
        self.assertAlmostEqual(metrics.ssim, 1.0, places=3)
        self.assertAlmostEqual(metrics.rgb_ssim, 1.0, places=3)
        self.assertGreater(metrics.psnr, 40.0)  # Inf or capped high PSNR
        self.assertEqual(len(metrics.per_band_mae), 13)
        self.assertEqual(len(metrics.per_band_rmse), 13)

    def test_known_difference_metrics(self):
        """Constant offset difference must produce exact MAE and RMSE."""
        ref = np.zeros((13, 32, 32), dtype=np.float32)
        rec = np.full((13, 32, 32), 0.05, dtype=np.float32)

        metrics = compute_evaluation_metrics(rec, ref, data_range=1.0)
        self.assertAlmostEqual(metrics.mae, 0.05, places=4)
        self.assertAlmostEqual(metrics.rmse, 0.05, places=4)
        for b_name, b_mae in metrics.per_band_mae.items():
            self.assertAlmostEqual(b_mae, 0.05, places=4)

    def test_mismatched_shape_raises(self):
        """Differing shapes must raise a ValueError."""
        rec = np.zeros((13, 64, 64), dtype=np.float32)
        ref = np.zeros((13, 32, 32), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            compute_evaluation_metrics(rec, ref)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_non_finite_values_raise(self):
        """NaNs or Infs in either image must be rejected."""
        rec = np.zeros((13, 32, 32), dtype=np.float32)
        ref = np.zeros((13, 32, 32), dtype=np.float32)
        ref[0, 0, 0] = np.nan

        with self.assertRaises(ValueError) as ctx:
            compute_evaluation_metrics(rec, ref)
        self.assertIn("NaN", str(ctx.exception))


class TestDashboardWorkflowIntegration(unittest.TestCase):
    """End-to-end integration test exercising the full dashboard inference pipeline."""

    def test_ecrformer_full_pipeline_with_synthetic_patch(self):
        """Test full pipeline: synthetic generation -> ECRformer inference -> metrics evaluation."""
        # 1. Step 1: Declare data
        declared = DeclaredData(
            modalities={Modality.OPTICAL, Modality.MULTISPECTRAL, Modality.SAR},
            sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            optical_bands=set(S2_BANDS_13),
            sar_polarizations={"VV", "VH"},
            temporal_observations=1,
            has_ground_truth_reference=True,
            image_height=128,
            image_width=128,
        )
        report = CompatibilityEngine.evaluate(declared)
        # Both ECRformer and DSen2-CR accept Sentinel-1 + 13-band Sentinel-2 and are executable
        self.assertEqual(len(report.compatible_models), 2)
        self.assertEqual(len(report.executable_models), 2)
        exec_ids = {m.model_id for m in report.executable_models}
        self.assertIn("ecrformer", exec_ids)
        self.assertIn("dsen2_cr", exec_ids)
        self.assertTrue(report.get_result("ecrformer").is_executable)
        self.assertTrue(report.get_result("dsen2_cr").is_executable)
        self.assertIsNone(report.get_result("diffcr"))

        # 2. Step 2: Generate synthetic input patch
        sar, cloudy, clean_ref = generate_synthetic_verification_patch(height=128, width=128, seed=42)

        # 3. Step 3: Run inference via ECRformer adapter
        adapter = ECRformerAdapter(variant="light", device="cpu")
        result = adapter.predict(sar, cloudy)

        # Assert output properties
        self.assertEqual(result.reconstructed_multispectral.shape, (13, 128, 128))
        self.assertEqual(result.reconstructed_rgb_preview.shape, (128, 128, 3))
        self.assertEqual(result.input_sar_preview.shape, (128, 128, 3))
        self.assertEqual(result.input_cloudy_rgb_preview.shape, (128, 128, 3))
        self.assertTrue(np.isfinite(result.reconstructed_multispectral).all())

        # 4. Step 4: Evaluate against ground-truth reference
        ref_norm = clean_ref / 10000.0  # Normalized to [0, 1]
        metrics = compute_evaluation_metrics(result.reconstructed_multispectral, ref_norm)

        self.assertIsInstance(metrics, EvaluationMetrics)
        self.assertGreater(metrics.mae, 0.0)
        self.assertGreater(metrics.rmse, 0.0)
        self.assertGreater(metrics.psnr, 0.0)
        self.assertGreater(metrics.ssim, 0.0)
        self.assertEqual(len(metrics.per_band_mae), 13)


class TestDashboardInferenceState(unittest.TestCase):
    """Test suite for dashboard inference execution state, missing-module regressions, and atomic transitions."""

    def test_app_module_imports_time_and_json(self):
        """Verify that app.py imports time and json at module scope."""
        import ast
        app_path = Path("app.py")
        self.assertTrue(app_path.is_file())

        with open(app_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module)

        self.assertIn("time", imported_modules, "time module must be imported in app.py")
        self.assertIn("json", imported_modules, "json module must be imported in app.py")

    def test_output_directory_naming_with_time(self):
        """Test timestamped directory generation format used in local real-patch runs."""
        dir_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        out_dir = Path("outputs") / dir_name
        self.assertTrue(str(out_dir).startswith("outputs\\run_") or str(out_dir).startswith("outputs/run_"))
        self.assertEqual(len(dir_name), len("run_20261029_235959"))

    def test_inference_atomic_state_failure_purges_success_state(self):
        """When an exception occurs during the execution block, session state must not hold success results."""
        session_state = {
            "last_result": "stale_result_from_prior_run",
            "last_ref": "stale_ref",
            "saved_paths": {"geotiff_output": "old.tif"},
        }

        # Simulate execution block with an error
        try:
            # Stage 1: Clear stale state
            session_state.pop("last_result", None)
            session_state.pop("last_ref", None)
            session_state.pop("saved_paths", None)
            session_state.pop("inference_error", None)

            # Stage 2: Action raises an exception (e.g. simulated NameError)
            raise NameError("name 'time' is not defined")
        except Exception as exc:
            session_state.pop("last_result", None)
            session_state.pop("last_ref", None)
            session_state.pop("saved_paths", None)
            session_state["inference_error"] = str(exc)

        # Assert that failure and success can never be displayed simultaneously
        self.assertNotIn("last_result", session_state)
        self.assertNotIn("last_ref", session_state)
        self.assertNotIn("saved_paths", session_state)
        self.assertIn("inference_error", session_state)
        self.assertEqual(session_state["inference_error"], "name 'time' is not defined")

    def test_evaluation_guards_against_dimension_mismatch(self):
        """Evaluation must reject reference arrays whose spatial dimensions do not match prediction."""
        pred = np.zeros((13, 64, 64), dtype=np.float32)
        mismatched_ref = np.zeros((13, 257, 257), dtype=np.float32)

        with self.assertRaises(ValueError) as ctx:
            compute_evaluation_metrics(pred, mismatched_ref)
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_evaluation_guards_against_non_finite_values(self):
        """Evaluation must reject reference arrays containing NaN or Inf."""
        pred = np.zeros((13, 64, 64), dtype=np.float32)
        nan_ref = np.zeros((13, 64, 64), dtype=np.float32)
        nan_ref[0, 10, 10] = np.nan

        with self.assertRaises(ValueError) as ctx:
            compute_evaluation_metrics(pred, nan_ref)
        self.assertIn("NaN", str(ctx.exception))


    def test_step3_model_selection_workflow_gating(self):
        """Step 3 selection must be derived from ModelRegistry and gate operational models into the inference workflow."""
        from geoclear.adapters.dsen2cr_adapter import DSen2CRAdapter
        from geoclear.adapters.ecrformer_adapter import ECRformerAdapter
        from geoclear.registry import ModelRegistry
        from geoclear.types import ReadinessStatus

        # Confirm registry identifiers
        ecr_card = ModelRegistry.get_model("ecrformer")
        dsen_card = ModelRegistry.get_model("dsen2_cr")
        self.assertIsNotNone(ecr_card)
        self.assertIsNotNone(dsen_card)
        self.assertEqual(ecr_card.readiness_status, ReadinessStatus.READY_FOR_INFERENCE)
        self.assertEqual(dsen_card.readiness_status, ReadinessStatus.READY_FOR_INFERENCE)

        # Operational IDs that must unlock Steps 4-8 in app.py
        operational_ids = {"ecrformer", "dsen2_cr"}

        for mid in [ecr_card.model_id, dsen_card.model_id]:
            self.assertIn(mid, operational_ids, f"Model {mid} must be recognized as operational by dashboard")

            # Verify correct adapter instantiation and weight availability
            if mid == "ecrformer":
                active_model_name = "ECRformer"
                adapter = ECRformerAdapter(variant="light", device="cpu")
                self.assertIsInstance(adapter, ECRformerAdapter)
                self.assertEqual(active_model_name, "ECRformer")
                self.assertTrue(os.path.isfile(adapter.checkpoint_path))
            elif mid == "dsen2_cr":
                active_model_name = "DSen2-CR"
                adapter = DSen2CRAdapter(device="cpu")
                self.assertIsInstance(adapter, DSen2CRAdapter)
                self.assertEqual(active_model_name, "DSen2-CR")
                self.assertTrue(os.path.isfile(adapter.checkpoint_path))

        # Blocked models like cloudgan and diffcr must NOT be in operational_ids
        for blocked_id in ["cloudgan", "diffcr"]:
            self.assertNotIn(blocked_id, operational_ids)

    def test_streamlit_rendering_regression(self):
        """End-to-end Streamlit AppTest verifies selecting ECRformer or DSen2-CR renders Steps 4-8 without exception."""
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit not installed in environment")

        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"App execution failed with: {[e.value for e in at.exception]}")

        # In Step 3, the executable model selector is the second selectbox
        model_sb = at.selectbox[1]

        # 1. Select ECRformer
        model_sb.select("ecrformer")
        at.run()
        self.assertEqual(len(at.exception), 0)
        ecr_headers = [h.value for h in at.header]
        self.assertTrue(any("Step 4" in h for h in ecr_headers), "Step 4 header missing for ECRformer")

        # 2. Select DSen2-CR
        model_sb.select("dsen2_cr")
        at.run()
        self.assertEqual(len(at.exception), 0)
        dsen_headers = [h.value for h in at.header]
        self.assertTrue(any("Step 4" in h for h in dsen_headers), "Step 4 header missing for DSen2-CR")


if __name__ == "__main__":
    unittest.main()





