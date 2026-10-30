"""Unit tests for local directory data ingestion, metadata inspection, and output manifests.

Tests:
- Non-destructive metadata inspection (inspect_raster_metadata) without full array reading
- File discovery heuristics (standard names, channel count inferences)
- Missing file and missing band detection
- Channel count mismatch enforcement (e.g. 12 bands vs 13 bands)
- Spatial dimension matching and divisibility by 32
- Coordinate Reference System (CRS) matching and resolution validation
- Finite value validation (NaN / Inf rejection)
- Output artifact generation (GeoTIFF, NPZ, PNGs, manifest, report)
- Source file preservation (original inputs never overwritten)
"""

import io
import json
import os
import shutil
import sys
import tempfile
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

from geoclear.types import S2_BANDS_13
from geoclear.adapters.ecrformer_adapter import ECRformerAdapter, ECRformerInferenceResult
from geoclear.ingestion import (
    LocalPatchFiles,
    PatchValidationReport,
    discover_local_patch,
    inspect_raster_metadata,
    load_local_patch,
    save_inference_outputs,
    validate_local_patch,
    compute_file_sha256,
    parse_manifest_metadata,
)
from geoclear.metrics import compute_evaluation_metrics


class TestLocalIngestion(unittest.TestCase):
    """Test suite for local folder patch ingestion, validation, and output manifests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.folder = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_dummy_geotiff(
        self,
        path: Path,
        channels: int,
        height: int = 64,
        width: int = 64,
        crs: str = "EPSG:32632",
        dtype: str = "float32",
        fill_val: float = 100.0,
        add_nan: bool = False,
    ) -> Path:
        """Helper to write a test GeoTIFF file."""
        data = np.full((channels, height, width), fill_val, dtype=np.float32)
        if add_nan:
            data[0, 5, 5] = np.nan

        transform = from_origin(500000.0, 5000000.0, 10.0, 10.0)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=channels,
            dtype=dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data)
        return path

    def test_inspect_raster_metadata_without_full_read(self):
        """inspect_raster_metadata must extract metadata accurately without loading pixel array."""
        sar_path = self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, height=64, width=64)
        meta = inspect_raster_metadata(sar_path)

        self.assertEqual(meta["channels"], 2)
        self.assertEqual(meta["height"], 64)
        self.assertEqual(meta["width"], 64)
        self.assertIn("32632", str(meta["crs"]))
        self.assertEqual(meta["res"], [10.0, 10.0])
        self.assertEqual(meta["file_name"], "s1_sar.tif")

    def test_discover_local_patch_standard_names(self):
        """discover_local_patch must identify SAR, Optical, Reference, and Manifest."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2)
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13)
        self._write_dummy_geotiff(self.folder / "s2_reference.tif", channels=13)

        manifest_content = {
            "s2_collection": "COPERNICUS/S2_HARMONIZED",
            "s2_sensing_date": "2023-07-15 10:30:00",
            "s1_sensing_date": "2023-07-15 17:15:00",
            "temporal_offset_hours": 6.75,
        }
        with open(self.folder / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_content, f)

        discovered = discover_local_patch(self.folder)

        self.assertIsNotNone(discovered.sar_path)
        self.assertEqual(discovered.sar_path.name, "s1_sar.tif")
        self.assertIsNotNone(discovered.opt_path)
        self.assertEqual(discovered.opt_path.name, "s2_cloudy.tif")
        self.assertIsNotNone(discovered.ref_path)
        self.assertEqual(discovered.ref_path.name, "s2_reference.tif")
        self.assertIsNotNone(discovered.manifest_data)
        self.assertEqual(discovered.manifest_data["temporal_offset_hours"], 6.75)

    def test_validate_local_patch_success(self):
        """validate_local_patch must approve well-formed 2-band SAR and 13-band Optical."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, height=64, width=64)
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13, height=64, width=64)

        discovered = discover_local_patch(self.folder)
        report = validate_local_patch(discovered)

        self.assertTrue(report.is_valid)
        self.assertTrue(report.can_run_inference)
        self.assertEqual(len(report.errors), 0)

    def test_validate_channel_count_mismatch(self):
        """validate_local_patch must reject optical data that lacks 13 bands (e.g. 12-band L2A)."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2)
        # 12 bands instead of 13 bands
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=12)

        discovered = discover_local_patch(self.folder)
        report = validate_local_patch(discovered)

        self.assertFalse(report.is_valid)
        self.assertFalse(report.can_run_inference)
        self.assertTrue(any("13" in err for err in report.errors))

    def test_validate_spatial_dimension_mismatch(self):
        """validate_local_patch must reject mismatched SAR and Optical spatial grids."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, height=64, width=64)
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13, height=128, width=128)

        discovered = discover_local_patch(self.folder)
        report = validate_local_patch(discovered)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("dimension mismatch" in err.lower() for err in report.errors))

    def test_validate_divisibility_by_32(self):
        """validate_local_patch must enforce dimensions divisible by 32."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, height=50, width=50)
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13, height=50, width=50)

        discovered = discover_local_patch(self.folder)
        report = validate_local_patch(discovered)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("divisible by 32" in err.lower() for err in report.errors))

    def test_validate_crs_mismatch(self):
        """validate_local_patch must reject differing coordinate reference systems."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, crs="EPSG:32632")
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13, crs="EPSG:4326")

        discovered = discover_local_patch(self.folder)
        report = validate_local_patch(discovered)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("CRS Mismatch" in err for err in report.errors))

    def test_load_local_patch_non_finite_rejection(self):
        """load_local_patch must reject rasters containing NaNs."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, add_nan=True)
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13)

        discovered = discover_local_patch(self.folder)
        with self.assertRaises(ValueError) as ctx:
            load_local_patch(discovered)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_load_local_patch_and_save_outputs(self):
        """load_local_patch and save_inference_outputs must generate complete provenance package."""
        s1_file = self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, height=64, width=64, fill_val=-12.0)
        s2_file = self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13, height=64, width=64, fill_val=2500.0)
        ref_file = self._write_dummy_geotiff(self.folder / "s2_reference.tif", channels=13, height=64, width=64, fill_val=1800.0)

        discovered = discover_local_patch(self.folder)
        sar_arr, opt_arr, ref_arr, meta = load_local_patch(discovered)

        self.assertEqual(sar_arr.shape, (2, 64, 64))
        self.assertEqual(opt_arr.shape, (13, 64, 64))
        self.assertEqual(ref_arr.shape, (13, 64, 64))

        # Run ECRformer inference
        adapter = ECRformerAdapter(variant="light", device="cpu")
        result = adapter.predict(sar_arr, opt_arr)

        # Compute metrics
        ref_norm = ref_arr / 10000.0
        metrics = compute_evaluation_metrics(result.reconstructed_multispectral, ref_norm)

        # Save outputs
        out_dir = self.folder / "outputs" / "test_run"
        saved = save_inference_outputs(
            output_dir=out_dir,
            result=result,
            patch_metadata=meta,
            patch_files=discovered,
            evaluation_metrics=metrics,
        )

        # Verify saved artifacts
        self.assertIn("npz_output", saved)
        self.assertIn("geotiff_output", saved)
        self.assertIn("rgb_preview", saved)
        self.assertIn("manifest", saved)
        self.assertIn("report", saved)

        self.assertTrue(Path(saved["npz_output"]).is_file())
        self.assertTrue(Path(saved["geotiff_output"]).is_file())
        self.assertTrue(Path(saved["manifest"]).is_file())

        # Verify manifest content
        with open(saved["manifest"], "r", encoding="utf-8") as f:
            manifest_json = json.load(f)

        self.assertEqual(manifest_json["model"]["model_name"], "ECRformer")
        self.assertEqual(manifest_json["model"]["variant"], "light")
        self.assertIn("sar_sha256", manifest_json["inputs"]["checksums"])
        self.assertIn("opt_sha256", manifest_json["inputs"]["checksums"])
        self.assertTrue(manifest_json["evaluation"]["reference_provided"])
        self.assertIsNotNone(manifest_json["evaluation"]["metrics"]["multispectral_mae"])

        # Check that original source files were NOT modified or deleted
        self.assertTrue(s1_file.is_file())
        self.assertTrue(s2_file.is_file())
        self.assertTrue(ref_file.is_file())

    def test_parse_manifest_metadata_v130(self):
        """parse_manifest_metadata must extract nested fields from GEE v1.3.0 manifest schema."""
        sample_manifest = {
            "geoclear_manifest_version": "1.3.0",
            "spatial_grid": {
                "crs": "EPSG:32632",
                "pixel_size_meters": 10,
                "patch_dimensions": [256, 256],
            },
            "sentinel_2_target": {
                "collection": "COPERNICUS/S2_HARMONIZED",
                "scene_id": "20230704T100031_20230704T100438_T32UPU",
                "sensing_date_utc": "2023-07-04 10:04:38",
            },
            "sentinel_1_sar": {
                "collection": "COPERNICUS/S1_GRD",
                "scene_id": "S1A_IW_GRDH_1SDV_20230704T052309_20230704T052334_049269_05ECC7_B5F9",
                "sensing_date_utc": "2023-07-04 05:23:09",
                "temporal_offset_hours": -4.7,
            },
            "reference_optical": {
                "scene_id": "20230624T100031_20230624T100424_T32UPU",
                "sensing_date_utc": "2023-06-24 10:04:24",
                "temporal_difference_days": -10.0,
            },
        }

        parsed = parse_manifest_metadata(sample_manifest)
        self.assertEqual(parsed["s2_scene_id"], "20230704T100031_20230704T100438_T32UPU")
        self.assertEqual(parsed["s2_sensing_date"], "2023-07-04 10:04:38")
        self.assertEqual(parsed["s2_collection"], "COPERNICUS/S2_HARMONIZED")
        self.assertEqual(parsed["s1_scene_id"], "S1A_IW_GRDH_1SDV_20230704T052309_20230704T052334_049269_05ECC7_B5F9")
        self.assertEqual(parsed["s1_sensing_date"], "2023-07-04 05:23:09")
        self.assertAlmostEqual(parsed["temporal_offset_hours"], -4.7)
        self.assertEqual(parsed["ref_scene_id"], "20230624T100031_20230624T100424_T32UPU")
        self.assertEqual(parsed["ref_sensing_date"], "2023-06-24 10:04:24")

    def test_parse_manifest_metadata_missing_leaves_unknown(self):
        """Missing manifest fields must remain None and never be fabricated."""
        empty_manifest = {}
        parsed = parse_manifest_metadata(empty_manifest)
        self.assertIsNone(parsed["s2_scene_id"])
        self.assertIsNone(parsed["s2_sensing_date"])
        self.assertIsNone(parsed["s1_scene_id"])
        self.assertIsNone(parsed["s1_sensing_date"])
        self.assertIsNone(parsed["temporal_offset_hours"])
        self.assertIsNone(parsed["ref_scene_id"])
        self.assertIsNone(parsed["ref_sensing_date"])

        none_manifest = parse_manifest_metadata(None)
        self.assertIsNone(none_manifest["s2_scene_id"])

    def test_real_patch_directory_inspection(self):
        """Inspect actual files in data/real_patch/ if present to verify metadata and error handling."""
        real_patch_dir = Path("data/real_patch")
        if not (real_patch_dir / "s1_sar.tif").exists():
            self.skipTest("data/real_patch not populated")

        discovered = discover_local_patch(real_patch_dir)
        self.assertIsNotNone(discovered.sar_path)
        self.assertIsNotNone(discovered.opt_path)
        self.assertIsNotNone(discovered.ref_path)

        # Inspect raster metadata
        sar_meta = inspect_raster_metadata(discovered.sar_path)
        opt_meta = inspect_raster_metadata(discovered.opt_path)
        ref_meta = inspect_raster_metadata(discovered.ref_path)

        # Confirm exact dimensions on disk (now 256x256 after corrected GEE export)
        self.assertEqual(sar_meta["width"], 256)
        self.assertEqual(sar_meta["height"], 256)
        self.assertEqual(sar_meta["channels"], 2)

        self.assertEqual(opt_meta["width"], 256)
        self.assertEqual(opt_meta["height"], 256)
        self.assertEqual(opt_meta["channels"], 13)

        self.assertEqual(ref_meta["width"], 256)
        self.assertEqual(ref_meta["height"], 256)
        self.assertEqual(ref_meta["channels"], 13)

        # Confirm shared CRS, transform and resolution
        self.assertEqual(sar_meta["crs"], opt_meta["crs"])
        self.assertEqual(sar_meta["transform"], opt_meta["transform"])
        self.assertEqual(sar_meta["res"], opt_meta["res"])
        self.assertEqual(opt_meta["crs"], ref_meta["crs"])
        self.assertEqual(opt_meta["transform"], ref_meta["transform"])

        # Confirm validate_local_patch validates the 256x256 patch
        val_report = validate_local_patch(discovered)
        self.assertTrue(val_report.is_valid)
        self.assertTrue(val_report.can_run_inference)
        self.assertEqual(len(val_report.errors), 0)

        # Confirm manifest parsing populated scene IDs and dates without fabrication
        # Supports both Munich (manifest.json) and Mumbai Dadar (manifest.csv) real patches
        if (real_patch_dir / "manifest.csv").exists():
            self.assertEqual(val_report.s2_scene_id, "20230616T053649_20230616T055224_T43QBB")
            self.assertEqual(val_report.s1_scene_id, "S1A_IW_GRDH_1SDV_20230611T010328_20230611T010353_048931_05E259_98AA")
            self.assertAlmostEqual(val_report.temporal_offset_hours, 124.8)
        else:
            self.assertEqual(val_report.s2_scene_id, "20230712T100601_20230712T101634_T32UPU")
            self.assertEqual(val_report.s1_scene_id, "S1A_IW_GRDH_1SDV_20230710T170747_20230710T170812_049364_05EFA1_04EC")
            self.assertAlmostEqual(val_report.temporal_offset_hours, -41.16)

    def test_discover_and_parse_csv_manifest(self):
        """discover_local_patch and parse_manifest_metadata must support manifest.csv."""
        csv_path = self.folder / "manifest.csv"
        csv_content = (
            "system:index,affine_transform,cloud_mask_applied,cloud_probability_collection,crs,"
            "dataset_name,export_date,grid_x_min,grid_y_max,height_pixels,location,pixel_size_m,"
            "reference_date,reference_gap_days,reference_roi_cloud_percent,reference_system_index,"
            "s1_collection,s2_band_order,s2_collection,s2_data_type,sar_band_order,sar_data_type,"
            "sar_date,sar_speckle_filter_applied,sar_system_index,sar_target_gap_days,target_date,"
            "target_roi_cloud_percent,target_system_index,width_pixels,.geo\n"
            '0,"10,0,271700,0,-10,2105530",false,COPERNICUS/S2_CLOUD_PROBABILITY,EPSG:32643,'
            'Mumbai_Dadar_20230616,2026-10-30T10:18:44.409Z,271700,2105530,256,"Dadar, Mumbai, India",10,'
            '2023-06-06,10,1.85,20230606T053649_20230606T055223_T42QZG,COPERNICUS/S1_GRD,'
            '"B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12",COPERNICUS/S2_HARMONIZED,Original L1C DN,"VV,VH",'
            'Original GRD backscatter in dB,2023-06-11,false,S1A_IW_GRDH_1SDV_20230611T010328_20230611T010353_048931_05E259_98AA,'
            '5.2,2023-06-16,99.58,20230616T053649_20230616T055224_T43QBB,256,"{""type"":""MultiPoint"",""coordinates"":[]}"\n'
        )
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(csv_content)

        discovered = discover_local_patch(self.folder)
        self.assertIsNotNone(discovered.manifest_path)
        self.assertEqual(discovered.manifest_path.name, "manifest.csv")
        self.assertIsNotNone(discovered.manifest_data)
        self.assertEqual(discovered.manifest_data["location"], "Dadar, Mumbai, India")

        parsed = parse_manifest_metadata(discovered.manifest_data)
        self.assertEqual(parsed["s2_scene_id"], "20230616T053649_20230616T055224_T43QBB")
        self.assertEqual(parsed["s2_sensing_date"], "2023-06-16")
        self.assertEqual(parsed["s1_scene_id"], "S1A_IW_GRDH_1SDV_20230611T010328_20230611T010353_048931_05E259_98AA")
        self.assertEqual(parsed["s1_sensing_date"], "2023-06-11")
        self.assertEqual(parsed["ref_scene_id"], "20230606T053649_20230606T055223_T42QZG")
        self.assertEqual(parsed["ref_sensing_date"], "2023-06-06")
        self.assertAlmostEqual(parsed["temporal_offset_hours"], 124.8)
        self.assertEqual(parsed["s2_collection"], "COPERNICUS/S2_HARMONIZED")
        self.assertEqual(parsed["cloud_assessment"]["aoi_mean_cloud_probability_percent"], 99.58)

    def test_validation_rejects_non_divisible_by_32(self):
        """validate_local_patch must reject rasters not divisible by 32 (e.g. 257x257)."""
        self._write_dummy_geotiff(self.folder / "s1_sar.tif", channels=2, height=257, width=257)
        self._write_dummy_geotiff(self.folder / "s2_cloudy.tif", channels=13, height=257, width=257)

        discovered = discover_local_patch(self.folder)
        val_report = validate_local_patch(discovered)
        self.assertFalse(val_report.is_valid)
        self.assertFalse(val_report.can_run_inference)
        self.assertTrue(any("divisible by 32" in err for err in val_report.errors))


if __name__ == "__main__":
    unittest.main()
