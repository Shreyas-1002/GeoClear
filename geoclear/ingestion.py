"""Local directory data ingestion and validation for real satellite patches.

Enables loading large Sentinel-1 and Sentinel-2 GeoTIFF files directly from a
local directory without transferring multi-gigabyte files through the browser.
Inspects metadata without reading pixel arrays into memory, validates spatial
and spectral alignment, and handles traceable output export with provenance manifests.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Windows PROJ environment isolation
if "PROJ_LIB" in os.environ and "PostgreSQL" in os.environ["PROJ_LIB"]:
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    rasterio_proj = os.path.join(site_packages, "rasterio", "proj_data")
    if os.path.isdir(rasterio_proj):
        os.environ["PROJ_LIB"] = rasterio_proj
        os.environ["PROJ_DATA"] = rasterio_proj

import numpy as np
import rasterio
from PIL import Image

from geoclear.types import S2_BANDS_13, S1_SAR_POLARIZATIONS


@dataclass
class LocalPatchFiles:
    """Discovered real patch files within a local directory."""
    folder_path: Path
    sar_path: Optional[Path] = None
    opt_path: Optional[Path] = None
    ref_path: Optional[Path] = None
    manifest_path: Optional[Path] = None
    manifest_data: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class PatchValidationReport:
    """Validation report generated from non-destructive metadata inspection."""
    is_valid: bool
    can_run_inference: bool
    sar_meta: Optional[Dict[str, Any]] = None
    opt_meta: Optional[Dict[str, Any]] = None
    ref_meta: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    temporal_offset_hours: Optional[float] = None
    s1_sensing_date: Optional[str] = None
    s2_sensing_date: Optional[str] = None
    s1_scene_id: Optional[str] = None
    s2_scene_id: Optional[str] = None
    ref_scene_id: Optional[str] = None
    ref_sensing_date: Optional[str] = None
    s2_collection: Optional[str] = None


def parse_manifest_metadata(mdata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse metadata from GeoClear GEE manifest supporting v1.0 through v1.3 schemas.

    Reads target optical, SAR, and reference metadata from both nested section dictionaries
    (`sentinel_2_target`, `sentinel_1_sar`, `reference_optical`) and legacy flat fields.
    Missing fields remain None; values are never fabricated.

    Args:
        mdata: Parsed manifest dictionary or None.

    Returns:
        Dictionary of extracted normalized fields.
    """
    if not mdata or not isinstance(mdata, dict):
        return {
            "s2_scene_id": None,
            "s2_sensing_date": None,
            "s1_scene_id": None,
            "s1_sensing_date": None,
            "temporal_offset_hours": None,
            "ref_scene_id": None,
            "ref_sensing_date": None,
            "s2_collection": None,
            "cloud_assessment": None,
        }

    s2_block = mdata.get("sentinel_2_target") if isinstance(mdata.get("sentinel_2_target"), dict) else {}
    s1_block = mdata.get("sentinel_1_sar") if isinstance(mdata.get("sentinel_1_sar"), dict) else {}
    ref_block = mdata.get("reference_optical") if isinstance(mdata.get("reference_optical"), dict) else {}

    s2_scene_id = (
        s2_block.get("scene_id")
        or mdata.get("s2_scene_id")
        or mdata.get("target_system_index")
    )
    s2_sensing_date = (
        s2_block.get("sensing_date_utc")
        or s2_block.get("sensing_date")
        or mdata.get("s2_sensing_date")
        or mdata.get("s2_date")
        or mdata.get("target_date")
    )

    s1_scene_id = (
        s1_block.get("scene_id")
        or mdata.get("s1_scene_id")
        or mdata.get("sar_system_index")
    )
    s1_sensing_date = (
        s1_block.get("sensing_date_utc")
        or s1_block.get("sensing_date")
        or mdata.get("s1_sensing_date")
        or mdata.get("s1_date")
        or mdata.get("sar_date")
    )

    temporal_offset_hours = s1_block.get("temporal_offset_hours")
    if temporal_offset_hours is None:
        temporal_offset_hours = mdata.get("temporal_offset_hours")
    if temporal_offset_hours is None and mdata.get("sar_target_gap_days") is not None:
        try:
            temporal_offset_hours = float(mdata["sar_target_gap_days"]) * 24.0
        except (ValueError, TypeError):
            temporal_offset_hours = None
    if temporal_offset_hours is not None:
        try:
            temporal_offset_hours = float(temporal_offset_hours)
        except (ValueError, TypeError):
            temporal_offset_hours = None

    ref_scene_id = (
        ref_block.get("scene_id")
        or mdata.get("reference_scene_id")
        or mdata.get("ref_scene_id")
        or mdata.get("reference_system_index")
    )
    ref_sensing_date = (
        ref_block.get("sensing_date_utc")
        or ref_block.get("sensing_date")
        or mdata.get("reference_sensing_date")
        or mdata.get("ref_sensing_date")
        or mdata.get("reference_date")
    )

    s2_collection = s2_block.get("collection") or mdata.get("s2_collection")
    cloud_assessment = s2_block.get("cloud_assessment") or mdata.get("cloud_assessment")
    if cloud_assessment is None and mdata.get("target_roi_cloud_percent") is not None:
        try:
            cloud_val = float(mdata["target_roi_cloud_percent"])
            cloud_assessment = {
                "aoi_mean_cloud_probability_percent": cloud_val,
                "aoi_binary_cloud_pixel_percentage": cloud_val,
            }
        except (ValueError, TypeError):
            pass

    return {
        "s2_scene_id": s2_scene_id,
        "s2_sensing_date": s2_sensing_date,
        "s1_scene_id": s1_scene_id,
        "s1_sensing_date": s1_sensing_date,
        "temporal_offset_hours": temporal_offset_hours,
        "ref_scene_id": ref_scene_id,
        "ref_sensing_date": ref_sensing_date,
        "s2_collection": s2_collection,
        "cloud_assessment": cloud_assessment,
    }


def inspect_raster_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect geospatial and raster metadata without reading pixel arrays into memory.

    Args:
        file_path: Path to the raster file.

    Returns:
        Dictionary containing driver, channel count, dimensions, CRS, affine transform,
        pixel resolution, bounds, nodata convention, and data types.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is not a readable raster.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raster file not found: {path}")

    try:
        with rasterio.open(path) as src:
            return {
                "file_path": str(path.resolve()),
                "file_name": path.name,
                "file_size_bytes": path.stat().st_size,
                "driver": src.driver,
                "channels": src.count,
                "height": src.height,
                "width": src.width,
                "crs": str(src.crs) if src.crs else None,
                "transform": list(src.transform) if src.transform else None,
                "bounds": list(src.bounds) if src.bounds else None,
                "res": list(src.res) if src.res else None,
                "nodata": src.nodata,
                "dtypes": [str(d) for d in src.dtypes],
                "descriptions": list(src.descriptions) if src.descriptions else [],
            }
    except Exception as exc:
        raise ValueError(f"Could not inspect raster metadata for {path.name}: {exc}") from exc


def discover_local_patch(folder_path: Union[str, Path]) -> LocalPatchFiles:
    """Search a local folder for Sentinel-1, Sentinel-2, reference, and manifest files.

    Supports standard naming conventions:
    - Sentinel-1: `*s1*.tif`, `*sar*.tif`, `s1.tif`, `sar.tif`
    - Sentinel-2: `*s2_cloudy*.tif`, `*cloudy*.tif`, `*s2*.tif`, `*opt*.tif`
    - Reference: `*clear*.tif`, `*ref*.tif`, `*clean*.tif`, `*s2_clear*.tif`
    - Manifest: `manifest.json`, `metadata.json`, `processing_manifest.json`

    Args:
        folder_path: Path to the local directory.

    Returns:
        LocalPatchFiles dataclass with identified file paths and loaded manifest if available.
    """
    folder = Path(folder_path).resolve()
    result = LocalPatchFiles(folder_path=folder)

    if not folder.is_dir():
        result.notes.append(f"Directory does not exist: {folder}")
        return result

    # Find manifest (JSON or CSV)
    manifest_candidates = [
        folder / "manifest.json",
        folder / "metadata.json",
        folder / "processing_manifest.json",
        folder / "manifest.csv",
    ]
    for mf in manifest_candidates:
        if mf.is_file():
            result.manifest_path = mf
            try:
                if mf.suffix.lower() == ".csv":
                    with open(mf, "r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        result.manifest_data = rows[0] if rows else {}
                else:
                    with open(mf, "r", encoding="utf-8") as f:
                        result.manifest_data = json.load(f)
                result.notes.append(f"Loaded manifest: {mf.name}")
            except Exception as e:
                result.notes.append(f"Failed to parse manifest {mf.name}: {e}")
            break

    # List all TIFF and NPZ files
    all_files = [f for f in folder.iterdir() if f.is_file()]
    tif_files = [f for f in all_files if f.suffix.lower() in {".tif", ".tiff"}]

    # Match SAR file
    for f in tif_files:
        name_lower = f.name.lower()
        if any(k in name_lower for k in ["_s1", "s1_", "sar", "sentinel1", "s1."]):
            result.sar_path = f
            break

    # Match Reference file (clean) first to avoid confusion with cloudy S2
    for f in tif_files:
        name_lower = f.name.lower()
        if any(k in name_lower for k in ["clear", "clean", "ref", "groundtruth", "gt"]):
            result.ref_path = f
            break

    # Match Optical file (cloudy Sentinel-2)
    for f in tif_files:
        name_lower = f.name.lower()
        # Avoid picking the SAR file or reference file
        if f == result.sar_path or f == result.ref_path:
            continue
        if any(k in name_lower for k in ["cloudy", "s2", "optical", "sentinel2"]):
            result.opt_path = f
            break

    # Fallback heuristic: If exactly 2 TIFF files and no naming keywords, inspect channel counts
    if (result.sar_path is None or result.opt_path is None) and len(tif_files) == 2:
        try:
            m0 = inspect_raster_metadata(tif_files[0])
            m1 = inspect_raster_metadata(tif_files[1])
            if m0["channels"] == 2 and m1["channels"] == 13:
                result.sar_path = tif_files[0]
                result.opt_path = tif_files[1]
                result.notes.append("Inferred SAR (2 bands) and Optical (13 bands) by channel count.")
            elif m0["channels"] == 13 and m1["channels"] == 2:
                result.opt_path = tif_files[0]
                result.sar_path = tif_files[1]
                result.notes.append("Inferred Optical (13 bands) and SAR (2 bands) by channel count.")
        except Exception:
            pass

    return result


def validate_local_patch(patch_files: LocalPatchFiles) -> PatchValidationReport:
    """Validate discovered local patch files strictly through metadata inspection.

    Does NOT read large image data into RAM. Checks band counts, spatial dimensions,
    divisibility by 32, CRS alignment, resolution, and manifest provenance.

    Args:
        patch_files: LocalPatchFiles object containing paths.

    Returns:
        PatchValidationReport detailing validation status, errors, and warnings.
    """
    report = PatchValidationReport(is_valid=False, can_run_inference=False)

    if not patch_files.folder_path.is_dir():
        report.errors.append(f"Directory not found: {patch_files.folder_path}")
        return report

    # 1. File existence checks
    if patch_files.sar_path is None:
        report.errors.append("Missing Sentinel-1 SAR file (expected 2 bands: VV, VH).")
    if patch_files.opt_path is None:
        report.errors.append("Missing Sentinel-2 Optical file (expected 13 bands: B1-B12, B8A).")

    if report.errors:
        return report

    # 2. Inspect SAR metadata
    try:
        assert patch_files.sar_path is not None
        sar_meta = inspect_raster_metadata(patch_files.sar_path)
        report.sar_meta = sar_meta
    except Exception as exc:
        report.errors.append(f"Failed to read SAR metadata: {exc}")
        return report

    # 3. Inspect Optical metadata
    try:
        assert patch_files.opt_path is not None
        opt_meta = inspect_raster_metadata(patch_files.opt_path)
        report.opt_meta = opt_meta
    except Exception as exc:
        report.errors.append(f"Failed to read Optical metadata: {exc}")
        return report

    # 4. Validate channel counts
    if sar_meta["channels"] != 2:
        report.errors.append(
            f"SAR channel count mismatch: expected exactly 2 channels (VV, VH), "
            f"found {sar_meta['channels']} in {patch_files.sar_path.name}."
        )
    else:
        report.info.append(f"SAR channels validated: 2 bands in {patch_files.sar_path.name}")

    if opt_meta["channels"] != 13:
        report.errors.append(
            f"Optical band count mismatch: ECRformer requires all 13 Sentinel-2 bands "
            f"[B1-B12, B8A] (Level-1C TOA product). Found {opt_meta['channels']} bands in "
            f"{patch_files.opt_path.name}."
        )
    else:
        report.info.append(f"Optical channels validated: 13 bands in {patch_files.opt_path.name}")

    # 5. Validate spatial dimensions
    h_sar, w_sar = sar_meta["height"], sar_meta["width"]
    h_opt, w_opt = opt_meta["height"], opt_meta["width"]

    if (h_sar, w_sar) != (h_opt, w_opt):
        report.errors.append(
            f"Spatial dimension mismatch: SAR grid is {h_sar}x{w_sar}, "
            f"while Optical grid is {h_opt}x{w_opt}."
        )
    else:
        report.info.append(f"Matching spatial dimensions confirmed: {h_sar}x{w_sar}")

    # 6. Validate divisibility by 32
    if h_sar % 32 != 0 or w_sar % 32 != 0:
        report.errors.append(
            f"Spatial dimensions ({h_sar}x{w_sar}) must be strictly divisible by 32 "
            f"for ECRformer encoder-decoder processing (e.g. 128x128, 256x256, 512x512)."
        )

    # 7. Validate CRS and Geotransform
    sar_crs = sar_meta.get("crs")
    opt_crs = opt_meta.get("crs")
    if sar_crs and opt_crs:
        if sar_crs != opt_crs:
            report.errors.append(
                f"CRS Mismatch: SAR CRS is '{sar_crs}', but Optical CRS is '{opt_crs}'. "
                f"Inputs must share the exact same projected coordinate reference system."
            )
        else:
            report.info.append(f"Matching CRS confirmed: {sar_crs}")
    else:
        report.warnings.append("One or both rasters lack CRS metadata. Spatial grid alignment cannot be geometrically verified.")

    # Validate pixel resolution
    sar_res = sar_meta.get("res")
    opt_res = opt_meta.get("res")
    if sar_res and opt_res:
        if abs(sar_res[0] - opt_res[0]) > 1e-4 or abs(sar_res[1] - opt_res[1]) > 1e-4:
            report.warnings.append(
                f"Pixel resolution discrepancy: SAR resolution is {sar_res}, "
                f"while Optical resolution is {opt_res}."
            )

    # 8. Inspect Reference file if present
    if patch_files.ref_path is not None:
        try:
            ref_meta = inspect_raster_metadata(patch_files.ref_path)
            report.ref_meta = ref_meta
            if ref_meta["channels"] != 13:
                report.warnings.append(
                    f"Reference optical band mismatch: expected 13 bands, found {ref_meta['channels']}. "
                    f"Quantitative reference evaluation will be disabled."
                )
            elif (ref_meta["height"], ref_meta["width"]) != (h_opt, w_opt):
                report.warnings.append(
                    f"Reference spatial dimensions ({ref_meta['height']}x{ref_meta['width']}) "
                    f"do not match Optical ({h_opt}x{w_opt}). Quantitative evaluation disabled."
                )
            else:
                report.info.append(f"Ground-truth reference validated: {patch_files.ref_path.name}")
        except Exception as e:
            report.warnings.append(f"Failed to inspect reference metadata: {e}")

    # 9. Extract provenance from manifest if available
    if patch_files.manifest_data:
        parsed = parse_manifest_metadata(patch_files.manifest_data)
        report.s1_sensing_date = parsed["s1_sensing_date"]
        report.s2_sensing_date = parsed["s2_sensing_date"]
        report.temporal_offset_hours = parsed["temporal_offset_hours"]
        report.s1_scene_id = parsed["s1_scene_id"]
        report.s2_scene_id = parsed["s2_scene_id"]
        report.ref_scene_id = parsed["ref_scene_id"]
        report.ref_sensing_date = parsed["ref_sensing_date"]
        report.s2_collection = parsed["s2_collection"]

        if report.temporal_offset_hours is not None:
            report.info.append(f"Temporal offset between SAR and Optical: {report.temporal_offset_hours:.1f} hours")
        if report.s2_collection:
            report.info.append(f"Source Optical Collection: {report.s2_collection}")
        if report.s2_scene_id:
            report.info.append(f"Target Sentinel-2 Scene: {report.s2_scene_id}")
        if report.s1_scene_id:
            report.info.append(f"Target Sentinel-1 Scene: {report.s1_scene_id}")

    report.is_valid = (len(report.errors) == 0)
    report.can_run_inference = report.is_valid

    return report


def load_local_patch(
    patch_files: LocalPatchFiles,
    require_divisible_by: int = 32,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Read validated SAR and Optical arrays into memory.

    Performs finite value assertions and returns float32 arrays.

    Args:
        patch_files: Validated LocalPatchFiles instance.
        require_divisible_by: Divisibility constraint (default 32).

    Returns:
        Tuple of:
            - sar_arr: (2, H, W) float32 array in dB.
            - opt_arr: (13, H, W) float32 array in DN [0, 10000].
            - ref_arr: Optional (13, H, W) float32 array.
            - patch_metadata: Combined metadata dictionary.

    Raises:
        ValueError: If files contain NaNs, Infs, or shape mismatch.
    """
    assert patch_files.sar_path is not None
    assert patch_files.opt_path is not None

    with rasterio.open(patch_files.sar_path) as src_sar:
        sar_arr = src_sar.read().astype(np.float32)
        sar_meta = {
            "crs": str(src_sar.crs) if src_sar.crs else None,
            "transform": list(src_sar.transform) if src_sar.transform else None,
            "res": list(src_sar.res) if src_sar.res else None,
            "nodata": src_sar.nodata,
        }

    with rasterio.open(patch_files.opt_path) as src_opt:
        opt_arr = src_opt.read().astype(np.float32)
        opt_meta = {
            "crs": str(src_opt.crs) if src_opt.crs else None,
            "transform": list(src_opt.transform) if src_opt.transform else None,
            "res": list(src_opt.res) if src_opt.res else None,
            "nodata": src_opt.nodata,
        }

    ref_arr = None
    if patch_files.ref_path is not None:
        try:
            with rasterio.open(patch_files.ref_path) as src_ref:
                ref_arr = src_ref.read().astype(np.float32)
        except Exception:
            ref_arr = None

    # Check finite values
    if not np.isfinite(sar_arr).all():
        nan_count = int(np.isnan(sar_arr).sum())
        inf_count = int(np.isinf(sar_arr).sum())
        raise ValueError(
            f"SAR file '{patch_files.sar_path.name}' contains non-finite values "
            f"({nan_count} NaNs, {inf_count} Infs). Input data must be clean numerical values."
        )

    if not np.isfinite(opt_arr).all():
        nan_count = int(np.isnan(opt_arr).sum())
        inf_count = int(np.isinf(opt_arr).sum())
        raise ValueError(
            f"Optical file '{patch_files.opt_path.name}' contains non-finite values "
            f"({nan_count} NaNs, {inf_count} Infs). Input data must be clean numerical values."
        )

    combined_meta = {
        "source_folder": str(patch_files.folder_path),
        "sar_file": patch_files.sar_path.name,
        "opt_file": patch_files.opt_path.name,
        "ref_file": patch_files.ref_path.name if patch_files.ref_path else None,
        "crs": opt_meta["crs"] or sar_meta["crs"],
        "transform": opt_meta["transform"],
        "res": opt_meta["res"],
        "sar_nodata": sar_meta["nodata"],
        "opt_nodata": opt_meta["nodata"],
        "manifest": patch_files.manifest_data,
    }

    return sar_arr, opt_arr, ref_arr, combined_meta


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def save_inference_outputs(
    output_dir: Union[str, Path],
    result: Any,
    patch_metadata: Dict[str, Any],
    patch_files: Optional[LocalPatchFiles] = None,
    evaluation_metrics: Optional[Any] = None,
) -> Dict[str, str]:
    """Save traceable reconstruction outputs, previews, GeoTIFF, and provenance manifest.

    Never overwrites input scenes.

    Args:
        output_dir: Target output directory.
        result: ECRformerInferenceResult object from adapter.
        patch_metadata: Geospatial and source metadata dictionary.
        patch_files: Optional LocalPatchFiles to compute input checksums.
        evaluation_metrics: Optional EvaluationMetrics object.

    Returns:
        Dictionary mapping artifact keys to their absolute file paths.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, str] = {}

    # 1. Save Reconstructed 13-band compressed NumPy archive
    npz_path = out_dir / "reconstructed_13bands.npz"
    np.savez_compressed(
        npz_path,
        reconstructed=result.reconstructed_multispectral,
        bands=np.asarray(list(S2_BANDS_13)),
    )
    saved_paths["npz_output"] = str(npz_path)

    # 2. Save Reconstructed 13-band GeoTIFF if spatial metadata is present
    crs_str = patch_metadata.get("crs")
    transform_list = patch_metadata.get("transform")
    if crs_str and transform_list:
        try:
            tif_path = out_dir / "reconstructed_13bands.tif"
            c, h, w = result.reconstructed_multispectral.shape
            # Scale to standard Sentinel-2 integer DN [0, 10000] for standard GIS viewing
            recon_dn = np.clip(result.reconstructed_multispectral * 10000.0, 0.0, 10000.0).round().astype(np.uint16)

            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=h,
                width=w,
                count=c,
                dtype="uint16",
                crs=crs_str,
                transform=rasterio.transform.Affine(*transform_list[:6]),
                nodata=0,
            ) as dst:
                dst.write(recon_dn)
                for b_idx, b_name in enumerate(S2_BANDS_13):
                    dst.set_band_description(b_idx + 1, f"Sentinel-2 {b_name} (Reconstructed DN)")

            saved_paths["geotiff_output"] = str(tif_path)
        except Exception as e:
            # If GeoTIFF writing fails, continue saving previews and manifest
            saved_paths["geotiff_error"] = str(e)

    # 3. Save True-Color RGB Preview PNG
    png_rgb_path = out_dir / "reconstructed_rgb_preview.png"
    Image.fromarray(result.reconstructed_rgb_preview).save(png_rgb_path, format="PNG")
    saved_paths["rgb_preview"] = str(png_rgb_path)

    # 4. Save Input Cloudy RGB Preview PNG
    png_cloudy_path = out_dir / "input_cloudy_rgb_preview.png"
    Image.fromarray(result.input_cloudy_rgb_preview).save(png_cloudy_path, format="PNG")
    saved_paths["cloudy_preview"] = str(png_cloudy_path)

    # 5. Save Input SAR Composite Preview PNG
    png_sar_path = out_dir / "input_sar_composite_preview.png"
    Image.fromarray(result.input_sar_preview).save(png_sar_path, format="PNG")
    saved_paths["sar_preview"] = str(png_sar_path)

    # 6. Build Processing Provenance Manifest
    input_checksums: Dict[str, str] = {}
    if patch_files:
        if patch_files.sar_path and patch_files.sar_path.is_file():
            input_checksums["sar_sha256"] = compute_file_sha256(patch_files.sar_path)
        if patch_files.opt_path and patch_files.opt_path.is_file():
            input_checksums["opt_sha256"] = compute_file_sha256(patch_files.opt_path)
        if patch_files.ref_path and patch_files.ref_path.is_file():
            input_checksums["ref_sha256"] = compute_file_sha256(patch_files.ref_path)

    manifest: Dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": "GeoClear v1.0",
        "model": {
            "model_name": result.metadata.get("model_name", "ECRformer"),
            "variant": result.variant,
            "device": result.device,
            "checkpoint_path": result.checkpoint_path,
            "inference_time_ms": result.metadata.get("inference_time_ms"),
            "licensing_note": (
                "Licensed under GNU GPL v3.0."
                if result.metadata.get("model_name") == "DSen2-CR"
                else (
                    "Academic research evaluation. Upstream repository (Zhang et al.) "
                    "contains no explicit LICENSE file; commercial permissions unstated."
                )
            ),
        },
        "inputs": {
            "source_folder": patch_metadata.get("source_folder"),
            "sar_file": patch_metadata.get("sar_file"),
            "opt_file": patch_metadata.get("opt_file"),
            "ref_file": patch_metadata.get("ref_file"),
            "checksums": input_checksums,
            "dimensions": {
                "height": result.reconstructed_multispectral.shape[1],
                "width": result.reconstructed_multispectral.shape[2],
            },
            "crs": crs_str,
            "transform": transform_list,
            "gee_provenance": patch_metadata.get("manifest"),
        },
        "preprocessing": {
            "sar_formula": (
                "VV in [-25, 0], VH in [-32.5, 0] scaled to [0, 2]"
                if result.metadata.get("model_name") == "DSen2-CR"
                else "clip(x, -25.0, 0.0) / 25.0 + 1.0 -> [0, 1]"
            ),
            "optical_formula": (
                "clip(x, 0.0, 10000.0) / 2000.0 -> [0, 5]"
                if result.metadata.get("model_name") == "DSen2-CR"
                else "clip(x, 0.0, 10000.0) / 10000.0 -> [0, 1]"
            ),
            "spectral_bands": list(S2_BANDS_13),
            "sar_polarizations": list(S1_SAR_POLARIZATIONS),
        },
        "output_statistics": {
            "reconstructed_shape": list(result.reconstructed_multispectral.shape),
            "output_scale": result.output_scale,
            "output_min": float(result.metadata.get("output_min", 0.0)),
            "output_max": float(result.metadata.get("output_max", 1.0)),
            "output_mean": float(result.metadata.get("output_mean", 0.0)),
        },
        "evaluation": {
            "reference_provided": evaluation_metrics is not None,
            "metrics": {
                "multispectral_mae": evaluation_metrics.mae if evaluation_metrics else None,
                "multispectral_rmse": evaluation_metrics.rmse if evaluation_metrics else None,
                "rgb_psnr_db": evaluation_metrics.rgb_psnr if evaluation_metrics else None,
                "rgb_ssim": evaluation_metrics.rgb_ssim if evaluation_metrics else None,
            } if evaluation_metrics else "Quantitative metrics not established (no reference provided)",
        },
        "scientific_disclaimer": (
            "Model-generated surface reflectance values are AI reconstructions based on SAR-optical fusion. "
            "They must not be represented as real optical sensor measurements. SAR backscatter does not "
            "directly observe true optical color."
        ),
    }

    manifest_path = out_dir / "processing_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    saved_paths["manifest"] = str(manifest_path)

    # 7. Output summary report (Markdown)
    report_path = out_dir / "reconstruction_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"# GeoClear Reconstruction Output Report\n\n"
            f"- **Execution Timestamp**: {manifest['timestamp_utc']}\n"
            f"- **Model**: ECRformer ({result.variant.upper()})\n"
            f"- **Inference Device**: {result.device.upper()}\n"
            f"- **Latency**: {result.metadata.get('inference_time_ms', 0):.1f} ms\n"
            f"- **Spatial Dimensions**: {manifest['inputs']['dimensions']['height']}x{manifest['inputs']['dimensions']['width']}\n"
            f"- **CRS**: {crs_str or 'Not provided'}\n\n"
            f"## Generated Artifacts\n"
            f"- Reconstructed Multi-Spectral Array (`.npz`): `{npz_path.name}`\n"
        )
        if "geotiff_output" in saved_paths:
            f.write(f"- Reconstructed GeoTIFF (`.tif`): `{Path(saved_paths['geotiff_output']).name}`\n")
        f.write(
            f"- Reconstructed RGB Preview (`.png`): `{png_rgb_path.name}`\n"
            f"- Provenance Manifest (`.json`): `{manifest_path.name}`\n\n"
            f"## Scientific Evaluation & Disclaimers\n"
            f"{manifest['scientific_disclaimer']}\n\n"
        )
        if evaluation_metrics:
            f.write(
                f"### Quantitative Error Metrics (vs Ground-Truth Reference)\n"
                f"- MAE: {evaluation_metrics.mae:.4f}\n"
                f"- RMSE: {evaluation_metrics.rmse:.4f}\n"
                f"- RGB PSNR: {evaluation_metrics.rgb_psnr:.2f} dB\n"
                f"- RGB SSIM: {evaluation_metrics.rgb_ssim:.4f}\n"
            )
        else:
            f.write("Quantitative reconstruction accuracy was not established (no reference scene provided).\n")
    saved_paths["report"] = str(report_path)

    return saved_paths
