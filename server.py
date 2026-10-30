"""GeoClear Web Application Server.

FastAPI-powered modern web server providing REST APIs and serving a high-performance
HTML/CSS/JS frontend for multimodal satellite cloud removal.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Windows PROJ environment isolation
if "PROJ_LIB" in os.environ and "PostgreSQL" in os.environ["PROJ_LIB"]:
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    rasterio_proj = os.path.join(site_packages, "rasterio", "proj_data")
    if os.path.isdir(rasterio_proj):
        os.environ["PROJ_LIB"] = rasterio_proj
        os.environ["PROJ_DATA"] = rasterio_proj

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from geoclear.types import (
    DeclaredData,
    Modality,
    ModelCard,
    ReadinessStatus,
    SensorType,
    S2_BANDS_13,
    RGB_BANDS,
    S1_SAR_POLARIZATIONS,
)
from geoclear.registry import ModelRegistry
from geoclear.compatibility import CompatibilityEngine
from geoclear.adapters.ecrformer_adapter import ECRformerAdapter
from geoclear.adapters.dsen2cr_adapter import DSen2CRAdapter
from geoclear.io import (
    read_raster_file,
    validate_spatial_alignment,
    export_npz_bytes,
    export_png_bytes,
)
from geoclear.metrics import compute_evaluation_metrics, EvaluationMetrics
from geoclear.demo_data import generate_synthetic_verification_patch
from geoclear.ingestion import (
    LocalPatchFiles,
    PatchValidationReport,
    discover_local_patch,
    inspect_raster_metadata,
    load_local_patch,
    save_inference_outputs,
    validate_local_patch,
    parse_manifest_metadata,
)

app = FastAPI(
    title="GeoClear — Earth Observation AI Platform",
    description="Multimodal Earth-Observation Satellite Cloud Removal Interface",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global In-Memory Working State
# ---------------------------------------------------------------------------

class WorkingState:
    def __init__(self):
        self.sar_arr: Optional[np.ndarray] = None
        self.opt_arr: Optional[np.ndarray] = None
        self.ref_arr: Optional[np.ndarray] = None
        self.patch_meta: Optional[Dict[str, Any]] = None
        self.patch_files: Optional[LocalPatchFiles] = None
        self.data_source_mode: str = "none"  # "real_local", "synthetic", "upload", "benchmark"
        self.selected_model_id: str = "ecrformer"
        self.last_result: Optional[Any] = None
        self.last_metrics: Optional[EvaluationMetrics] = None
        self.saved_paths: Optional[Dict[str, str]] = None
        self.dataset_title: str = "No Data Loaded"


state = WorkingState()


# ---------------------------------------------------------------------------
# Helper Functions: Image Encoding
# ---------------------------------------------------------------------------

def array_to_base64_png(arr_rgb_uint8: np.ndarray) -> str:
    """Encode an RGB uint8 array (H, W, 3) to a base64 PNG data URL."""
    img = Image.fromarray(arr_rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def array_greyscale_to_base64_png(arr_2d: np.ndarray) -> str:
    """Encode a 2D float array (H, W) to normalized greyscale base64 PNG data URL."""
    c_min, c_max = float(arr_2d.min()), float(arr_2d.max())
    denom = c_max - c_min if (c_max - c_min) > 1e-8 else 1.0
    norm = np.clip((arr_2d - c_min) / denom, 0.0, 1.0)
    uint8_img = (norm * 255.0).astype(np.uint8)
    img = Image.fromarray(uint8_img, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def get_optical_preview(opt_data: np.ndarray) -> str:
    """Generate 3x-brightness RGB preview for 13-band Sentinel-2 array."""
    # B4=3, B3=2, B2=1
    rgb = opt_data[[3, 2, 1], :, :].transpose(1, 2, 0).copy()
    if rgb.max() > 1.5:
        rgb = np.clip(rgb / 10000.0, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    stretched = np.clip(rgb * 3.0, 0.0, 1.0)
    uint8_rgb = (stretched * 255.0).astype(np.uint8)
    return array_to_base64_png(uint8_rgb)


def get_sar_preview(sar_data: np.ndarray) -> str:
    """Generate false-color RGB preview for 2-band Sentinel-1 SAR (VV, VH in dB)."""
    vv = sar_data[0].copy()
    vh = sar_data[1].copy()
    vv_norm = np.clip((vv - (-25.0)) / 25.0, 0.0, 1.0)
    vh_norm = np.clip((vh - (-32.5)) / 32.5, 0.0, 1.0)
    ratio = np.clip((vv - vh) / 15.0, 0.0, 1.0)
    composite = np.stack([vv_norm, vh_norm, ratio], axis=-1)
    uint8_sar = (composite * 255.0).astype(np.uint8)
    return array_to_base64_png(uint8_sar)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_system_status():
    """Return platform status and currently active data / model state."""
    has_sar = state.sar_arr is not None
    has_opt = state.opt_arr is not None
    has_ref = state.ref_arr is not None

    shape = None
    if has_opt:
        shape = [int(x) for x in state.opt_arr.shape]

    return {
        "status": "ready",
        "data_source_mode": state.data_source_mode,
        "dataset_title": state.dataset_title,
        "has_sar": has_sar,
        "has_opt": has_opt,
        "has_ref": has_ref,
        "dimensions": shape,
        "selected_model_id": state.selected_model_id,
        "has_result": state.last_result is not None,
    }


@app.get("/api/models")
def get_available_models():
    """Return the operational models registered in GeoClear."""
    models_data = []
    for card in ModelRegistry.list_models():
        models_data.append({
            "model_id": card.model_id,
            "display_name": card.display_name,
            "description": card.description,
            "paper_citation": card.paper_citation,
            "paper_url": card.paper_url,
            "framework": card.framework,
            "framework_version": card.framework_version,
            "readiness_status": card.readiness_status.value,
            "verification_status": card.verification_status.value,
            "checkpoint_present": card.checkpoint_present_locally,
            "channels": card.input_requirements.expected_channels,
            "modalities": [m.value for m in card.input_requirements.required_modalities],
            "sensors": [s.value for s in card.input_requirements.required_sensors],
            "parameters": "~3.70M" if card.model_id == "ecrformer" else "~18.87M",
            "latency": "~3.9s CPU" if card.model_id == "ecrformer" else "~11.5s CPU",
            "family": "Vision Transformer" if card.model_id == "ecrformer" else "Deep ResNet (CARL)",
        })
    return {"models": models_data, "selected": state.selected_model_id}


@app.post("/api/models/select")
def select_model(model_id: str = Form(...)):
    """Set the active operational model for inference."""
    if model_id not in {"ecrformer", "dsen2_cr"}:
        raise HTTPException(status_code=400, detail=f"Invalid model ID '{model_id}'. Must be 'ecrformer' or 'dsen2_cr'.")
    state.selected_model_id = model_id
    return {"status": "success", "selected_model_id": model_id}


@app.post("/api/declare")
def evaluate_data_declaration(
    has_s1: bool = Form(True),
    has_s2: bool = Form(True),
    optical_bands: str = Form("13"),  # "13" or "3"
    sar_vv: bool = Form(True),
    sar_vh: bool = Form(True),
    temporal_count: int = Form(1),
    has_reference: bool = Form(False),
    height: int = Form(256),
    width: int = Form(256),
):
    """Run strict compatibility check based on declared physics and sensors."""
    modalities = set()
    sensors = set()

    if has_s1:
        sensors.add(SensorType.SENTINEL_1)
    if has_s2:
        sensors.add(SensorType.SENTINEL_2)

    if optical_bands == "13":
        b_set = set(S2_BANDS_13)
        modalities.add(Modality.OPTICAL)
        modalities.add(Modality.MULTISPECTRAL)
    else:
        b_set = {"B4", "B3", "B2"}
        modalities.add(Modality.OPTICAL)

    pols = set()
    if has_s1:
        if sar_vv:
            pols.add("VV")
        if sar_vh:
            pols.add("VH")
        if pols:
            modalities.add(Modality.SAR)

    if temporal_count >= 2:
        modalities.add(Modality.MULTI_TEMPORAL)

    declared = DeclaredData(
        modalities=modalities,
        sensors=sensors,
        optical_bands=b_set,
        sar_polarizations=pols,
        temporal_observations=temporal_count,
        has_clear_optical_reference=has_reference,
        has_ground_truth_reference=has_reference,
        image_height=height,
        image_width=width,
    )

    report = CompatibilityEngine.evaluate(declared)

    compat_list = []
    for res in report.compatible_models:
        card = ModelRegistry.get_model(res.model_id)
        compat_list.append({
            "model_id": res.model_id,
            "display_name": card.display_name,
            "is_compatible": res.is_compatible,
            "is_executable": res.is_executable,
            "description": card.description,
            "blockers": res.execution_blockers,
        })

    incompat_list = []
    for res in report.incompatible_models:
        card = ModelRegistry.get_model(res.model_id)
        incompat_list.append({
            "model_id": res.model_id,
            "display_name": card.display_name if card else res.model_name,
            "reasons": res.exclusion_reasons,
            "missing_inputs": res.missing_inputs,
        })

    return {
        "status": "success",
        "compatible_count": len(report.compatible_models),
        "incompatible_count": len(report.incompatible_models),
        "compatible": compat_list,
        "incompatible": incompat_list,
    }


@app.post("/api/ingest/inspect_local")
def inspect_local_folder(folder_path: str = Form("data/real_patch")):
    """Non-destructively inspect files, metadata, and manifest in a local directory."""
    dir_p = Path(folder_path)
    if not dir_p.is_dir():
        return {
            "status": "error",
            "message": f"Directory '{folder_path}' does not exist on disk.",
            "is_valid": False,
        }

    discovered = discover_local_patch(dir_p)
    val_report = validate_local_patch(discovered)

    manifest_info = None
    if discovered.manifest_data:
        parsed_m = parse_manifest_metadata(discovered.manifest_data)
        raw_m = discovered.manifest_data
        manifest_info = {
            "dataset_name": raw_m.get("dataset_name") or parsed_m.get("dataset_name", "Unknown"),
            "location": raw_m.get("location") or parsed_m.get("location", "N/A"),
            "crs": raw_m.get("crs") or parsed_m.get("crs", "N/A"),
            "s2_sensing_date": parsed_m.get("s2_sensing_date") or raw_m.get("target_date", "N/A"),
            "s1_sensing_date": parsed_m.get("s1_sensing_date") or raw_m.get("sar_date", "N/A"),
            "reference_date": parsed_m.get("reference_date") or raw_m.get("reference_date", "N/A"),
            "temporal_offset_hours": parsed_m.get("temporal_offset_hours"),
            "cloud_percent": raw_m.get("target_roi_cloud_percent") or parsed_m.get("cloud_assessment", {}).get("aoi_mean_cloud_probability_percent"),
            "reference_cloud_percent": raw_m.get("reference_roi_cloud_percent"),
        }

    return {
        "status": "success",
        "folder": str(discovered.folder_path),
        "sar_file": discovered.sar_path.name if discovered.sar_path else None,
        "opt_file": discovered.opt_path.name if discovered.opt_path else None,
        "ref_file": discovered.ref_path.name if discovered.ref_path else None,
        "manifest_file": discovered.manifest_path.name if discovered.manifest_path else None,
        "is_valid": val_report.is_valid,
        "errors": val_report.errors,
        "warnings": val_report.warnings,
        "sar_meta": val_report.sar_meta,
        "opt_meta": val_report.opt_meta,
        "ref_meta": val_report.ref_meta,
        "manifest": manifest_info,
    }


@app.post("/api/ingest/load_local")
def load_local_folder_rasters(folder_path: str = Form("data/real_patch")):
    """Load validated rasters from local directory into server memory."""
    dir_p = Path(folder_path)
    discovered = discover_local_patch(dir_p)
    val_report = validate_local_patch(discovered)

    if not val_report.is_valid:
        raise HTTPException(status_code=400, detail="; ".join(val_report.errors))

    sar_arr, opt_arr, ref_arr, patch_meta = load_local_patch(discovered)
    state.sar_arr = sar_arr
    state.opt_arr = opt_arr
    state.ref_arr = ref_arr
    state.patch_meta = patch_meta
    state.patch_files = discovered
    state.data_source_mode = "real_local"
    state.dataset_title = patch_meta.get("manifest", {}).get("dataset_name", f"Local Patch ({folder_path})")
    state.last_result = None
    state.last_metrics = None

    cloudy_preview = get_optical_preview(opt_arr)
    sar_preview = get_sar_preview(sar_arr)
    ref_preview = get_optical_preview(ref_arr) if ref_arr is not None else None

    # Parse temporal baseline
    manifest_meta = parse_manifest_metadata(patch_meta.get("manifest", {})) if patch_meta else {}

    return {
        "status": "success",
        "dataset_title": state.dataset_title,
        "sar_shape": list(sar_arr.shape),
        "opt_shape": list(opt_arr.shape),
        "has_reference": ref_arr is not None,
        "cloudy_preview": cloudy_preview,
        "sar_preview": sar_preview,
        "ref_preview": ref_preview,
        "manifest": manifest_meta,
    }


@app.post("/api/ingest/synthetic")
def load_synthetic_demo():
    """Generate and load reproducible 128x128 synthetic verification tensors."""
    sar_arr, opt_arr, ref_arr = generate_synthetic_verification_patch(height=128, width=128, seed=42)
    state.sar_arr = sar_arr
    state.opt_arr = opt_arr
    state.ref_arr = ref_arr
    state.patch_meta = {
        "dataset_name": "Synthetic Verification Patch",
        "location": "Synthetic Demonstration (Deterministic Seed 42)",
        "crs": "EPSG:32632",
        "grid_x_min": 500000.0,
        "grid_y_max": 5000000.0,
        "pixel_size_m": 10.0,
        "height_pixels": 128,
        "width_pixels": 128,
    }
    state.patch_files = None
    state.data_source_mode = "synthetic"
    state.dataset_title = "Synthetic Demonstration Patch (128x128)"
    state.last_result = None
    state.last_metrics = None

    cloudy_preview = get_optical_preview(opt_arr)
    sar_preview = get_sar_preview(sar_arr)
    ref_preview = get_optical_preview(ref_arr)

    return {
        "status": "success",
        "dataset_title": state.dataset_title,
        "sar_shape": list(sar_arr.shape),
        "opt_shape": list(opt_arr.shape),
        "has_reference": True,
        "cloudy_preview": cloudy_preview,
        "sar_preview": sar_preview,
        "ref_preview": ref_preview,
        "is_synthetic": True,
    }


@app.post("/api/ingest/upload")
async def upload_rasters(
    sar_file: UploadFile = File(...),
    opt_file: UploadFile = File(...),
    ref_file: Optional[UploadFile] = File(None),
):
    """Upload rasters via browser (GeoTIFF / NPZ / NPY) and load into server memory."""
    try:
        sar_bytes = await sar_file.read()
        opt_bytes = await opt_file.read()
        sar_arr, sar_meta = read_raster_file(sar_bytes, expected_channels=2, name="Sentinel-1 SAR")
        opt_arr, opt_meta = read_raster_file(opt_bytes, expected_channels=13, name="Sentinel-2 Optical")

        is_aligned, align_notes = validate_spatial_alignment(
            sar_meta, opt_meta, sar_shape=sar_arr.shape, opt_shape=opt_arr.shape
        )
        if not is_aligned:
            raise HTTPException(status_code=400, detail="Spatial alignment error: " + "; ".join(align_notes))

        ref_arr = None
        if ref_file is not None:
            ref_bytes = await ref_file.read()
            ref_arr, _ = read_raster_file(ref_bytes, expected_channels=13, name="Clear Optical Reference")
            if ref_arr.shape != opt_arr.shape:
                ref_arr = None

        state.sar_arr = sar_arr
        state.opt_arr = opt_arr
        state.ref_arr = ref_arr
        state.patch_meta = {
            "dataset_name": f"Uploaded ({opt_file.filename})",
            "crs": opt_meta.get("crs", "Unknown"),
            "height_pixels": opt_arr.shape[1],
            "width_pixels": opt_arr.shape[2],
        }
        state.patch_files = None
        state.data_source_mode = "upload"
        state.dataset_title = f"Uploaded Imagery ({sar_file.filename} + {opt_file.filename})"
        state.last_result = None
        state.last_metrics = None

        cloudy_preview = get_optical_preview(opt_arr)
        sar_preview = get_sar_preview(sar_arr)
        ref_preview = get_optical_preview(ref_arr) if ref_arr is not None else None

        return {
            "status": "success",
            "dataset_title": state.dataset_title,
            "sar_shape": list(sar_arr.shape),
            "opt_shape": list(opt_arr.shape),
            "has_reference": ref_arr is not None,
            "cloudy_preview": cloudy_preview,
            "sar_preview": sar_preview,
            "ref_preview": ref_preview,
            "align_notes": align_notes,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/reconstruct")
def execute_reconstruction(model_id: Optional[str] = Form(None)):
    """Execute real forward pass inference using the selected operational model."""
    if state.sar_arr is None or state.opt_arr is None:
        raise HTTPException(status_code=400, detail="No satellite data loaded. Please ingest data first.")

    target_model_id = model_id or state.selected_model_id
    if target_model_id not in {"ecrformer", "dsen2_cr"}:
        raise HTTPException(status_code=400, detail=f"Invalid model '{target_model_id}'.")

    state.selected_model_id = target_model_id

    t0 = time.perf_counter()
    if target_model_id == "ecrformer":
        adapter = ECRformerAdapter(variant="light", device="cpu")
        model_name = "ECRformer-Light"
    else:
        adapter = DSen2CRAdapter(device="cpu")
        model_name = "DSen2-CR (CARL)"

    result = adapter.predict(state.sar_arr, state.opt_arr)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Scientifically compute metrics if clear optical reference is available
    metrics = None
    metrics_dict = None
    if state.ref_arr is not None:
        ref = state.ref_arr
        if ref.shape == result.reconstructed_multispectral.shape and np.isfinite(ref).all():
            ref_norm = np.clip(ref, 0.0, 10000.0) / 10000.0 if ref.max() > 1.5 else ref
            metrics = compute_evaluation_metrics(result.reconstructed_multispectral, ref_norm)
            metrics_dict = {
                "mae": round(float(metrics.mae), 4),
                "rmse": round(float(metrics.rmse), 4),
                "rgb_psnr": round(float(metrics.rgb_psnr), 2),
                "rgb_ssim": round(float(metrics.rgb_ssim), 4),
                "per_band_mae": {k: round(float(v), 5) for k, v in metrics.per_band_mae.items()},
                "per_band_rmse": {k: round(float(v), 5) for k, v in metrics.per_band_rmse.items()},
            }

    # Save to outputs if local real patch
    saved_paths = None
    if state.data_source_mode == "real_local" and state.patch_meta:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / f"run_{target_model_id}_{timestamp_str}"
        saved_paths = save_inference_outputs(
            output_dir=out_dir,
            result=result,
            patch_metadata=state.patch_meta,
            patch_files=state.patch_files,
            evaluation_metrics=metrics,
        )
        saved_paths = {k: str(v) for k, v in saved_paths.items()}

    # Update state
    state.last_result = result
    state.last_metrics = metrics
    state.saved_paths = saved_paths

    recon_preview = array_to_base64_png(result.reconstructed_rgb_preview)
    cloudy_preview = array_to_base64_png(result.input_cloudy_rgb_preview)
    sar_preview = array_to_base64_png(result.input_sar_preview)
    ref_preview = get_optical_preview(state.ref_arr) if state.ref_arr is not None else None

    return {
        "status": "success",
        "model_id": target_model_id,
        "model_name": model_name,
        "latency_ms": round(latency_ms, 2),
        "device": str(result.device).upper(),
        "output_min": round(float(result.metadata["output_min"]), 4),
        "output_max": round(float(result.metadata["output_max"]), 4),
        "finite_valid": bool(np.all(np.isfinite(result.reconstructed_multispectral))),
        "reconstructed_preview": recon_preview,
        "cloudy_preview": cloudy_preview,
        "sar_preview": sar_preview,
        "ref_preview": ref_preview,
        "metrics": metrics_dict,
        "saved_paths": saved_paths,
    }


@app.get("/api/bands")
def get_multispectral_bands():
    """Return all 13 Sentinel-2 bands for the active reconstruction."""
    if state.last_result is None:
        raise HTTPException(status_code=400, detail="No reconstruction result available. Execute cloud removal first.")

    recon = state.last_result.reconstructed_multispectral
    bands_data = []

    band_meta = [
        {"name": "B1", "desc": "Coastal Aerosol", "wl": "443 nm", "gsd": "60 m"},
        {"name": "B2", "desc": "Blue", "wl": "490 nm", "gsd": "10 m"},
        {"name": "B3", "desc": "Green", "wl": "560 nm", "gsd": "10 m"},
        {"name": "B4", "desc": "Red", "wl": "665 nm", "gsd": "10 m"},
        {"name": "B5", "desc": "Red Edge 1", "wl": "705 nm", "gsd": "20 m"},
        {"name": "B6", "desc": "Red Edge 2", "wl": "740 nm", "gsd": "20 m"},
        {"name": "B7", "desc": "Red Edge 3", "wl": "783 nm", "gsd": "20 m"},
        {"name": "B8", "desc": "Broad NIR", "wl": "842 nm", "gsd": "10 m"},
        {"name": "B8A", "desc": "Narrow NIR", "wl": "865 nm", "gsd": "20 m"},
        {"name": "B9", "desc": "Water Vapor", "wl": "945 nm", "gsd": "60 m"},
        {"name": "B10", "desc": "Cirrus (Tropospheric Absorption)", "wl": "1375 nm", "gsd": "60 m"},
        {"name": "B11", "desc": "SWIR 1", "wl": "1610 nm", "gsd": "20 m"},
        {"name": "B12", "desc": "SWIR 2", "wl": "2190 nm", "gsd": "20 m"},
    ]

    for idx in range(13):
        b_slice = recon[idx]
        b_info = band_meta[idx]
        b_img = array_greyscale_to_base64_png(b_slice)
        bands_data.append({
            "index": idx,
            "band_code": S2_BANDS_13[idx],
            "display_name": f"{S2_BANDS_13[idx]} — {b_info['desc']}",
            "wavelength": b_info["wl"],
            "gsd": b_info["gsd"],
            "description": b_info["desc"],
            "min": round(float(b_slice.min()), 5),
            "max": round(float(b_slice.max()), 5),
            "mean": round(float(b_slice.mean()), 5),
            "image": b_img,
        })

    return {"status": "success", "bands": bands_data}


@app.get("/api/benchmarks/{city}")
def get_benchmark_city(city: str):
    """Return historical verified experimental results for Munich or Mumbai Dadar."""
    city_lower = city.lower()
    if "munich" in city_lower:
        return {
            "city": "Munich, Germany",
            "grid": "256x256 pixels, 10m GSD, EPSG:32632",
            "cloud_cover": "100.0% (Thick Cumulus Cover)",
            "temporal_offset": "-17 days (2023-06-25)",
            "s2_target_date": "2023-07-12 10:17:35 UTC",
            "s1_sar_date": "2023-07-10 17:07:47 UTC (-41.2 hours)",
            "reference_date": "2023-06-25 10:27:30 UTC",
            "models": {
                "ecrformer": {
                    "name": "ECRformer-Light",
                    "latency_ms": 4373.17,
                    "mae": 0.0157,
                    "rmse": 0.0241,
                    "rgb_psnr": 32.95,
                    "rgb_ssim": 0.8604,
                    "output_range": [0.0011, 0.4686],
                },
                "dsen2_cr": {
                    "name": "DSen2-CR (CARL)",
                    "latency_ms": 10308.94,
                    "mae": 0.0201,
                    "rmse": 0.0297,
                    "rgb_psnr": 31.99,
                    "rgb_ssim": 0.8323,
                    "output_range": [0.0000, 0.6465],
                },
                "cloudy_vs_ref": {
                    "name": "Cloudy Input Baseline",
                    "mae": 0.0384,
                    "rmse": 0.0492,
                    "rgb_psnr": 24.12,
                    "rgb_ssim": 0.5412,
                },
            },
        }
    elif "mumbai" in city_lower:
        return {
            "city": "Mumbai Dadar, India",
            "grid": "256x256 pixels, 10m GSD, EPSG:32643",
            "cloud_cover": "99.58% (Monsoon Overcast)",
            "temporal_offset": "-10 days (2023-06-06)",
            "s2_target_date": "2023-06-16",
            "s1_sar_date": "2023-06-11 (5.2 days)",
            "reference_date": "2023-06-06",
            "models": {
                "ecrformer": {
                    "name": "ECRformer-Light",
                    "latency_ms": 3914.73,
                    "mae": 0.0258,
                    "rmse": 0.0357,
                    "rgb_psnr": 29.62,
                    "rgb_ssim": 0.6760,
                    "output_range": [0.0007, 0.6172],
                },
                "dsen2_cr": {
                    "name": "DSen2-CR (CARL)",
                    "latency_ms": 11686.86,
                    "mae": 0.0256,
                    "rmse": 0.0380,
                    "rgb_psnr": 29.08,
                    "rgb_ssim": 0.6579,
                    "output_range": [0.0000, 0.5685],
                },
                "cloudy_vs_ref": {
                    "name": "Cloudy Input Baseline",
                    "mae": 0.0412,
                    "rmse": 0.0561,
                    "rgb_psnr": 23.85,
                    "rgb_ssim": 0.4912,
                },
            },
            "per_band_mae": {
                "B1": {"cloudy": 0.035, "ecrformer": 0.0084, "dsen2_cr": 0.0102},
                "B2": {"cloudy": 0.038, "ecrformer": 0.0200, "dsen2_cr": 0.0200},
                "B3": {"cloudy": 0.041, "ecrformer": 0.0227, "dsen2_cr": 0.0222},
                "B4": {"cloudy": 0.045, "ecrformer": 0.0301, "dsen2_cr": 0.0296},
                "B5": {"cloudy": 0.042, "ecrformer": 0.0273, "dsen2_cr": 0.0246},
                "B6": {"cloudy": 0.046, "ecrformer": 0.0296, "dsen2_cr": 0.0316},
                "B7": {"cloudy": 0.048, "ecrformer": 0.0286, "dsen2_cr": 0.0418},
                "B8": {"cloudy": 0.051, "ecrformer": 0.0327, "dsen2_cr": 0.0408},
                "B8A": {"cloudy": 0.049, "ecrformer": 0.0302, "dsen2_cr": 0.0446},
                "B9": {"cloudy": 0.025, "ecrformer": 0.0199, "dsen2_cr": 0.0043},
                "B10": {"cloudy": 0.0015, "ecrformer": 0.0002, "dsen2_cr": 0.0011},
                "B11": {"cloudy": 0.055, "ecrformer": 0.0437, "dsen2_cr": 0.0317},
                "B12": {"cloudy": 0.052, "ecrformer": 0.0419, "dsen2_cr": 0.0302},
            },
        }
    else:
        raise HTTPException(status_code=404, detail="Benchmark not found. Choose 'munich' or 'mumbai'.")


@app.get("/api/export/npz")
def export_npz():
    """Download 13-band reconstructed multispectral array as NPZ."""
    if state.last_result is None:
        raise HTTPException(status_code=400, detail="No reconstruction result available.")
    data_bytes = export_npz_bytes(state.last_result.reconstructed_multispectral, "reconstructed_13bands")
    return Response(
        content=data_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=geoclear_reconstructed_s2_13bands.npz"},
    )


@app.get("/api/export/png")
def export_png():
    """Download reconstructed true-color preview as PNG."""
    if state.last_result is None:
        raise HTTPException(status_code=400, detail="No reconstruction result available.")
    png_bytes = export_png_bytes(state.last_result.reconstructed_rgb_preview)
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=geoclear_reconstructed_preview.png"},
    )


@app.get("/api/export/manifest")
def export_manifest():
    """Download processing manifest JSON if available."""
    if state.saved_paths and "manifest" in state.saved_paths:
        p = Path(state.saved_paths["manifest"])
        if p.is_file():
            return FileResponse(p, filename="processing_manifest.json", media_type="application/json")
    # Return in-memory metadata
    meta = {
        "dataset_name": state.dataset_title,
        "model_id": state.selected_model_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    return JSONResponse(meta)


@app.get("/api/export/report")
def export_report():
    """Download markdown reconstruction report if available."""
    if state.saved_paths and "report" in state.saved_paths:
        p = Path(state.saved_paths["report"])
        if p.is_file():
            return FileResponse(p, filename="reconstruction_report.md", media_type="text/markdown")

    content = f"# GeoClear Reconstruction Report\nDataset: {state.dataset_title}\nModel: {state.selected_model_id}\n"
    return Response(content=content, media_type="text/markdown", headers={"Content-Disposition": "attachment; filename=reconstruction_report.md"})


# Mount static files
static_dir = _ROOT / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index_page():
    """Serve the single-page application entry point."""
    index_file = static_dir / "index.html"
    if not index_file.is_file():
        return HTMLResponse("<h1>GeoClear API is running. static/index.html is loading...</h1>")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("GEOCLEAR WEB APPLICATION SERVER")
    print("Serving on http://127.0.0.1:8000")
    print("=" * 70)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
