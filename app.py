"""GeoClear: Data-Aware Multimodal Earth-Observation AI Platform.

Streamlit interactive dashboard for data declaration, strict model compatibility
filtering, input validation, ECRformer inference, and scientifically valid evaluation.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import streamlit as st

# Windows PROJ environment isolation
if "PROJ_LIB" in os.environ and "PostgreSQL" in os.environ["PROJ_LIB"]:
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    rasterio_proj = os.path.join(site_packages, "rasterio", "proj_data")
    if os.path.isdir(rasterio_proj):
        os.environ["PROJ_LIB"] = rasterio_proj
        os.environ["PROJ_DATA"] = rasterio_proj

# Ensure repository root is on sys.path
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
from geoclear.adapters.ecrformer_adapter import ECRformerAdapter, ECRformerInferenceResult
from geoclear.adapters.dsen2cr_adapter import DSen2CRAdapter, DSen2CRInferenceResult
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


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GeoClear — Earth Observation AI Platform",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphic Dark-Themed Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Gradient Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(16, 24, 40, 0.95) 0%, rgba(30, 41, 59, 0.98) 50%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }

    .hero-title {
        font-size: 2.2rem;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.5;
        margin-bottom: 0px;
    }

    /* Model Card Containers */
    .model-card-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .model-card-box:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-compatible {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    .badge-incompatible {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
    .badge-ready {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-blocked {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Disclaimers */
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        color: #fef3c7;
        font-size: 0.9rem;
        margin: 14px 0;
    }

    .info-box {
        background-color: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 4px 8px 8px 4px;
        color: #e0f2fe;
        font-size: 0.9rem;
        margin: 14px 0;
    }

    /* Metrics Table Styling */
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar: Platform Information & Candidate Status
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/google/material-design-icons/master/png/action/satellite_alt/materialicons/48dp/2x/baseline_satellite_alt_black_48dp.png",
        width=48,
    )
    st.title("GeoClear Platform")
    st.caption("Multimodal Earth-Observation Cloud Removal")

    st.markdown("---")
    st.markdown("### Available Models")

    for card in ModelRegistry.list_models():
        with st.expander(f"{card.display_name}", expanded=False):
            st.markdown(f"**Description:** {card.description}")
            st.markdown(f"**Input Modality:** Dual-Sensor (SAR + Optical)")
            st.markdown(f"**Channels:** 15 (2 SAR polarizations + 13 Multispectral bands)")
            st.markdown("**Status:** Ready for Operational Inference")




# ---------------------------------------------------------------------------
# Header / Hero Section
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">GeoClear — Multimodal Cloud Removal</div>
        <div class="hero-subtitle">
            A data-aware satellite AI dashboard. Input eligibility is strictly derived from 
            real sensor data declarations. Models are never claimed operational without verified execution.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Workflow Step 1: Declare Available Data
# ---------------------------------------------------------------------------

st.header("Step 1: Declare Your Available Data")
st.markdown("Declare what satellite data you possess. GeoClear will filter eligible candidates strictly based on real physics and documented model requirements.")

col1, col2, col3 = st.columns([1.2, 1.2, 1.2])

with col1:
    st.subheader("1. Sensors & Platforms")
    has_s2 = st.checkbox("Sentinel-2 Multispectral Optical", value=True)
    has_s1 = st.checkbox("Sentinel-1 Synthetic Aperture Radar (SAR)", value=True)
    has_generic_rgb = st.checkbox("Standard RGB Optical (Drone / Commercial)", value=False)

    sensors: Set[SensorType] = set()
    if has_s2:
        sensors.add(SensorType.SENTINEL_2)
    if has_s1:
        sensors.add(SensorType.SENTINEL_1)
    if has_generic_rgb:
        sensors.add(SensorType.GENERIC_RGB)

with col2:
    st.subheader("2. Band Coverage")
    band_choice = st.radio(
        "Optical Band Availability:",
        [
            "All 13 Sentinel-2 Bands (B1-B12, B8A)",
            "RGB Only (B4, B3, B2)",
            "None / SAR Only",
        ],
        index=0,
    )

    if band_choice == "All 13 Sentinel-2 Bands (B1-B12, B8A)":
        optical_bands = set(S2_BANDS_13)
        modalities: Set[Modality] = {Modality.OPTICAL, Modality.MULTISPECTRAL}
    elif band_choice == "RGB Only (B4, B3, B2)":
        optical_bands = {"B4", "B3", "B2"}
        modalities = {Modality.OPTICAL}
    else:
        optical_bands = set()
        modalities = set()

    sar_pols: Set[str] = set()
    if has_s1:
        st.markdown("**SAR Polarizations:**")
        vv = st.checkbox("VV Polarization", value=True)
        vh = st.checkbox("VH Polarization", value=True)
        if vv:
            sar_pols.add("VV")
        if vh:
            sar_pols.add("VH")
        if sar_pols:
            modalities.add(Modality.SAR)

with col3:
    st.subheader("3. Temporal & Reference Data")
    temporal_count = st.number_input(
        "Co-registered Temporal Passes:",
        min_value=1,
        max_value=12,
        value=1,
        help="Number of repeat cloudy optical observations of the exact same footprint.",
    )
    if temporal_count >= 2:
        modalities.add(Modality.MULTI_TEMPORAL)

    has_ref = st.checkbox(
        "I have a clear optical reference for evaluation",
        value=False,
        help="Required only if calculating quantitative reconstruction metrics (MAE, PSNR, SSIM).",
    )

    img_dim = st.selectbox(
        "Spatial Patch Dimensions:",
        ["128x128", "256x256", "512x512", "Arbitrary / Non-standard"],
        index=1,
    )
    if img_dim == "128x128":
        h_dim, w_dim = 128, 128
    elif img_dim == "256x256":
        h_dim, w_dim = 256, 256
    elif img_dim == "512x512":
        h_dim, w_dim = 512, 512
    else:
        h_dim, w_dim = None, None

# Build DeclaredData instance
try:
    user_declared = DeclaredData(
        modalities=modalities,
        sensors=sensors,
        optical_bands=optical_bands,
        sar_polarizations=sar_pols,
        temporal_observations=temporal_count,
        has_ground_truth_reference=has_ref,
        image_height=h_dim,
        image_width=w_dim,
    )
    user_declared.validate()
    declaration_valid = True
except ValueError as e:
    st.error(f"Invalid Data Declaration: {e}")
    declaration_valid = False


# ---------------------------------------------------------------------------
# Workflow Step 2: Show Compatible Models & Status
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Step 2: Compatibility Evaluation & Operational Readiness")

if declaration_valid:
    report = CompatibilityEngine.evaluate(user_declared)

    st.markdown(
        f"Evaluated **{report.total_evaluated} candidate models** against your declared inputs: "
        f"**{len(report.compatible_models)} input-compatible**, **{len(report.incompatible_models)} excluded**."
    )

    tab_comp, tab_incomp = st.tabs(["Eligible Models", "Excluded Models"])

    with tab_comp:
        if not report.compatible_models:
            st.warning("No candidate models are compatible with your currently declared data.")
        else:
            for res in report.compatible_models:
                card = ModelRegistry.get_model(res.model_id)
                with st.container():
                    st.markdown(
                        f"""
                        <div class="model-card-box">
                            <span class="badge badge-compatible">Input Compatible</span>
                            <span class="badge {'badge-ready' if res.is_executable else 'badge-blocked'}">
                                {'Executable' if res.is_executable else 'Execution Blocked'}
                            </span>
                            <span class="badge" style="background-color: rgba(148, 163, 184, 0.15); color: #cbd5e1;">
                                {card.verification_status.value}
                            </span>
                            <h3 style="margin-top: 8px; margin-bottom: 4px;">{res.model_name}</h3>
                            <p style="color: #94a3b8; font-size: 0.92rem; margin-bottom: 8px;">{card.description}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if res.execution_blockers:
                        st.caption("⚠️ **Execution Blockers:**")
                        for blk in res.execution_blockers:
                            st.caption(f"- {blk}")

    with tab_incomp:
        if not report.incompatible_models:
            st.success("All candidate models are eligible!")
        else:
            for res in report.incompatible_models:
                card = ModelRegistry.get_model(res.model_id)
                with st.expander(f"❌ {res.model_name} — Ineligible", expanded=False):
                    st.markdown(f"**Model Description:** {card.description}")
                    st.markdown("**Reasons for Exclusion:**")
                    for reason in res.exclusion_reasons:
                        st.markdown(f"- 🚫 {reason}")
                    if res.missing_inputs:
                        st.markdown("**Missing Required Inputs:**")
                        for inp in res.missing_inputs:
                            st.markdown(f"- 📦 {inp}")


# ---------------------------------------------------------------------------
# Workflow Step 3: Select a Model
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Step 3: Model Selection")

selected_model_id = None
executable_ids = [m.model_id for m in report.executable_models] if declaration_valid else []

if not executable_ids:
    st.info(
        "Currently, only **ECRformer-Light** is configured and verified for execution in GeoClear. "
        "Declare both **Sentinel-1 SAR (VV/VH)** and **All 13 Sentinel-2 Bands** in Step 1 to unlock inference."
    )
else:
    selected_model_id = st.selectbox(
        "Select an Executable Model:",
        options=executable_ids,
        format_func=lambda mid: f"{ModelRegistry.get_model(mid).display_name} (Ready for Inference)",
    )

    selected_card = ModelRegistry.get_model(selected_model_id)

    st.markdown(
        f"""
        <div class="info-box">
            <b>Operational Model Selected:</b> {selected_card.display_name}<br>
            <b>Description:</b> {selected_card.description}<br>
            <b>Architecture:</b> 15-channel multimodal input (Sentinel-1 SAR + Sentinel-2 MSI) &rarr; 13-band surface reflectance
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Workflow Steps 4-7: Upload Inputs, Validate & Run Inference
# ---------------------------------------------------------------------------

if selected_model_id in {"ecrformer", "dsen2_cr"}:
    st.markdown("---")
    st.header("Step 4 & 5: Upload Inputs & Validation")

    if selected_model_id == "ecrformer":
        active_model_name = "ECRformer"
        st.markdown(
            "ECRformer requires **two co-registered input files**: "
            "Sentinel-1 SAR (`VV, VH` in dB) and Sentinel-2 Optical (`B1-B12, B8A` in DN `[0, 10000]`)."
        )
    else:
        active_model_name = "DSen2-CR"
        st.markdown(
            "DSen2-CR requires **two co-registered input files**: "
            "Sentinel-1 SAR (`VV, VH` in dB) and Sentinel-2 Optical (`B1-B12, B8A` in DN `[0, 10000]`)."
        )

    with st.expander("📖 Input Format & Co-Registration Specifications", expanded=False):
        st.markdown(
            """
            #### Strict Input Requirements
            - **Sentinel-1 SAR**:
              - Channels: Exactly 2 bands corresponding to **[VV, VH]** polarizations.
              - Scale: Calibrated backscatter $\\sigma_0$ in decibels (dB), typically in $[-25.0, 0.0]$ dB.
              - Preprocessing: Scaled internally per model specifications.
            - **Sentinel-2 Multispectral Optical**:
              - Channels: Exactly 13 bands in standard SEN12MS-CR order:  
                `[B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]`.
              - Scale: Level-1C / Level-2A Top-of-Atmosphere or Surface Reflectance Digital Numbers (DN) in $[0, 10000]$.
              - Preprocessing: Scaled internally per model specifications.
            - **Spatial Dimensions**:
              - Patch height and width must be strictly divisible by **32** (e.g. $128 \\times 128$, $256 \\times 256$, $512 \\times 512$).
            - **Supported File Types**:
              - **GeoTIFF** (`.tif`, `.tiff`): Recommended. Preserves coordinate reference system (CRS), bounds, and pixel resolution for co-registration checks.
              - **NumPy** (`.npz`, `.npy`): Supported for automated testing. Arrays must have shape `(C, H, W)` or `(H, W, C)`.
            - **Data Integrity**:
              - All values must be finite numerical values. Files with NaN or infinite values are rejected.
            """
        )

    with st.expander("🛰️ Instructions for Obtaining & Preparing Real Satellite Data", expanded=False):
        st.markdown(
            """
            #### How to Prepare Real Co-Registered Patches
            1. **SEN12MS-CR Benchmark (Schmitt et al.)**:
               - Download co-registered Sentinel-1 SAR and Sentinel-2 cloudy/cloud-free image pairs from the official SEN12MS-CR repository.
               - Extract matching ROI tiles and save as GeoTIFF or NumPy arrays.
            2. **Copernicus Data Space / Microsoft Planetary Computer / Google Earth Engine**:
               - Query Sentinel-1 GRD (IW mode) and Sentinel-2 L1C/L2A orthorectified imagery over your area of interest.
               - Reproject Sentinel-1 to match the 10m Sentinel-2 UTM pixel grid.
               - Stack all 13 Sentinel-2 bands in exact spectral order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12.
               - Crop to dimensions divisible by 32 (e.g. 256x256).
            3. **Operational Verification Note**:
               - *Real-data inference verification (`REAL_DATA_INFERENCE_VERIFIED`) requires genuine satellite scenes.*
               - The synthetic test patch verifies model graph loading, tensor shapes, normalization, and memory management.
            """
        )

    # Model configuration synchronized with Step 3 as single source of truth
    if selected_model_id == "ecrformer":
        variant = st.radio(
            "Model Variant:",
            ["ECRformer-Light (~3.7M params, Fast)", "ECRformer-Full (~11.3M params)"],
            index=0,
        )
        adapter_variant = "light" if "Light" in variant else "full"
    else:
        st.info("ℹ️ **Active Architecture**: DSen2-CR (16 residual blocks, 256 feature channels, CARL loss checkpoint, ~18.9M params). Evaluates single-temporal Sentinel-1 + Sentinel-2 directly.")
        adapter_variant = "carl"

    st.markdown("### Select Data Ingestion Mode")
    ingestion_mode = st.radio(
        "Ingestion Method:",
        [
            "📁 Local Directory Ingestion (Recommended for Large & Real Sentinel Data)",
            "📤 Manual Browser Upload (Small In-Memory Files)",
            "🧪 Synthetic Test Demonstration Patch",
        ],
        index=0,
    )

    sar_arr: Optional[np.ndarray] = None
    opt_arr: Optional[np.ndarray] = None
    ref_arr: Optional[np.ndarray] = None
    sar_meta: Optional[Dict] = None
    opt_meta: Optional[Dict] = None
    patch_meta: Optional[Dict] = None
    patch_files: Optional[LocalPatchFiles] = None
    is_demo = False

    # -----------------------------------------------------------------------
    # Mode 1: Local Directory Ingestion
    # -----------------------------------------------------------------------
    if ingestion_mode.startswith("📁"):
        st.markdown("#### Local Data Folder")
        st.caption("Reads GeoTIFFs directly from local disk. No browser upload size limits; inspects metadata non-destructively.")

        local_dir_input = st.text_input(
            "Local Directory Path:",
            value="data/real_patch",
            help="Path to folder containing s1_sar.tif, s2_cloudy.tif, and optional manifest.json / s2_reference.tif.",
        )

        col_d1, col_d2 = st.columns([1.5, 2.5])
        with col_d1:
            btn_inspect = st.button("🔍 Inspect & Validate Local Folder", type="secondary")

        discovered = discover_local_patch(local_dir_input)
        val_report = validate_local_patch(discovered)

        # Always show discovered files overview if directory exists
        if discovered.folder_path.is_dir():
            st.markdown("##### Discovered Files:")
            fcols = st.columns(3)
            with fcols[0]:
                if discovered.sar_path:
                    st.success(f"**SAR:** `{discovered.sar_path.name}`")
                else:
                    st.error("**SAR:** Missing (`*s1*.tif` / `*sar*.tif`)")
            with fcols[1]:
                if discovered.opt_path:
                    st.success(f"**Optical:** `{discovered.opt_path.name}`")
                else:
                    st.error("**Optical:** Missing (`*s2*.tif` / `*cloudy*.tif`)")
            with fcols[2]:
                if discovered.ref_path:
                    st.info(f"**Reference:** `{discovered.ref_path.name}`")
                else:
                    st.caption("**Reference:** None (Evaluation disabled)")

            if discovered.manifest_data:
                with st.expander("🛰️ Provenance Manifest (from GEE / Script)", expanded=True):
                    m_parsed = parse_manifest_metadata(discovered.manifest_data)
                    mcols = st.columns(3)
                    s2_date = m_parsed.get("s2_sensing_date") or "Unknown"
                    s1_date = m_parsed.get("s1_sensing_date") or "Unknown"
                    offset = m_parsed.get("temporal_offset_hours")
                    mcols[0].markdown(f"**S2 Date:** `{s2_date}`")
                    mcols[1].markdown(f"**S1 Date:** `{s1_date}`")
                    mcols[2].markdown(f"**Temporal Offset:** `{f'{offset:.1f} hours' if offset is not None else 'Unknown'}`")
                    s2_scene = m_parsed.get("s2_scene_id") or "N/A"
                    s1_scene = m_parsed.get("s1_scene_id") or "N/A"
                    collection = m_parsed.get("s2_collection") or "Unknown"
                    st.caption(f"S2 Scene: `{s2_scene}` | S1 Scene: `{s1_scene}` | Collection: `{collection}`")

            # Validation details
            if val_report.is_valid:
                st.success("✅ **Metadata Validation Passed**: Exact band counts, divisible-by-32 dimensions, and spatial grids verified.")
                with st.expander("View Grid & Metadata Properties", expanded=False):
                    st.json({
                        "SAR": val_report.sar_meta,
                        "Optical": val_report.opt_meta,
                        "Reference": val_report.ref_meta,
                    })

                if st.button("📥 Load Validated Patch into Memory", type="primary"):
                    try:
                        sar_arr, opt_arr, ref_arr, patch_meta = load_local_patch(discovered)
                        st.session_state["sar_arr"] = sar_arr
                        st.session_state["opt_arr"] = opt_arr
                        st.session_state["ref_arr"] = ref_arr
                        st.session_state["patch_meta"] = patch_meta
                        st.session_state["patch_files"] = discovered
                        st.session_state["data_source_mode"] = "real_local"
                        st.session_state["is_demo"] = False
                        st.success(f"Successfully loaded real imagery patch! SAR: {sar_arr.shape}, Optical: {opt_arr.shape}")
                    except Exception as exc:
                        st.error(f"Failed to load arrays: {exc}")
            else:
                for err in val_report.errors:
                    st.error(f"❌ {err}")
                for warn in val_report.warnings:
                    st.warning(f"⚠️ {warn}")
        else:
            st.info(f"Directory `{local_dir_input}` does not exist yet. Run the GEE export script or place co-registered GeoTIFFs in this path.")

    # -----------------------------------------------------------------------
    # Mode 2: Manual Browser Upload
    # -----------------------------------------------------------------------
    elif ingestion_mode.startswith("📤"):
        st.markdown("#### Manual Upload")
        col_upload_s1, col_upload_s2 = st.columns(2)

        with col_upload_s1:
            s1_file = st.file_uploader(
                "Upload Sentinel-1 SAR (GeoTIFF / NPZ / NPY, 2 bands: VV, VH)",
                type=["tif", "tiff", "npz", "npy"],
                key="sar_upload",
            )

        with col_upload_s2:
            s2_file = st.file_uploader(
                "Upload Sentinel-2 Optical (GeoTIFF / NPZ / NPY, 13 bands)",
                type=["tif", "tiff", "npz", "npy"],
                key="opt_upload",
            )

        ref_file = None
        if has_ref:
            ref_file = st.file_uploader(
                "Upload Cloud-Free Optical Reference (GeoTIFF / NPZ / NPY, 13 bands)",
                type=["tif", "tiff", "npz", "npy"],
                key="ref_upload",
                help="Used strictly for ground-truth evaluation metrics. Never passed as input to the model.",
            )

        if s1_file is not None and s2_file is not None:
            try:
                sar_arr, sar_meta = read_raster_file(s1_file.getvalue(), expected_channels=2, name="Sentinel-1 SAR")
                opt_arr, opt_meta = read_raster_file(s2_file.getvalue(), expected_channels=13, name="Sentinel-2 Optical")
                st.session_state["sar_arr"] = sar_arr
                st.session_state["opt_arr"] = opt_arr
                st.session_state["data_source_mode"] = "upload"
                st.session_state["is_demo"] = False

                is_aligned, align_notes = validate_spatial_alignment(
                    sar_meta, opt_meta, sar_shape=sar_arr.shape, opt_shape=opt_arr.shape
                )
                if not is_aligned:
                    for note in align_notes:
                        st.error(f"Spatial Alignment Error: {note}")
                    sar_arr, opt_arr = None, None
                else:
                    st.success(f"Validated upload files! SAR shape: {sar_arr.shape}, Optical shape: {opt_arr.shape}")
                    for note in align_notes:
                        st.info(f"Spatial Check: {note}")

                if ref_file is not None and sar_arr is not None:
                    try:
                        ref_arr, _ = read_raster_file(ref_file.getvalue(), expected_channels=13, name="Ground Truth Reference")
                        if ref_arr.shape != opt_arr.shape:
                            st.error(f"Reference shape {ref_arr.shape} does not match Optical shape {opt_arr.shape}. Metrics disabled.")
                            ref_arr = None
                        else:
                            st.session_state["ref_arr"] = ref_arr
                            st.success("Ground-truth reference registered for post-inference evaluation.")
                    except Exception as r_exc:
                        st.error(f"Ground Truth Reference Error: {r_exc}")
                        ref_arr = None
            except Exception as exc:
                st.error(f"Upload Validation Error: {exc}")
                sar_arr, opt_arr = None, None
        elif s1_file is not None or s2_file is not None:
            missing = ["Sentinel-1 SAR (2 bands: VV, VH)"] if s1_file is None else ["Sentinel-2 Optical (13 bands)"]
            st.warning(f"Incomplete upload: please upload both required files before running inference. Missing: {', '.join(missing)}.")

    # -----------------------------------------------------------------------
    # Mode 3: Synthetic Demonstration
    # -----------------------------------------------------------------------
    else:
        st.markdown("#### Synthetic Test Generator")
        st.caption("Generate multi-band synthetic test tensors to test software mechanics and pipeline execution without downloading satellite files.")
        if st.button("🧪 Generate & Load Synthetic Verification Patch (128x128)", type="primary"):
            sar_arr, opt_arr, ref_arr = generate_synthetic_verification_patch(height=128, width=128, seed=42)
            st.session_state["sar_arr"] = sar_arr
            st.session_state["opt_arr"] = opt_arr
            st.session_state["ref_arr"] = ref_arr
            st.session_state["data_source_mode"] = "synthetic"
            st.session_state["is_demo"] = True
            st.success("Loaded synthetic multi-band verification patch! (Tagged as synthetic demonstration)")

    # Restore from session state if available
    if "sar_arr" in st.session_state:
        sar_arr = st.session_state["sar_arr"]
        opt_arr = st.session_state["opt_arr"]
        ref_arr = st.session_state.get("ref_arr")
        patch_meta = st.session_state.get("patch_meta")
        patch_files = st.session_state.get("patch_files")
        data_source_mode = st.session_state.get("data_source_mode", "synthetic")
        is_demo = (data_source_mode == "synthetic")
    else:
        data_source_mode = "none"

    # Step 6 & 7: Execution & Results
    if sar_arr is not None and opt_arr is not None:
        st.markdown("---")
        st.header("Step 6: Run Inference")

        if data_source_mode == "synthetic":
            st.warning(
                "🧪 **SYNTHETIC TEST DEMONSTRATION**: The loaded inputs are synthetic verification tensors "
                "generated to test software mechanics and tensor alignment. This test does NOT represent "
                "real-world satellite cloud removal accuracy."
            )
        elif data_source_mode == "real_local":
            st.markdown(
                """
                <div class="info-box">
                    <b>📁 REAL SATELLITE DATA INGESTION</b>: Running inference on genuine Sentinel-1 SAR and 
                    Sentinel-2 optical imagery loaded directly from local GeoTIFF files.
                </div>
                """,
                unsafe_allow_html=True,
            )
            if patch_meta and patch_meta.get("manifest"):
                parsed_pm = parse_manifest_metadata(patch_meta["manifest"])
                toff = parsed_pm.get("temporal_offset_hours")
                if toff is not None:
                    st.caption(f"ℹ️ **Temporal Difference**: SAR acquired {abs(toff):.1f} hours {'after' if toff > 0 else 'before'} Optical scene. (SAR is not a simultaneous observation).")

        if st.button("🚀 Execute Cloud Removal", type="primary"):
            with st.spinner(f"Executing {active_model_name} forward pass..."):
                try:
                    # Clear any prior run results or errors
                    st.session_state.pop("last_result", None)
                    st.session_state.pop("last_ref", None)
                    st.session_state.pop("saved_paths", None)
                    st.session_state.pop("inference_error", None)

                    if active_model_name == "ECRformer":
                        adapter = ECRformerAdapter(variant=adapter_variant, device="cpu")
                    else:
                        adapter = DSen2CRAdapter(device="cpu")

                    result = adapter.predict(sar_arr, opt_arr)

                    # If running on local patch: save output bundle with provenance manifest
                    saved_paths = None
                    if data_source_mode == "real_local" and patch_meta:
                        out_dir = Path("outputs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
                        saved_paths = save_inference_outputs(
                            output_dir=out_dir,
                            result=result,
                            patch_metadata=patch_meta,
                            patch_files=patch_files,
                            evaluation_metrics=None,
                        )

                    # Atomically commit state only when forward pass and artifact exports have both completed
                    st.session_state["last_result"] = result
                    st.session_state["last_ref"] = ref_arr
                    st.session_state["last_model_name"] = active_model_name
                    if saved_paths:
                        st.session_state["saved_paths"] = saved_paths
                    else:
                        st.session_state.pop("saved_paths", None)
                except Exception as exc:
                    # Ensure partial or stale state is purged so success UI and error banner never co-occur
                    st.session_state.pop("last_result", None)
                    st.session_state.pop("last_ref", None)
                    st.session_state.pop("saved_paths", None)
                    st.session_state["inference_error"] = str(exc)
                    st.error(f"Inference Failure: {exc}")

    # Display Results
    if "last_result" in st.session_state:
        result: Any = st.session_state["last_result"]
        ref_for_eval = st.session_state.get("last_ref")
        saved_paths = st.session_state.get("saved_paths")
        model_name_disp = st.session_state.get("last_model_name", "Model")

        st.markdown("---")
        st.header("Step 7: Reconstructed Surface Reflectance & Previews")

        # Three Previews Side by Side
        pcol1, pcol2, pcol3 = st.columns(3)

        with pcol1:
            st.subheader("1. Input Cloudy Optical")
            st.image(
                result.input_cloudy_rgb_preview,
                caption="Sentinel-2 True-Color RGB (B4, B3, B2) [Brightness x3.0]",
                use_container_width=True,
            )

        with pcol2:
            st.subheader("2. Sentinel-1 SAR")
            st.image(
                result.input_sar_preview,
                caption="SAR False-Color Composite (R: VV, G: VH, B: ratio)",
                use_container_width=True,
            )

        with pcol3:
            st.subheader("3. Reconstructed Optical Imagery")
            st.image(
                result.reconstructed_rgb_preview,
                caption=f"{model_name_disp} Reconstructed Optical RGB (B4, B3, B2) [Brightness x3.0]",
                use_container_width=True,
            )

        # Output Telemetry Bar
        tcol1, tcol2, tcol3, tcol4 = st.columns(4)
        with tcol1:
            st.markdown(f"<div class='metric-label'>Inference Latency</div><div class='metric-value'>{result.metadata['inference_time_ms']} ms</div>", unsafe_allow_html=True)
        with tcol2:
            st.markdown(f"<div class='metric-label'>Device Used</div><div class='metric-value'>{result.device.upper()}</div>", unsafe_allow_html=True)
        with tcol3:
            st.markdown(f"<div class='metric-label'>Output Range</div><div class='metric-value'>[{result.metadata['output_min']:.3f}, {result.metadata['output_max']:.3f}]</div>", unsafe_allow_html=True)
        with tcol4:
            st.markdown(f"<div class='metric-label'>Finite Check</div><div class='metric-value'>✅ Passed</div>", unsafe_allow_html=True)

        # Local Saved Outputs Banner
        if saved_paths:
            with st.container():
                st.markdown(
                    f"""
                    <div class="info-box">
                        <b>💾 Traceable Output Package Saved to Disk:</b><br>
                        - <b>GeoTIFF (13 Bands):</b> <code>{saved_paths.get('geotiff_output', 'N/A')}</code><br>
                        - <b>NumPy Array (.npz):</b> <code>{saved_paths.get('npz_output', 'N/A')}</code><br>
                        - <b>RGB Preview (.png):</b> <code>{saved_paths.get('rgb_preview', 'N/A')}</code><br>
                        - <b>Provenance Manifest:</b> <code>{saved_paths.get('manifest', 'N/A')}</code><br>
                        - <b>Summary Report:</b> <code>{saved_paths.get('report', 'N/A')}</code>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Band Explorer
        with st.expander("🔍 Multispectral Band Explorer (All 13 Bands)", expanded=False):
            band_idx = st.selectbox(
                "Select a reconstructed Sentinel-2 band to inspect:",
                options=list(range(13)),
                format_func=lambda i: f"Band {i+1}: {S2_BANDS_13[i]}",
            )
            band_data = result.reconstructed_multispectral[band_idx]
            b_norm = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)
            st.image(
                (b_norm * 255.0).astype(np.uint8),
                caption=f"Reconstructed {S2_BANDS_13[band_idx]} (Normalized Greyscale)",
                use_container_width=True,
            )
            st.caption(f"Reflectance Range: Min = {band_data.min():.5f}, Max = {band_data.max():.5f}, Mean = {band_data.mean():.5f}")

        # Download Options
        st.markdown("### Export Outputs")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            npz_bytes = export_npz_bytes(result.reconstructed_multispectral, "reconstructed_13bands")
            st.download_button(
                "📥 Download 13-Band Multispectral Output (.npz)",
                data=npz_bytes,
                file_name="geoclear_reconstructed_s2_13bands.npz",
                mime="application/octet-stream",
            )
        with dcol2:
            png_bytes = export_png_bytes(result.reconstructed_rgb_preview)
            st.download_button(
                "📥 Download Reconstructed True-Color RGB Preview (.png)",
                data=png_bytes,
                file_name="geoclear_reconstructed_preview.png",
                mime="image/png",
            )

        # Step 8: Scientifically Valid Evaluation
        st.markdown("---")
        st.header("Step 8: Evaluation & Verification Assessment")

        if ref_for_eval is not None:
            st.markdown("#### Quantitative Error & Similarity Metrics (vs. Clear Optical Reference for Evaluation)")
            st.caption("⚠️ **Methodological Notice**: A clear optical scene from a nearby sensing date is an empirical proxy, not an absolute simultaneous ground truth. Surface phenology, moisture, and vegetation state evolve over time.")
            # Sanity checks before computing metrics: dimensions, finite coverage, and scale
            if ref_for_eval.shape != result.reconstructed_multispectral.shape:
                st.warning(
                    f"⚠️ **Spatial Dimension Mismatch**: Reference array shape {ref_for_eval.shape} does not match "
                    f"reconstructed shape {result.reconstructed_multispectral.shape}. "
                    "Quantitative evaluation cannot be computed and is omitted to maintain scientific validity."
                )
            elif not np.isfinite(ref_for_eval).all():
                nan_count = int(np.isnan(ref_for_eval).sum())
                inf_count = int(np.isinf(ref_for_eval).sum())
                st.warning(
                    f"⚠️ **Non-Finite Pixels Detected**: Reference optical array contains {nan_count} NaNs and {inf_count} Infs. "
                    "Quantitative evaluation cannot be reliably computed on unmasked missing pixels."
                )
            else:
                try:
                    # Normalize reference to [0.0, 1.0] if given in DN [0, 10000]
                    ref_norm = np.clip(ref_for_eval, 0.0, 10000.0) / 10000.0 if ref_for_eval.max() > 1.5 else ref_for_eval
                    metrics = compute_evaluation_metrics(result.reconstructed_multispectral, ref_norm)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Multispectral MAE", f"{metrics.mae:.4f}")
                    m2.metric("Multispectral RMSE", f"{metrics.rmse:.4f}")
                    m3.metric("True-Color RGB PSNR", f"{metrics.rgb_psnr:.2f} dB")
                    m4.metric("True-Color RGB SSIM", f"{metrics.rgb_ssim:.4f}")

                    with st.expander("Detailed Per-Band Errors (MAE & RMSE)", expanded=False):
                        st.table({
                            "Band": list(metrics.per_band_mae.keys()),
                            "MAE": list(metrics.per_band_mae.values()),
                            "RMSE": list(metrics.per_band_rmse.values()),
                        })

                    # If local run: update manifest with metrics
                    if saved_paths and "manifest" in saved_paths:
                        try:
                            with open(saved_paths["manifest"], "r", encoding="utf-8") as f:
                                m_data = json.load(f)
                            m_data["evaluation"] = {
                                "reference_provided": True,
                                "metrics": {
                                    "multispectral_mae": metrics.mae,
                                    "multispectral_rmse": metrics.rmse,
                                    "rgb_psnr_db": metrics.rgb_psnr,
                                    "rgb_ssim": metrics.rgb_ssim,
                                    "per_band_mae": metrics.per_band_mae,
                                    "per_band_rmse": metrics.per_band_rmse,
                                },
                            }
                            with open(saved_paths["manifest"], "w", encoding="utf-8") as f:
                                json.dump(m_data, f, indent=2)
                        except Exception:
                            pass
                except Exception as exc:
                    st.error(f"Error computing reference metrics: {exc}")
        else:
            st.markdown(
                """
                <div class="disclaimer-box">
                    <b>Quantitative Metrics Unavailable:</b> No clear optical reference for evaluation was provided. 
                    Reference-dependent metrics (MAE, RMSE, PSNR, SSIM, SAM) cannot be scientifically computed 
                    and are not fabricated. A visually clear RGB image does not alone establish radiometric surface accuracy.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="info-box">
                <b>Scientific Disclaimer:</b> Outputs are model-generated optical estimates informed by SAR and optical inputs.
                They must not be represented as direct optical measurements. SAR backscatter informs structure and texture but does not directly observe optical color.
                Clear optical references from nearby sensing dates represent temporal evaluation proxies, not simultaneous ground truth.
            </div>
            """,
            unsafe_allow_html=True,
        )
