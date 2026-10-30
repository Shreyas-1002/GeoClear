# GeoClear — Streamlit Dashboard User & Engineering Guide

This guide details the architecture, input formats, validation mechanics, and evaluation policies of the GeoClear Streamlit interactive dashboard (`app.py`).

---

## 1. User Workflow Walkthrough

### Step 1: Declare Available Data
The user specifies their available satellite assets using interactive UI controls:
- **Sensors:** Sentinel-2 Multispectral, Sentinel-1 SAR, and/or Generic RGB.
- **Spectral Bands:** All 13 Sentinel-2 bands (`B1-B12, B8A`), RGB only (`B4, B3, B2`), or None.
- **SAR Polarizations:** `VV`, `VH`, or both.
- **Temporal Passes:** Number of repeat passes (e.g. 1 for single-pass, $\ge 2$ for multi-temporal).
- **Ground-Truth Reference:** Checkbox indicating whether a cloud-free optical reference is available for evaluation.
- **Spatial Patch Dimensions:** $128 \times 128$, $256 \times 256$, $512 \times 512$, or arbitrary.

### Step 2: Compatibility Evaluation & Operational Readiness
The `CompatibilityEngine` evaluates all candidates from `ModelRegistry`:
- **Eligible Candidates:** Models whose physical input requirements match the declared data.
- **Excluded Candidates:** Models that cannot operate on the declared data, with human-readable explanations detailing missing sensors, channels, or temporal requirements.
- **Readiness Badges:** Distinguishes `Input Compatible` from `Executable` vs `Execution Blocked`.

### Step 3: Model Selection
- Only models that are **both input-compatible and executable** (`READY_FOR_INFERENCE`) can be selected for inference.
- **ECRformer-Light** is prioritized as the primary working operational path.
- Non-operational candidates (DiffCR, DSen2-CR, CloudGAN) display their specific architectural and dependency blockers.
- If ECRformer is selected, an advisory banner notes the authors' unstated repository licensing terms.

### Step 4 & 5: Upload Inputs & Validation
- **Uploaders:** Dedicated upload targets for Sentinel-1 SAR (2 bands) and Sentinel-2 Optical (13 bands), plus an optional target for the cloud-free reference.
- **Synthetic Test Generator:** A one-click button to load a realistic synthetic multi-band test patch (tagged as a demonstration) when real data files are not readily available.
- **Validation Engine (`geoclear/io.py`):**
  - Confirms exact channel counts (2 for SAR, 13 for Optical).
  - Asserts that spatial dimensions $H$ and $W$ are divisible by 32.
  - Verifies that all pixel values are finite numbers (rejects NaNs and Infs).
  - Cross-validates geospatial metadata (CRS, resolution, bounds) if GeoTIFFs are uploaded.
  - If array dimensions or CRS mismatch, inference execution is halted.

### Step 6: Inference Execution
- Calls the standalone adapter `ECRformerAdapter(variant="light", device="cpu")`.
- Normalizes SAR to $[0, 1]$ from dB $[-25, 0]$ and Optical to $[0, 1]$ from DN $[0, 10000]$.
- Runs single-pass encoder-decoder inference in PyTorch without altering upstream code or weights.

### Step 7: Results & Visualization
- **Three Side-by-Side Previews:** Input cloudy RGB preview, input SAR composite (`[VV, VH, VV/VH]`), and recovered optical RGB preview.
- **Telemetry Bar:** Displays latency in milliseconds, device used, reflectance value range, and finite check confirmation.
- **13-Band Multispectral Explorer:** Lets users inspect any of the 13 reconstructed Sentinel-2 bands in normalized greyscale with min/max/mean reflectance statistics.
- **Download Options:** One-click download of the complete 13-band reconstruction as `.npz` and the true-color RGB preview as `.png`.

### Step 8: Evaluation & Verification Assessment
- If a cloud-free reference is uploaded:
  - Calculates Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), Peak Signal-to-Noise Ratio (**PSNR**), and Structural Similarity (**SSIM**).
  - Provides a per-band MAE/RMSE error table.
- If no reference is provided:
  - Displays a scientific disclaimer explaining that quantitative error metrics cannot be computed without a reference. Metrics are **never fabricated**.

---

## 2. Input Specifications

### Sentinel-1 SAR
- **Format:** GeoTIFF or NumPy (`.npz` / `.npy`).
- **Channels:** 2 bands: `Band 1 = VV`, `Band 2 = VH`.
- **Units:** Decibels (dB), expected range $[-25.0, 0.0]$ dB.
- **Spatial Size:** $H \times W$, divisible by 32 (e.g. $128 \times 128$, $256 \times 256$).

### Sentinel-2 Optical
- **Format:** GeoTIFF or NumPy (`.npz` / `.npy`).
- **Channels:** 13 bands in SEN12MS-CR order:
  `[B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]`.
- **Units:** Digital Numbers (DN) $[0, 10000]$.
- **Spatial Size:** Identical $H \times W$ to SAR, divisible by 32.

---

## 3. Real-Data Verification Status

*Software execution and pipeline mechanics are fully verified using synthetic multi-band test patches. Real-data inference verification (`REAL_DATA_INFERENCE_VERIFIED`) remains pending until user-provided real satellite scenes from SEN12MS-CR or Copernicus Hub are processed.*
