# GeoClear — Google Earth Engine (GEE) Real Patch Export Guide

This directory contains reproducible workflows for preparing and exporting genuine, co-registered Sentinel-1 SAR and Sentinel-2 optical imagery patches for inference in GeoClear.

---

## 🛰️ 1. Scientific & Radiometric Provenance

### Why `COPERNICUS/S2_HARMONIZED` (Level-1C TOA) is Required
ECRformer was trained on the **SEN12MS-CR** benchmark dataset (Ebel et al., 2021; Schmitt et al., 2019).
- **SEN12MS-CR uses Sentinel-2 Level-1C (L1C) Top-Of-Atmosphere (TOA) imagery**:
  - Contains **all 13 spectral bands**: `['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']`.
  - Values represent calibrated **TOA reflectance scaled by 10,000 (DN)**. Dividing by $10,000.0$ and clipping to $[0.0, 1.0]$ scales values to the $[0, 1]$ reflectance range expected by ECRformer. These are radiometrically calibrated and orthorectified TOA values, **not** raw instrument detector digital counts.
- **Level-2A Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`) omits Band 10 (Cirrus, 1375 nm)**:
  - Band 10 is omitted in Level-2A processing because severe water-vapour atmospheric absorption prevents 1375 nm radiation from reaching the surface; it is an atmospheric cirrus detection band.
  - Using `S2_SR_HARMONIZED` provides only 12 bands and breaks ECRformer's 13-band architecture.
  - Therefore, you **must use `COPERNICUS/S2_HARMONIZED`** to match ECRformer's verified preprocessing.

### Sentinel-1 SAR Backscatter Semantics
- **Collection**: `COPERNICUS/S1_GRD` (Interferometric Wide swath, Ground Range Detected).
- **Polarizations**: `['VV', 'VH']` in that exact order.
- **Radiometric Units**: Backscatter coefficient $\sigma^0$ in **decibels (dB)** with radiometric calibration and terrain correction already applied by Earth Engine.
- **Model Normalization**: ECRformer clips values to $[-25.0, 0.0]\text{ dB}$ and scales via $(x + 25.0) / 25.0$ to $[0.0, 1.0]$. No additional logarithmic conversion is performed.

---

## 📐 2. Exact Shared Pixel Grid & Resampling Policy

To guarantee mathematical co-registration without silent spatial distortions or grid misalignments:

1. **Target Grid Definition**:
   - The native projected Coordinate Reference System (CRS, typically UTM) and 10-meter spacing are extracted directly from the target Sentinel-2 Band 4 (Red, native 10m).
   - The upper-left origin $(xMin, yMax)$ is snapped to the nearest 10-meter boundary in projected coordinates.
2. **Deterministic Planar Bounding Polygon**:
   - An exact $2560\text{ m} \times 2560\text{ m}$ bounding polygon is constructed using `geodesic: false` in the projected CRS.
   - At 10m resolution, this defines a deterministic grid of exactly **$256 \times 256$ pixels**.
3. **Earth Engine Parameter Exclusivity Compliance**:
   - In Earth Engine, specifying `crsTransform` together with `scale` or `dimensions` causes an error.
   - The script specifies **`crs`**, **`crsTransform: [10, 0, xMin, 0, -10, yMax]`**, and **`region`**, leaving `scale` and `dimensions` omitted. Earth Engine exports an identical $256 \times 256$ raster for all tasks.
4. **Resampling Policy & Resolution Limitations**:
   - Both optical and SAR data are resampled to the common 10m grid using **bilinear interpolation**.
   - **Important Limitation**: Sentinel-2 bands have differing native resolutions:
     - 10m bands: `B2, B3, B4, B8` (native detail).
     - 20m bands: `B5, B6, B7, B8A, B11, B12` (smoothly interpolated).
     - 60m bands: `B1, B9, B10` (smoothly interpolated).
     - Resampling coarser bands to 10m does **not** synthesize true sub-resolution physical detail. In particular, Band 10 (Cirrus) has ~60m native resolution, so cirrus features appear smooth.

---

## 🔍 3. AOI Cloud Assessment: `s2cloudless` vs QA60 Historical Gap

### The QA60 Availability Gap (2022–2023)
- Beginning **January 25, 2022**, ESA transitioned operational Sentinel-2 processing to **Processing Baseline 04.00**.
- This change altered how cloud mask vectors were generated. In Earth Engine's `COPERNICUS/S2_HARMONIZED` and `COPERNICUS/S2_SR_HARMONIZED` archives, the legacy `QA60` band for 2022 and 2023 acquisitions is frequently **unpopulated (all zeros)**.
- **Do not assume `QA60` is valid simply because the band exists.** Treating an unpopulated `QA60` band as ground truth will erroneously report 0% cloud cover over completely overcast scenes.

### Robust Cloud Assessment Method
1. **Joining `COPERNICUS/S2_CLOUD_PROBABILITY`**:
   - The script joins the Sentinel Hub `s2cloudless` product (`COPERNICUS/S2_CLOUD_PROBABILITY`) by matching the scene's unique `system:index`.
   - The join is checked explicitly. If no matching cloud probability record exists, the script returns an explicit **`Unknown`** status and logs a warning. Missing data is **never** silently treated as 0% cloud cover.
2. **Distinguishing Continuous Probability from Binary Masks**:
   - **Mean Cloud Probability**: The continuous spatial average of pixel probability values ($0\text{--}100\%$) across the patch.
   - **Binary Cloud Mask**: Pixels with $P(\text{cloud}) \ge \text{threshold}$ (configurable via `CONFIG.cloud_probability_threshold`, default $40\%$).
   - Both continuous probability and binary classified coverage are recorded distinctly in console outputs and in `manifest.json`.
3. **Reference Scene Policy**:
   - The same `s2cloudless` assessment is executed for the optional reference scene.
   - If reference cloud assessment is `Unknown`, a warning states that reference clarity could not be automatically confirmed.
   - If reference binary cloud cover is $> 0\%$, it is flagged as `is_strict_ground_truth: false`.
   - Even a clear reference is an **unaligned temporal proxy**, not simultaneous ground truth; seasonal phenology, moisture, and harvesting create real spectral differences.
4. **Valid-Pixel Coverage Checks**:
   - Inspects unmasked pixel coverage for both optical and SAR data over the patch.
   - If either sensor has $< 100\%$ valid coverage (e.g. crossing a swath edge or detector gap), a prominent warning is printed. ECRformer requires fully observed, finite numeric arrays.

---

## 🚀 4. Step-by-Step Export Instructions

### Step 1: Open Google Earth Engine Code Editor
1. Open [https://code.earthengine.google.com/](https://code.earthengine.google.com/) in your browser.
2. Sign in with your Earth Engine account.

### Step 2: Paste and Configure
1. Open [`scripts/gee/export_patch.js`](export_patch.js) and paste its entire contents into the center editor pane.
2. In lines 45–75 (`CONFIG`), adjust if desired:
   - `center_lon`, `center_lat`: Geographic coordinates of your area of interest.
   - `s2_target_start`, `s2_target_end`: Date range for cloudy Sentinel-2 scenes.
   - `sar_window_days`: Temporal search window for Sentinel-1 IW acquisitions (default: 5 days).
   - `cloud_probability_threshold`: Threshold for classifying cloud pixels (default: $40\%$).
   - `export_reference`: Boolean flag (`true`/`false`) to export or skip the comparison optical scene.

### Step 3: Run & Inspect
1. Click the **Run** button at the top of the editor.
2. Review the **Console** tab (top-right panel):
   - Scene candidate counts.
   - Selected scene IDs and acquisition dates.
   - Temporal difference between optical and SAR passes.
   - AOI Cloud Assessment Status (`Valid` or `Unknown`).
   - AOI Mean Cloud Probability ($0\text{--}100\%$) and Classified Cloud Pixels ($\ge 40\%$).
   - Valid-pixel coverage percentages.
3. Review the interactive map layers:
   - Check `1. Target S2 RGB` and `2. Target S2 Cirrus (B10)` to visually confirm clouds over the red bounding box.
   - Inspect `3. Target Cloud Probability (0-100%)` and `4. Target Cloud Mask` to verify `s2cloudless` detection over the patch.
   - Check `5. Target S1 SAR` to confirm radar surface structure.

### Step 4: Export to Google Drive
1. Switch to the **Tasks** tab in the top-right panel.
2. Click **Run** next to:
   - `s1_sar`: Sentinel-1 VV/VH GeoTIFF ($256 \times 256$).
   - `s2_cloudy`: Sentinel-2 13-band TOA GeoTIFF ($256 \times 256$).
   - `s2_reference`: Optional clean Sentinel-2 comparison GeoTIFF ($256 \times 256$).
3. In the export dialogs, confirm the destination folder (default: `GeoClear_Real_Patch`) and click **Run**.
4. Wait for the tasks to finish (typically 1–2 minutes).

### Step 5: Save Local Files for GeoClear Ingestion
1. Download the exported files from Google Drive into a local directory, for example:
   ```text
   D:\GeoClear\data\real_patch\
   ├── s1_sar.tif
   ├── s2_cloudy.tif
   └── s2_reference.tif   (optional)
   ```
2. Copy the JSON block printed in the GEE Console between the markers:
   ```text
   ================== 3. GEOCLEAR PROVENANCE MANIFEST ==================
   { ... }
   =====================================================================
   ```
   and save it as:
   ```text
   D:\GeoClear\data\real_patch\manifest.json
   ```

### Step 6: Ingest & Run in GeoClear Dashboard
1. Start GeoClear:
   ```powershell
   streamlit run app.py
   ```
2. In the sidebar, select:
   - Model: **ECRformer-Light**
   - Mode: **📁 Local Directory Ingestion**
   - Directory Path: `D:\GeoClear\data\real_patch`
3. GeoClear inspects the headers non-destructively, confirms 13 optical bands, 2 SAR bands, exact CRS alignment, and divisibility by 32, and displays provenance metadata from `manifest.json`.
4. Click **"🚀 Execute ECRformer-Light Inference"** to run local reconstruction and save outputs into `outputs/run_<timestamp>/`.
