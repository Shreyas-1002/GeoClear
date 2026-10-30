# GeoClear — Phase 4 Real Satellite Patch Ingestion Audit

**Audit Date**: October 2026  
**Auditor**: Antigravity AI  
**Scope**: Verification of data product specifications, preprocessing fidelity, model checkpoints, spatial/spectral constraints, and ingestion architecture for real Sentinel-1/Sentinel-2 imagery prior to Phase 4 implementation.

---

## 1. ECRformer Input & Output Specifications

| Property | Value / Specification | Reference |
| :--- | :--- | :--- |
| **Merged Input Tensor** | `(1, 15, H, W)` float32 | `models/ecrformer_model.py:DecoupledEncoder` |
| **Input Split** | Branch 0: 2 channels (SAR); Branch 1: 13 channels (Optical) | `DecoupledEncoder(in_ch_list=[2, 13])` |
| **SAR Channels** | Exactly 2 channels: `[VV, VH]` in that exact order | `data/sen12mscr_dataset.py` |
| **SAR Physical Units** | Orthorectified $\sigma_0$ backscatter in decibels (dB), typical range $[-25.0, 0.0]$ dB | `process_SAR(..., method='default')` |
| **SAR Normalization** | $\text{clip}(x, -25.0, 0.0) \rightarrow \frac{x + 25.0}{25.0} \in [0.0, 1.0]$ | `rescale(img, -25, 0)` |
| **Optical Bands** | Exactly 13 bands in standard ESA Sentinel-2 L1C order: `[B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]` | `data/convert_sen12mscr_to_npz.py` |
| **Optical Physical Units**| Top-Of-Atmosphere (TOA) Digital Numbers (DN) in $[0, 10000]$ (reflectance $\times 10^4$) | `data/sen12mscr_dataset.py` |
| **Optical Normalization**| $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0} \in [0.0, 1.0]$ | `rescale(img, 0, 10000)` |
| **Output Tensor** | `(1, 13, H, W)` float32 in $[0.0, 1.0]$ (reconstructed cloud-free optical reflectance) | `models/ecrformer_model.py:final_conv` |
| **Spatial Constraints** | $H$ and $W$ must be divisible by 32 (due to 3-stage `PixelUnshuffle(2)` downsampling and multiscale projections) | `models/ecrformer_model.py:downsampler` |

---

## 2. Satellite Collection Verification: TOA (`COPERNICUS/S2_HARMONIZED`) vs Surface Reflectance (`COPERNICUS/S2_SR_HARMONIZED`)

A critical scientific distinction exists between Google Earth Engine's Sentinel-2 collections:

1. **`COPERNICUS/S2_SR_HARMONIZED` (Level-2A Surface Reflectance)**:
   - Contains **12 spectral bands**: `B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12`.
   - **Does NOT contain Band 10 (Cirrus, 1375 nm)**. Band 10 is omitted from Level-2A products by the European Space Agency (ESA) because cirrus absorption in the lower atmosphere means virtually zero solar radiation at 1375 nm reaches the ground; it cannot be inverted to surface reflectance.
2. **`COPERNICUS/S2_HARMONIZED` (Level-1C Top-Of-Atmosphere Reflectance)**:
   - Contains **all 13 spectral bands**: `B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12`.
   - Scaled as integer DN with scale factor 10,000.

### Determination for ECRformer:
- **SEN12MS-CR was constructed using Sentinel-2 Level-1C Top-Of-Atmosphere imagery**, not Level-2A surface reflectance.
- The official ECRformer network expects **all 13 bands including Band 10** in the optical branch.
- **Decision**: The export workflow must use **`COPERNICUS/S2_HARMONIZED`** (Level-1C TOA). Exporting `COPERNICUS/S2_SR_HARMONIZED` would yield only 12 bands and fail ECRformer's 13-channel input assertion. Band 10 must not be fabricated.

---

## 3. Checkpoint & Preprocessing Verification

### Available Checkpoints:
1. `models/ECRformer/ECRformer-master/pretrained/light.ckpt` (15.2 MB, ~3.7M parameters).
   - Configuration: `features_start=32`, `num_blocks=[2, 2, 1, 1]`, `num_refine=2`.
   - Fully compatible with `ECRformerAdapter(variant='light')`.
2. `models/ECRformer/ECRformer-master/pretrained/full.ckpt` (45.7 MB, ~11.3M parameters).
   - Configuration: `features_start=48`, `num_blocks=[2, 3, 2, 2]`, `num_refine=4`.
   - Fully compatible with `ECRformerAdapter(variant='full')`.

### Fidelity of `ecrformer_adapter.py`:
- The adapter implements the exact clipping and scaling formulas:
  - SAR: `(np.clip(sar, -25.0, 0.0) + 25.0) / 25.0`
  - Optical: `np.clip(optical, 0.0, 10000.0) / 10000.0`
- Nan/Inf sanitization via `np.nan_to_num` aligns directly with `sen12mscr_dataset.py`.
- RGB visualization uses True-Color bands $(B4, B3, B2) \rightarrow (3, 2, 1)$ with official brightness multiplier $3.0$ matching `DEFAULT_PNG_BRIGHTNESS = 3.0` in `test.py`.

---

## 4. Installed Dependencies & Execution Environment

- **Python**: 3.13.2 (Windows amd64).
- **PyTorch**: 2.12.1+cpu (no CUDA available; inference runs on CPU).
- **Geospatial**: `rasterio` 1.4.3 with in-process `PROJ_LIB` isolation (preventing PostgreSQL/PostGIS `proj.db` conflicts).
- **Image Processing**: `scikit-image` 0.25.2, `Pillow` 11.1.0, `numpy` 2.2.4.
- **UI**: `streamlit` 1.43.2.
- **Earth Engine API**: `ee` is not installed globally. To uphold the non-negotiable safeguard (*"Do not install or upgrade global dependencies"*), the reproducible GEE export workflow is provided via **Google Earth Engine Code Editor JavaScript** (requiring only a web browser) with an optional standalone Python script for users with an authenticated virtual environment.

---

## 5. Licensing & Scope Safeguards

- **ECRformer License**: No explicit LICENSE file in repository. Academic evaluation only; commercial permissions unstated.
- **Legacy U-Net Pipeline**: Completely quarantined under `models/Unet/`; untouched.
- **External Candidates (DiffCR, DSen2-CR, CloudGAN)**: Remain non-operational and blocked per Phase 1 audit.
