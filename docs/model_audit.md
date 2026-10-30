# GeoClear — External Pretrained Model Integration Audit (Phase 1)

**Audit Date**: October 2026  
**Auditor**: Antigravity AI  
**Scope**: In-depth inspection of the four external candidate model repositories located under `models/` (`CloudGAN`, `DSen2-CR`, `DiffCR`, `ECRformer`). The existing custom U-Net pipeline under `models/Unet` is excluded from modification, refactoring, or integration during this phase.

---

## Executive Summary & Comparison Matrix

| Property | CloudGAN | DSen2-CR | DiffCR | ECRformer |
| :--- | :--- | :--- | :--- | :--- |
| **Origin & Year** | Jerry Schonenberg (2022) | Meraner et al., ISPRS (2020) | Zou et al., IEEE TGRS (2024) | Zhang et al., ISPRS (2026) |
| **License** | MIT (code) / Apache 2.0 (SN-PatchGAN) | GNU GPL v3.0 | Unspecified / No License | Unspecified / No License |
| **Primary Framework** | TensorFlow 1.15 (Graph/Session) | Keras 2.2.4 (TF 1.15 backend) | PyTorch (torch >= 1.6) | PyTorch (torch >= 2.0) |
| **Python Compatibility** | Python 3.6.13 | Python 3.7 | Python 3.8 – 3.13 | Python 3.10 – 3.13 |
| **Pretrained Checkpoints Present?** | ❌ **No local weights** (external GDrive) | ❌ **No local weights** (external GDrive) | ✅ **Yes: 2 checkpoints** (`diffcr_new.pth`, `diffcr_old.pth`) | ✅ **Yes: 2 checkpoints** (`full.ckpt`, `light.ckpt`) |
| **Input Modalities** | Single Optical (RGB) | Multimodal (Optical + SAR) | Multi-temporal Optical (3x RGB) | Multimodal (Optical + SAR) |
| **Sensors & Channels** | 3 channels (RGB) | 15 channels (13 S2 + 2 S1 SAR) | 9 channels (3 timestamps × 3 RGB) | 15 channels (2 S1 SAR + 13 S2) |
| **Channel Ordering** | RGB (H, W, 3) | `[S2 (13), S1 (2)]` (15, H, W) | `[T0 (3), T1 (3), T2 (3)]` (9, H, W) | `[S1 (2), S2 (13)]` (15, H, W) |
| **Required Dimensions** | Strictly 256 × 256 | 128 × 128 or 256 × 256 | 256 × 256 (or multiples of 16) | 256 × 256 (or multiples of 16/32) |
| **Auxiliary Inputs Required** | Auto-generated or user mask | None for test inference | None (3 temporal images required) | None |
| **Inference Mechanism** | Feedforward Inpainting GAN | Feedforward Deep ResNet | 20-step DPM-Solver++ Diffusion | Feedforward Decoupled Transformer |
| **Output Type & Format** | 3-channel RGB uint8 PNG | 13-band Sentinel-2 reflectance | 3-channel RGB float [-1, 1] / PNG | 13-band Sentinel-2 reflectance [0, 1] |
| **Execution Feasibility in Current Env** | ❌ Blocked (TF 1.15 / tf.contrib) | ❌ Blocked (Keras 2.2 / TF 1.15 / no weights) | ⚠️ Partial (Code bug in imports + diffusion CPU latency) | ✅ **High** (Verified PyTorch weights & fast feedforward) |
| **Audit Status** | **Audited — Incompatible** | **Audited — Incompatible** | **Audited — Feasible with Fixes** | **Audited — Recommended for Phase 4** |

---

## 1. Candidate Audit: ECRformer

### 1.1 Repository Metadata & License
- **Original Paper**: *"ECRformer: An efficient cloud removal Transformer with semantic-decoupled learning for multimodal satellite imagery"*, ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 237, pp. 323–338, 2026.
- **Authors**: Zaiyan Zhang, Jie Li, Yuanqi Liang, Jining Yan, Yi Xiao, Xin Su, Qiangqiang Yuan (Wuhan University, CUG, Zhengzhou University).
- **Codebase Path**: `models/ECRformer/ECRformer-master`
- **License**: Unspecified (No `LICENSE` file present in the repository).

### 1.2 Framework & Dependency Environment
- **Framework**: PyTorch >= 2.0. The repository uses `pytorch-lightning >= 2.0` for training and testing loops.
- **Python Version**: Documented for Python >= 3.10 (tested and compatible with current environment Python 3.13).
- **Dependencies (`requirements.txt`)**:
  - `torch>=2.0`, `pytorch-lightning>=2.0`, `timm`, `einops`, `numpy`, `tensorboard`, `natsort`, `tqdm`, `matplotlib`, `Pillow`, `lpips`, `rasterio`.
- **Environment Findings**:
  - `torch 2.12.1+cpu`, `einops 0.8.2`, `rasterio 1.5.0`, `numpy 2.2.6`, `Pillow 11.2.1` are active in the local environment.
  - The model core (`ECRformerModel` in `models/ecrformer_model.py`) is standard PyTorch `nn.Module`.
  - While the official training scripts import `pytorch-lightning` and `timm`, the inference core requires only `torch`, `einops`, and small utility functions (`to_2tuple`, `trunc_normal_`, `DropPath`). An isolated adapter can run inference without forcing heavy global dependencies.

### 1.3 Pretrained Checkpoints
Confirmed present in the repository under `models/ECRformer/ECRformer-master/pretrained/`:
1. `full.ckpt` (45,682,805 bytes / ~45.7 MB): Full ECRformer model (`features_start=48`, `num_blocks=[2, 3, 2, 2]`, `num_refine=4`, ~11.29M parameters).
2. `light.ckpt` (15,176,872 bytes / ~15.2 MB): ECRformer-Light variant (`features_start=32`, `num_blocks=[2, 2, 1, 1]`, `num_refine=2`, ~3.70M parameters).
- **Inspection Result**: Both checkpoints were verified via `torch.load(..., map_location='cpu')`. They contain complete model state dictionaries under the `'state_dict'` key with weights prefixed by `'net.'` and SDFL projection heads (`down_proj`, `up_proj`).

### 1.4 Input Modalities, Sensors, Bands & Channel Ordering
- **Modality**: Multimodal Optical + Synthetic Aperture Radar (SAR).
- **Sensors**:
  - **Sentinel-1 SAR**: 2 polarizations: `[VV, VH]`.
  - **Sentinel-2 Multispectral Optical**: 13 spectral bands: `[B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12]`.
- **Channel Count & Ordering**:
  - Total channels: 15.
  - Required concatenation order: `torch.cat([SAR, cloudy_optical], dim=1)`
    - Channels 0–1: Sentinel-1 SAR (VV, VH).
    - Channels 2–14: Sentinel-2 Optical (13 bands in standard Sentinel-2 order).
- **Decoupled Architecture**: ECRformer’s stem (`DecoupledEncoder`) specifically takes `in_ch_list=[2, 13]` to decouple SAR microwave features from optical multispectral features before multimodal cross-attention.

### 1.5 Dimensions, Spatial Resolution & Registration
- **Spatial Dimensions**: Native patch size is 256 × 256 (SEN12MS-CR standard) or 128 × 128 crops. Operates fully convolutionally/window-wise on any dimensions divisible by 32.
- **Spatial Resolution**: 10 meters ground sampling distance (GSD). Sentinel-2 bands at 20m and 60m are pre-resampled to 10m to align with 10m bands.
- **Registration**: SAR (Sentinel-1) and optical (Sentinel-2) imagery must be strictly co-registered on the same geographic grid.

### 1.6 Normalization & Preprocessing
Verified from `data/sen12mscr_dataset.py`:
- **Sentinel-1 SAR (VV, VH)**:
  - Values in dB.
  - Clipped: `s1 = np.clip(s1, -25.0, 0.0)`
  - Scaled to `[0, 1]`: `(s1 - (-25.0)) / (0.0 - (-25.0)) = (s1 + 25.0) / 25.0`
- **Sentinel-2 Cloudy Optical (13 bands)**:
  - Raw digital numbers / surface reflectance in `[0, 10000]`.
  - Clipped: `s2 = np.clip(s2, 0.0, 10000.0)`
  - Scaled to `[0, 1]`: `s2 / 10000.0`
- **NoData & NaN Handling**:
  - `np.nan_to_num(x, copy=False)` to eliminate invalid or nodata pixels.

### 1.7 Auxiliary Inputs
- **Cloud Masks**: Not required as an input. The network recovers cloudy areas directly using multimodal attention.
- **Ground Truth**: Only required when computing quantitative evaluation metrics (RMSE, MAE, PSNR, SSIM, SAM, LPIPS). Never required for operational inference.

### 1.8 Output Characteristics
- **Bands**: 13 bands corresponding to cloud-free Sentinel-2 surface reflectance.
- **Range**: `[0, 1]` normalized reflectance (multiply by 10000 to recover original reflectance integer scale).
- **RGB Visualization**: Extracted using bands `[3, 2, 1]` (B4-Red, B3-Green, B2-Blue), clipped with brightness multiplier 3.0: `np.clip(rgb * 3.0, 0.0, 1.0) * 255.0` (uint8).

### 1.9 Official Inference Commands
```bash
python test.py --config ecrformer --ckpt-path pretrained/full.ckpt --gpu -1 --export-format both
```

### 1.10 Verified Limitations & Assumptions
- Cannot run on optical data alone; requires co-registered Sentinel-1 SAR imagery.
- Requires all 13 Sentinel-2 bands; cannot operate on 3-band RGB imagery alone without documented band synthesis.

---

## 2. Candidate Audit: DiffCR

### 2.1 Repository Metadata & License
- **Original Paper**: *"DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal From Optical Satellite Images"*, IEEE Transactions on Geoscience and Remote Sensing (TGRS), Vol. 62, 2024.
- **Authors**: Xuechao Zou, Kai Li, Junliang Xing, Yu Zhang, Shiying Wang, Lei Jin, Pin Tao (Tsinghua University).
- **Codebase Path**: `models/DiffCR/DiffCR-main`
- **License**: Unspecified (No `LICENSE` file present in the repository).

### 2.2 Framework & Dependency Environment
- **Framework**: PyTorch (`torch>=1.6`), `torchvision`, `numpy`, `pandas`, `tqdm`, `scipy`, `opencv-python`.
- **Tested Dependencies**: Compatible with modern PyTorch.

### 2.3 Pretrained Checkpoints
Confirmed present in the repository under `models/DiffCR/DiffCR-main/pretrained/`:
1. `diffcr_new.pth` (91,863,129 bytes / ~91.8 MB): Trained on `Sen2_MTC_New` dataset.
2. `diffcr_old.pth` (91,863,129 bytes / ~91.8 MB): Trained on `Sen2_MTC_Old` dataset.
- **Inspection Result**: Verified via `torch.load(..., map_location='cpu')`. Checkpoints contain the full state dictionary for the `denoise_fn` NAFNet UNet.

### 2.4 Input Modalities, Sensors, Bands & Channel Ordering
- **Modality**: **Multi-temporal Optical Imagery**.
- **Crucial Architectural Requirement**: DiffCR does **not** take SAR imagery. It strictly requires **three temporal acquisitions** of cloudy optical images of the exact same scene (`img0`, `img1`, `img2`).
- **Channels**: 3 optical images × 3 channels (RGB) = 9 conditioning channels total.
- **Conditioning Tensor**: `torch.cat([img0, img1, img2], dim=1)` -> shape `(B, 9, H, W)`.
- During diffusion, the model samples a noisy target image `y_t` (3 channels) and concatenates it with the condition: `torch.cat([cond, y_t], dim=1)` -> total 12 input channels into `denoise_fn`.

### 2.5 Dimensions, Spatial Resolution & Registration
- **Spatial Dimensions**: 256 × 256 pixels.
- **Registration**: All three temporal images must be co-registered.

### 2.6 Normalization & Preprocessing
Verified from `data/dataset.py`:
- Input images loaded as TIFF or PNG.
- Scaled: `image = image / 10000.0`
- Normalized: `(image - 0.5) / 0.5` mapping values into `[-1.0, 1.0]`.

### 2.7 Known Bugs & Execution Issues (Confirmed by Audit)
1. **Upstream Code Bug in `network_x0_dpm_solver.py`**:
   - Line 47 executes: `from .ours.ours_double_encoder_splitcaCond_splitcaUnet import UNet`
   - However, the actual file on disk in `models/ours/` is named `nafnet_double_encoder_splitcaCond_splitcaUnet.py`.
   - Running official commands crashes immediately with:
     ```
     ModuleNotFoundError: No module named 'models.ours.ours_double_encoder_splitcaCond_splitcaUnet'
     ```
   - *Audit Fix*: We verified that importing directly from `models.ours.nafnet_double_encoder_splitcaCond_splitcaUnet` successfully loads the checkpoint weights into `UNet` and runs a forward pass.
2. **Diffusion Inference Latency**:
   - Uses DPM-Solver++ with 20 reverse diffusion steps. On CPU-only environments, executing 20 passes of a 91 MB deep UNet per image takes noticeably longer than single-pass feedforward models.
3. **Data Availability Constraint**:
   - The user must provide three separate cloudy temporal optical images for the same location. Single-image optical uploads cannot satisfy the model's condition channels.

---

## 3. Candidate Audit: DSen2-CR

### 3.1 Repository Metadata & License
- **Original Paper**: *"Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion"*, ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 166, pp. 333–346, 2020. Best Paper / Helava Award.
- **Authors**: Andrea Meraner, Patrick Ebel, Xiao Xiang Zhu, Michael Schmitt (DLR / TUM).
- **Codebase Path**: `models/DSen2-CR/dsen2-cr-main`
- **License**: **GNU General Public License v3.0 (GPLv3)**.

### 3.2 Framework & Dependency Environment
- **Primary Framework**: Keras 2.2.4 with TensorFlow 1.15.0 backend, Python 3.7.
- **Secondary PyTorch Script**: `Code/dsen2cr_pytorch_model.py` provides a standalone PyTorch implementation of `ResnetStackedArchitecture(nn.Module)`.

### 3.3 Pretrained Checkpoints
- **Local Checkpoints**: **NONE**.
- **External Links**: Checkpoints are hosted on external Google Drive links in Keras `.h5` format. No PyTorch `.pth` checkpoint was ever published or included.

### 3.4 Input Modalities, Sensors, Bands & Channel Ordering
- **Modality**: Multimodal Optical + SAR.
- **Sensors**: Sentinel-2 (13 bands) + Sentinel-1 SAR (2 bands: VV, VH).
- **Total Channels**: 15 channels.
- **Channel Ordering**: In PyTorch model: `torch.cat([cloudy_MS, SAR], dim=1)` (13 optical + 2 SAR = 15 channels).

### 3.5 Normalization & Preprocessing
Verified from `Code/tools/dataIO.py`:
- Sentinel-2 optical: clipped to `[0, 10000]`, divided by scale factor 2000 (`[0, 5]` range).
- Sentinel-1 SAR: clipped to `[-25.0, 0.0]` (VV) and `[-32.5, 0.0]` (VH), scaled to `[0, 2]`.

### 3.6 Audit Verdict
- **Blocked**: Missing local pretrained weights. Main codebase is locked to legacy Keras 2.2 / TF 1.15. GPLv3 copyleft restrictions apply.

---

## 4. Candidate Audit: CloudGAN

### 4.1 Repository Metadata & License
- **Original Paper**: *"CloudGAN: Detecting and Removing Clouds from RGB-images using Image Inpainting"*, Jerry Schonenberg, 2022.
- **Codebase Path**: `models/CloudGAN/CloudGAN-main`
- **License**:
  - Repository root: MIT License (Jerry Schonenberg 2022).
  - Subdirectory `cloud_removal/LICENSE`: Apache License 2.0 (SN-PatchGAN / Jiahui Yu).

### 4.2 Framework & Dependency Environment
- **Framework**: TensorFlow 1.15.0 (`tensorflow-gpu==1.15`).
- **Python Version**: Documented strictly for Python 3.6.13.
- **Critical Incompatibilities**:
  - Code uses `tf.compat.v1.Session`, static graphs, and `tf.contrib.framework.load_variable`.
  - `tf.contrib` was permanently removed from TensorFlow 2.0+ and cannot be run in modern environments.
  - Requires `neuralgym` from a legacy Git commit.

### 4.3 Pretrained Checkpoints
- **Local Checkpoints**: **NONE**.
- **External Links**: Checkpoints for the Auto-Encoder (`checkpoint.h5`), U-Net, and SN-PatchGAN must be downloaded from external Google Drive links.

### 4.4 Input Modalities & Preprocessing
- **Modality**: Optical RGB only (3 channels).
- **Dimensions**: Strictly fixed at 256 × 256.
- **Auxiliary Inputs**: Autoencoder generates a binary cloud mask, or user supplies a 256×256×3 mask.
- **Approach**: Pure 2D computer-vision inpainting. Hallucinates missing texture rather than recovering physical surface information from radar or multi-temporal passes.

### 4.5 Audit Verdict
- **Blocked**: Missing local weights, deprecated TF1 / `tf.contrib` codebase, incompatible with modern Python / PyTorch environments.

---

## 5. Comparative Evaluation & Risk Assessment

| Assessment Dimension | ECRformer | DiffCR | DSen2-CR | CloudGAN |
| :--- | :--- | :--- | :--- | :--- |
| **Local Checkpoint Usability** | **Confirmed** (Full & Light `.ckpt` verified) | **Confirmed** (New & Old `.pth` verified) | ❌ None (external GDrive only) | ❌ None (external GDrive only) |
| **Dependency Health** | **Clean PyTorch** (no legacy C-extensions) | **Clean PyTorch** (minor file path bug) | ❌ Legacy Keras/TF 1.15 | ❌ Deprecated TF1.15 / tf.contrib |
| **Input Feasibility** | Multimodal: S1 SAR (2) + S2 Optical (13) | Multi-temporal: 3 × RGB Optical | Multimodal: S1 SAR (2) + S2 Optical (13) | Single-image RGB (256x256) |
| **Inference Speed** | **Fast** (Single forward pass) | **Slow** (20 diffusion sampling steps) | Fast (feedforward) | Medium (GAN inpainting) |
| **Scientific Soundness** | Physically grounded multimodal fusion | Optical temporal redundancy | Physically grounded multimodal fusion | 2D image hallucination |
| **Licensing Risk** | Low (Academic code) | Low (Academic code) | Medium (GPLv3 copyleft) | Low (MIT + Apache 2.0) |

---

## 6. Phase 1 Recommendation: Model to Integrate First

### Primary Recommendation: **ECRformer**

**Rationale**:
1. **Verified Checkpoints on Disk**: Contains two verified PyTorch checkpoints (`full.ckpt` and `light.ckpt`) directly in the workspace.
2. **Modern Architecture**: Pure PyTorch transformer design (`torch.nn.Module`) compatible with Python 3.10–3.13 and CPU/GPU execution without legacy frameworks.
3. **True Multimodal Earth Observation**: Integrates Sentinel-1 SAR microwave penetration with Sentinel-2 optical multispectral data, directly aligning with GeoClear's mission.
4. **Fast & Deterministic Inference**: Single feedforward forward pass without the latency or stochasticity of iterative diffusion models.
5. **Clear Adapter Encapsulation**: Can be encapsulated into an adapter that handles S1/S2 normalization and band extraction cleanly without pulling in PyTorch Lightning into the dashboard runtime.

### Secondary Candidate for Follow-up: **DiffCR**
- **Role**: Optical multi-temporal specialist.
- **Integration Prerequisite**: Patch the internal import mismatch (`ours_double_encoder...` -> `nafnet_double_encoder...`) and ensure the user workflow explicitly supports multi-temporal triplet input declarations.

---

## 7. Next Steps for Implementation (Phase 2 – Phase 4)

1. **Phase 2 — Model Registry & Compatibility Engine**:
   - Implement `GeoClearRegistry` storing metadata, modalities, sensors, bands, dimensions, and operational status for all audited models.
   - Implement `CompatibilityEngine` that filters models based strictly on declared user inputs (excluding models if SAR or multi-temporal optical is missing).
2. **Phase 3 — Dashboard User Journey**:
   - Dynamic UI allowing data declaration -> compatibility filtering -> model selection -> input upload -> validation -> inference -> valid metric computation.
3. **Phase 4 — ECRformer Adapter & Inference Verification**:
   - Build `ECRformerAdapter` to load `light.ckpt` / `full.ckpt`, validate 15-channel inputs, apply clipping/normalization, execute CPU/GPU inference, and produce calibrated outputs and RGB previews.
   - Implement automated unit tests for registry, compatibility, preprocessing, and inference.
