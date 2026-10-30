# ECRformer Adapter Specification & Usage Guide

## Overview

The `ECRformerAdapter` (`geoclear.adapters.ecrformer_adapter.ECRformerAdapter`) provides an isolated, dependency-shimmed inference interface for the ECRformer model family (*Zhang et al., ISPRS J. PRS 2026*).

It directly loads the official pretrained checkpoints (`light.ckpt` and `full.ckpt`) and executes inference on CPU (and CUDA where available) without requiring `pytorch-lightning` or external `timm` installations.

---

## Licensing Advisory

> [!WARNING]
> **Unresolved Usage-Permission Status**  
> The original ECRformer repository contains no explicit `LICENSE` file. Under standard international copyright law, the lack of an explicit license means copyright remains fully with the authors (Zhang et al., Wuhan University). Unrestricted redistribution or commercial deployment cannot be assumed without explicit grant from the authors.

---

## Input Requirements & Channel Ordering

ECRformer requires **15 input channels** at 10m spatial resolution on a matching spatial grid (co-registered):

| Channel Index | Sensor | Band / Polarization | Native Scale | Normalization Formula | Normalized Range |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | Sentinel-1 SAR | **VV** | dB (decibels) | $\text{clip}(x, -25.0, 0.0) \rightarrow \frac{x + 25.0}{25.0}$ | $[0.0, 1.0]$ |
| `1` | Sentinel-1 SAR | **VH** | dB (decibels) | $\text{clip}(x, -25.0, 0.0) \rightarrow \frac{x + 25.0}{25.0}$ | $[0.0, 1.0]$ |
| `2` | Sentinel-2 Optical | **B1** (Coastal Aerosol) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `3` | Sentinel-2 Optical | **B2** (Blue) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `4` | Sentinel-2 Optical | **B3** (Green) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `5` | Sentinel-2 Optical | **B4** (Red) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `6` | Sentinel-2 Optical | **B5** (Vegetation Red Edge 1) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `7` | Sentinel-2 Optical | **B6** (Vegetation Red Edge 2) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `8` | Sentinel-2 Optical | **B7** (Vegetation Red Edge 3) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `9` | Sentinel-2 Optical | **B8** (NIR Broad) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `10` | Sentinel-2 Optical | **B8A** (NIR Narrow) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `11` | Sentinel-2 Optical | **B9** (Water Vapour) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `12` | Sentinel-2 Optical | **B10** (SWIR Cirrus) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `13` | Sentinel-2 Optical | **B11** (SWIR 1) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |
| `14` | Sentinel-2 Optical | **B12** (SWIR 2) | Digital Number (DN) | $\text{clip}(x, 0.0, 10000.0) \rightarrow \frac{x}{10000.0}$ | $[0.0, 1.0]$ |

### Spatial Dimensions
- Height ($H$) and Width ($W$) must match between SAR and Optical inputs.
- Minimum size: $32 \times 32$ pixels.
- Both dimensions must be divisible by $32$ (e.g., $128 \times 128$, $256 \times 256$, $512 \times 512$).

---

## Output Specifications

The adapter returns an `ECRformerInferenceResult` dataclass:
- **`reconstructed_multispectral`**: `np.ndarray` of shape `(13, H, W)` in float32, representing recovered cloud-free surface reflectance bounded in $[0.0, 1.0]$. (Multiply by 10000 to convert back to standard Sentinel-2 L1C/L2A digital numbers).
- **`reconstructed_rgb_preview`**: `np.ndarray` of shape `(H, W, 3)` in uint8 $[0, 255]$, constructed using bands B4 (Red, index 3), B3 (Green, index 2), and B2 (Blue, index 1) with an official $3.0\times$ linear brightness stretch.
- **`input_cloudy_rgb_preview`**: `np.ndarray` of shape `(H, W, 3)` in uint8 $[0, 255]$ for the cloudy input optical image.
- **`input_sar_preview`**: `np.ndarray` of shape `(H, W, 3)` in uint8 $[0, 255]$ false-color composite (R: VV, G: VH, B: ratio).
- **`metadata`**: Includes inference execution time (ms), input/output shapes, output min/max/mean, and finite verification flags.

---

## Python Usage Example

```python
import numpy as np
from geoclear.adapters import ECRformerAdapter

# Initialize adapter (defaults to 'light' variant on CPU or auto-detected CUDA)
adapter = ECRformerAdapter(variant="light", device="cpu")

# Supply co-registered Sentinel-1 SAR (2, H, W) and Sentinel-2 Optical (13, H, W)
sar_input = ...      # float32 array in dB
optical_input = ...  # float32 array in [0, 10000] DN

result = adapter.predict(sar_input, optical_input)

# Access reconstructed 13-band multispectral reflectance array
clean_ms = result.reconstructed_multispectral  # Shape: (13, H, W)

# Access RGB visualization preview
rgb_img = result.reconstructed_rgb_preview     # Shape: (H, W, 3), uint8
```
