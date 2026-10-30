"""Synthetic verification patch generator for GeoClear demonstration and testing.

Provides realistic multi-band synthetic test data (Sentinel-1 SAR, cloudy Sentinel-2
multispectral optical, and clean reference optical) to verify software execution
when real satellite datasets are not readily on hand.

Disclaimer:
    Data produced by this module is purely synthetic and intended strictly for
    software testing and pipeline demonstration. It must never be used to claim
    scientific cloud-removal performance on real Earth-observation imagery.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def generate_synthetic_verification_patch(
    height: int = 128,
    width: int = 128,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic co-registered Sentinel-1 SAR and Sentinel-2 optical patches.

    Args:
        height: Spatial height (must be divisible by 32, default 128).
        width: Spatial width (must be divisible by 32, default 128).
        seed: Random seed for repeatability.

    Returns:
        Tuple of:
            - sar: (2, H, W) float32 in dB [-25.0, 0.0] (VV, VH)
            - cloudy_optical: (13, H, W) float32 in DN [0.0, 10000.0]
            - clean_reference: (13, H, W) float32 in DN [0.0, 10000.0]
    """
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(f"Dimensions must be divisible by 32, got {height}x{width}")

    rng = np.random.default_rng(seed)

    # 1. Base terrain elevation and texture pattern
    y, x = np.mgrid[0:height, 0:width]
    terrain = (
        0.5 * np.sin(x / 16.0) * np.cos(y / 16.0)
        + 0.3 * np.sin(x / 8.0)
        + 0.2 * np.cos(y / 8.0)
    )
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())  # [0, 1]

    # 2. Synthetic Clean Optical Reference (13 bands in Sentinel-2 DN scale [0, 10000])
    clean_optical = np.zeros((13, height, width), dtype=np.float32)
    # Band baseline reflectance multipliers (roughly mimicking soil/vegetation spectral curves)
    band_scales = [
        1200.0,  # B1: Coastal aerosol
        1400.0,  # B2: Blue
        1800.0,  # B3: Green
        1600.0,  # B4: Red
        2200.0,  # B5: Red Edge 1
        3400.0,  # B6: Red Edge 2
        4000.0,  # B7: Red Edge 3
        4200.0,  # B8: NIR Broad
        4300.0,  # B8A: NIR Narrow
        1100.0,  # B9: Water Vapour
        300.0,   # B10: SWIR Cirrus
        2400.0,  # B11: SWIR 1
        1600.0,  # B12: SWIR 2
    ]

    for b in range(13):
        noise = rng.normal(0, 0.05, (height, width)).astype(np.float32)
        base = np.clip(terrain + noise, 0.0, 1.0)
        clean_optical[b] = np.clip(base * band_scales[b] * 1.5, 100.0, 9500.0)

    # 3. Synthetic Cloud & Haze Layer
    cloud_center_x, cloud_center_y = width * 0.5, height * 0.4
    dist = np.sqrt((x - cloud_center_x) ** 2 + (y - cloud_center_y) ** 2)
    cloud_density = np.exp(-((dist / (width * 0.35)) ** 2))
    cloud_noise = rng.uniform(0.7, 1.3, (height, width)).astype(np.float32)
    cloud_mask = np.clip(cloud_density * cloud_noise, 0.0, 1.0)

    # Cloudy Optical: blend clean surface with thick cloud reflectance (~8000-10000 DN)
    cloudy_optical = clean_optical.copy()
    for b in range(13):
        cloud_val = 8500.0 + rng.uniform(-500.0, 500.0, (height, width)).astype(np.float32)
        cloudy_optical[b] = (
            (1.0 - cloud_mask) * clean_optical[b] + cloud_mask * cloud_val
        )
    cloudy_optical = np.clip(cloudy_optical, 0.0, 10000.0)

    # 4. Synthetic Sentinel-1 SAR (VV, VH in dB scale [-25.0, 0.0])
    # SAR penetrates clouds unaffected, but shows backscatter from surface texture
    sar = np.zeros((2, height, width), dtype=np.float32)
    # VV: higher co-polarized return
    sar_vv = -18.0 + terrain * 12.0 + rng.normal(0, 1.5, (height, width)).astype(np.float32)
    # VH: cross-polarized return (~6-8 dB lower than VV)
    sar_vh = -24.0 + terrain * 10.0 + rng.normal(0, 1.5, (height, width)).astype(np.float32)

    sar[0] = np.clip(sar_vv, -25.0, 0.0)
    sar[1] = np.clip(sar_vh, -25.0, 0.0)

    return sar, cloudy_optical, clean_optical
