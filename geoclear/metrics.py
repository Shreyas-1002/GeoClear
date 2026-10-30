"""Evaluation metrics for reconstructed Earth-observation optical imagery.

Computes MAE, RMSE, PSNR, and SSIM strictly when a suitable ground-truth
reference image is provided. Never fabricates evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from geoclear.types import S2_BANDS_13


@dataclass
class EvaluationMetrics:
    """Quantitative evaluation metrics against a ground-truth reference."""
    mae: float
    rmse: float
    psnr: float
    ssim: float
    per_band_mae: Dict[str, float]
    per_band_rmse: Dict[str, float]
    rgb_psnr: float
    rgb_ssim: float


def compute_evaluation_metrics(
    reconstructed: np.ndarray,
    reference: np.ndarray,
    band_names: Optional[List[str]] = None,
    data_range: float = 1.0,
) -> EvaluationMetrics:
    """Compute quantitative error and quality metrics between reconstruction and reference.

    Args:
        reconstructed: (C, H, W) float32 array in [0.0, 1.0].
        reference: (C, H, W) float32 array in [0.0, 1.0].
        band_names: List of C band names. Defaults to Sentinel-2 13 bands.
        data_range: Dynamic range of input arrays (default 1.0).

    Returns:
        EvaluationMetrics with overall and per-band statistics.

    Raises:
        ValueError: If shapes, dimensions, or finite value assertions fail.
    """
    if reconstructed.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: reconstructed is {reconstructed.shape}, but reference is {reference.shape}."
        )

    if reconstructed.ndim != 3:
        raise ValueError(
            f"Arrays must have 3 dimensions (C, H, W), got {reconstructed.ndim}."
        )

    if np.isnan(reconstructed).any() or np.isnan(reference).any():
        raise ValueError("Arrays contain NaN values; cannot compute valid metrics.")

    if np.isinf(reconstructed).any() or np.isinf(reference).any():
        raise ValueError("Arrays contain infinite values; cannot compute valid metrics.")

    c, h, w = reconstructed.shape
    if band_names is None:
        if c == 13:
            band_names = list(S2_BANDS_13)
        else:
            band_names = [f"Band_{i+1}" for i in range(c)]

    # Overall MAE and RMSE
    diff = reconstructed - reference
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # Per-band MAE and RMSE
    per_band_mae: Dict[str, float] = {}
    per_band_rmse: Dict[str, float] = {}
    for idx, name in enumerate(band_names):
        b_diff = diff[idx]
        per_band_mae[name] = float(np.mean(np.abs(b_diff)))
        per_band_rmse[name] = float(np.sqrt(np.mean(b_diff ** 2)))

    # PSNR (multispectral average)
    try:
        mse = float(np.mean(diff ** 2))
        if mse == 0.0:
            psnr_val = 100.0  # Perfect reconstruction representation
        else:
            psnr_val = float(peak_signal_noise_ratio(reference, reconstructed, data_range=data_range))
    except Exception:
        psnr_val = float("nan")

    # SSIM (multispectral)
    try:
        ssim_val = float(
            structural_similarity(
                reference,
                reconstructed,
                channel_axis=0,
                data_range=data_range,
            )
        )
    except Exception:
        ssim_val = float("nan")

    # True-Color RGB PSNR / SSIM (bands B4=3, B3=2, B2=1 if 13 bands)
    if c >= 4:
        rgb_ref = reference[[3, 2, 1]]
        rgb_rec = reconstructed[[3, 2, 1]]
        rgb_diff = rgb_rec - rgb_ref
        try:
            rgb_mse = float(np.mean(rgb_diff ** 2))
            if rgb_mse == 0.0:
                rgb_psnr = 100.0
            else:
                rgb_psnr = float(peak_signal_noise_ratio(rgb_ref, rgb_rec, data_range=data_range))
        except Exception:
            rgb_psnr = float("nan")
        try:
            rgb_ssim = float(
                structural_similarity(
                    rgb_ref,
                    rgb_rec,
                    channel_axis=0,
                    data_range=data_range,
                )
            )
        except Exception:
            rgb_ssim = float("nan")
    else:
        rgb_psnr = psnr_val
        rgb_ssim = ssim_val

    return EvaluationMetrics(
        mae=round(mae, 4),
        rmse=round(rmse, 4),
        psnr=round(psnr_val, 2),
        ssim=round(ssim_val, 4),
        per_band_mae={k: round(v, 4) for k, v in per_band_mae.items()},
        per_band_rmse={k: round(v, 4) for k, v in per_band_rmse.items()},
        rgb_psnr=round(rgb_psnr, 2),
        rgb_ssim=round(rgb_ssim, 4),
    )
