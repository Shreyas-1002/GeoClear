"""Dedicated adapter for DSen2-CR multimodal cloud removal models.

Implements the official DSen2-CR deep residual neural network architecture
(Meraner et al., 2020, ISPRS Journal of Photogrammetry and Remote Sensing)
in native PyTorch.

Ingests Sentinel-1 SAR (VV, VH in dB) and Sentinel-2 Optical (all 13 bands in DN [0, 10000]),
applies the documented preprocessing and scaling, executes inference safely under CPU or CUDA,
and converts the reconstructed surface reflectance back with RGB preview extraction.

Licensing:
    DSen2-CR is licensed under GNU General Public License v3.0 (GPL-3.0).
    Official pretrained CARL weights were converted from Keras HDF5 (model_SARcarl.hdf5)
    to PyTorch state dict (dsen2cr_carl.pth) with exact numerical equivalence.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from geoclear.types import S2_BANDS_13, S1_SAR_POLARIZATIONS


# ---------------------------------------------------------------------------
# Native PyTorch Architecture Definition
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    """Residual block matching DSen2-CR definition."""

    def __init__(
        self,
        dim: int,
        use_dropout: bool = False,
        use_bias: bool = True,
        res_scale: float = 0.1,
        padding_size: int = 1,
    ) -> None:
        super(ResnetBlock, self).__init__()
        self.res_scale = res_scale
        self.padding_size = padding_size
        conv_block = [
            nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.conv_block(x)


class ResnetStackedArchitecture(nn.Module):
    """Native PyTorch implementation of DSen2-CR architecture."""

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 13,
        F: int = 256,
        B: int = 16,
        res_scale: float = 0.1,
    ) -> None:
        super(ResnetStackedArchitecture, self).__init__()
        self.F = F
        self.B = B
        self.scale_res = res_scale
        model = [
            nn.Conv2d(in_channels, self.F, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
        ]
        for _ in range(self.B):
            model.append(
                ResnetBlock(
                    self.F,
                    use_dropout=False,
                    use_bias=True,
                    res_scale=self.scale_res,
                    padding_size=1,
                )
            )
        model.append(nn.Conv2d(self.F, out_channels, kernel_size=3, padding=1, bias=True))
        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Long skip-connection: add cloudy MS optical input (first 13 channels) to model output
        return input[:, :-2, ...] + self.model(input)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DSen2CRInferenceResult:
    """Structured output returned by the DSen2-CR adapter."""

    reconstructed_multispectral: np.ndarray  # Shape: (13, H, W), float32 in [0.0, 1.0] (reflectance)
    reconstructed_rgb_preview: np.ndarray    # Shape: (H, W, 3), uint8 in [0, 255]
    input_cloudy_rgb_preview: np.ndarray     # Shape: (H, W, 3), uint8 in [0, 255]
    input_sar_preview: np.ndarray            # Shape: (H, W, 3), uint8 false-color composite
    output_scale: str                        # Description of numerical scale
    bands: List[str]                         # 13 Sentinel-2 band names
    device: str                              # Device used for inference ('cpu' or 'cuda')
    checkpoint_path: str                     # Checkpoint path loaded
    variant: str                             # 'carl'
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DSen2-CR Adapter
# ---------------------------------------------------------------------------

class DSen2CRAdapter:
    """Dedicated inference adapter for DSen2-CR models."""

    # Default relative path from workspace root
    DEFAULT_CHECKPOINT = "models/DSen2-CR/checkpoints/dsen2cr_carl.pth"

    # Preprocessing limits documented in upstream dataIO.py
    OPT_SCALE = 2000.0
    OPT_CLIP_MIN = 0.0
    OPT_CLIP_MAX = 10000.0

    SAR_VV_MIN = -25.0
    SAR_VV_MAX = 0.0
    SAR_VH_MIN = -32.5
    SAR_VH_MAX = 0.0
    SAR_MAX_VAL = 2.0  # Linear scaling target in [0, 2]

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize and load the DSen2-CR model.

        Args:
            checkpoint_path: Optional path to checkpoint. Defaults to converted official CARL weights.
            device: 'cpu' or 'cuda'. If None, auto-selects cuda if available, else cpu.

        Raises:
            ValueError: If device is invalid or requested CUDA is unavailable.
            FileNotFoundError: If the specified checkpoint file does not exist.
        """
        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was explicitly requested, but torch.cuda.is_available() is False.")
            self.device = torch.device("cuda")
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Invalid device '{device}'. Must be 'cpu' or 'cuda'.")

        # Resolve checkpoint path
        if checkpoint_path is None:
            base_ws = Path(__file__).resolve().parents[2]
            self.checkpoint_path = str(base_ws / self.DEFAULT_CHECKPOINT)
        else:
            self.checkpoint_path = checkpoint_path

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"DSen2-CR checkpoint not found at '{self.checkpoint_path}'. "
                "Ensure scripts/convert_dsen2cr_weights.py has been run."
            )

        # Build and load model
        self.model = ResnetStackedArchitecture(in_channels=15, out_channels=13, F=256, B=16, res_scale=0.1)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_optical(self, opt_data: np.ndarray) -> np.ndarray:
        """Preprocess 13-band Sentinel-2 optical input.

        According to DSen2-CR dataIO.py:
        1. Ensure shape (13, H, W).
        2. Clip raw values to [0, 10000].
        3. Divide by scale=2000.0.

        Args:
            opt_data: (13, H, W) numpy array, raw DN in [0, 10000] or reflectance.

        Returns:
            (13, H, W) float32 array scaled by 2000.
        """
        if opt_data.ndim != 3 or opt_data.shape[0] != 13:
            raise ValueError(f"Expected optical input with shape (13, H, W), got {opt_data.shape}")

        if not np.all(np.isfinite(opt_data)):
            raise ValueError("Optical input contains NaN or Inf values.")

        opt_clipped = np.clip(opt_data.astype(np.float32), self.OPT_CLIP_MIN, self.OPT_CLIP_MAX)
        opt_norm = opt_clipped / self.OPT_SCALE
        return opt_norm

    def preprocess_sar(self, sar_data: np.ndarray) -> np.ndarray:
        """Preprocess 2-band Sentinel-1 SAR input (VV, VH in dB).

        According to DSen2-CR dataIO.py:
        1. Channel 0 (VV): clip to [-25.0, 0.0], subtract min, scale to [0, max_val_sar=2].
        2. Channel 1 (VH): clip to [-32.5, 0.0], subtract min, scale to [0, max_val_sar=2].

        Args:
            sar_data: (2, H, W) numpy array, dB backscatter.

        Returns:
            (2, H, W) float32 array in [0.0, 2.0].
        """
        if sar_data.ndim != 3 or sar_data.shape[0] != 2:
            raise ValueError(f"Expected SAR input with shape (2, H, W), got {sar_data.shape}")

        if not np.all(np.isfinite(sar_data)):
            raise ValueError("SAR input contains NaN or Inf values.")

        sar_norm = np.empty_like(sar_data, dtype=np.float32)

        # VV channel
        vv = np.clip(sar_data[0].astype(np.float32), self.SAR_VV_MIN, self.SAR_VV_MAX)
        vv = (vv - self.SAR_VV_MIN) / (self.SAR_VV_MAX - self.SAR_VV_MIN) * self.SAR_MAX_VAL
        sar_norm[0] = vv

        # VH channel
        vh = np.clip(sar_data[1].astype(np.float32), self.SAR_VH_MIN, self.SAR_VH_MAX)
        vh = (vh - self.SAR_VH_MIN) / (self.SAR_VH_MAX - self.SAR_VH_MIN) * self.SAR_MAX_VAL
        sar_norm[1] = vh

        return sar_norm

    def predict(self, sar_input: np.ndarray, optical_input: np.ndarray) -> DSen2CRInferenceResult:
        """Run end-to-end inference using DSen2-CR.

        Args:
            sar_input: (2, H, W) Sentinel-1 SAR imagery in dB (VV, VH).
            optical_input: (13, H, W) Sentinel-2 optical imagery in DN [0, 10000] or reflectance.

        Returns:
            DSen2CRInferenceResult containing 13-band reconstruction and previews.
        """
        start_time = time.perf_counter()

        # Validate dimensions match
        if sar_input.shape[1:] != optical_input.shape[1:]:
            raise ValueError(
                f"Spatial dimension mismatch between SAR {sar_input.shape[1:]} and Optical {optical_input.shape[1:]}"
            )

        # Preprocessing
        opt_norm = self.preprocess_optical(optical_input)
        sar_norm = self.preprocess_sar(sar_input)

        # Concatenate: [Optical (13 channels), SAR (2 channels)] -> (15, H, W)
        input_np = np.concatenate([opt_norm, sar_norm], axis=0)
        input_tensor = torch.from_numpy(input_np).unsqueeze(0).to(self.device)  # (1, 15, H, W)

        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        recon_norm = output_tensor.squeeze(0).cpu().numpy().astype(np.float32)  # (13, H, W)

        if not np.all(np.isfinite(recon_norm)):
            raise ValueError("DSen2-CR model output contains non-finite values (NaN or Inf)!")

        # Invert optical normalization:
        # Model operates in normalized space where 1.0 = 2000 DN (0.20 reflectance).
        # To convert to physical surface reflectance [0.0, 1.0], multiply by 2000 / 10000 = 0.20,
        # or equivalently recon_norm * (scale / 10000).
        # To obtain standard Sentinel-2 integer DN [0, 10000], multiply by scale=2000.
        recon_dn = recon_norm * self.OPT_SCALE
        recon_reflectance = np.clip(recon_dn / 10000.0, 0.0, 1.0).astype(np.float32)

        # Cloudy optical normalized reflectance for preview
        cloudy_reflectance = np.clip(optical_input / 10000.0, 0.0, 1.0).astype(np.float32)

        # Generate True-Color RGB previews (B4=idx 3, B3=idx 2, B2=idx 1)
        recon_rgb = self._make_rgb_preview(recon_reflectance, bands=(3, 2, 1), brightness=3.0)
        cloudy_rgb = self._make_rgb_preview(cloudy_reflectance, bands=(3, 2, 1), brightness=3.0)
        sar_preview = self._make_sar_preview(sar_input)

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = {
            "inference_time_ms": round(inference_time_ms, 2),
            "input_shape": list(input_tensor.shape),
            "output_shape": list(recon_reflectance.shape),
            "output_min": float(recon_reflectance.min()),
            "output_max": float(recon_reflectance.max()),
            "output_mean": float(recon_reflectance.mean()),
            "finite_values_verified": True,
            "device": str(self.device),
            "variant": "carl",
            "brightness_stretch": 3.0,
            "model_name": "DSen2-CR",
        }

        return DSen2CRInferenceResult(
            reconstructed_multispectral=recon_reflectance,
            reconstructed_rgb_preview=recon_rgb,
            input_cloudy_rgb_preview=cloudy_rgb,
            input_sar_preview=sar_preview,
            output_scale="[0.0, 1.0] normalized surface reflectance (multiply by 10000 for standard Sentinel-2 DN)",
            bands=list(S2_BANDS_13),
            device=str(self.device),
            checkpoint_path=self.checkpoint_path,
            variant="carl",
            metadata=metadata,
        )

    @staticmethod
    def _make_rgb_preview(
        image_ms: np.ndarray,
        bands: Tuple[int, int, int] = (3, 2, 1),
        brightness: float = 3.0,
    ) -> np.ndarray:
        """Extract True-Color RGB preview with standard brightness multiplier."""
        r, g, b = bands
        clean_ms = np.nan_to_num(image_ms, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = np.stack([clean_ms[r], clean_ms[g], clean_ms[b]], axis=-1)
        rgb_stretched = np.clip(rgb * brightness, 0.0, 1.0)
        return (rgb_stretched * 255.0).round().astype(np.uint8)

    @staticmethod
    def _make_sar_preview(sar_raw: np.ndarray) -> np.ndarray:
        """Create a 3-channel composite preview for 2-band SAR (VV, VH)."""
        clean_sar = np.nan_to_num(sar_raw, nan=0.0, posinf=0.0, neginf=-25.0)
        vv = np.clip((clean_sar[0] + 25.0) / 25.0, 0.0, 1.0)
        vh = np.clip((clean_sar[1] + 32.5) / 32.5, 0.0, 1.0)
        diff = np.clip((vv - vh + 1.0) / 2.0, 0.0, 1.0)

        composite = np.stack([vv, vh, diff], axis=-1)
        return (composite * 255.0).round().astype(np.uint8)
