"""Dedicated adapter for ECRformer multimodal cloud removal models.

Supports ECRformer-Light (default) and full ECRformer checkpoints.
Encapsulates dependency shimming, strict input validation, preprocessing,
CPU/CUDA execution, finite value checking, and RGB preview extraction.

Licensing Note:
    The original ECRformer repository (Zhang et al., Wuhan University) does not
    contain an explicit LICENSE file in its codebase. Under standard copyright
    principles, the absence of an explicit license means copyright remains fully
    with the authors; unrestricted distribution or commercial usage cannot be
    assumed without explicit grant from the authors.
"""

from __future__ import annotations

import os
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from geoclear.types import S2_BANDS_13, S1_SAR_POLARIZATIONS


# ---------------------------------------------------------------------------
# Isolated Dependency Shim for timm.layers
# ---------------------------------------------------------------------------

def _ensure_timm_shim() -> None:
    """Ensure lightweight shims for timm helper functions exist in sys.modules.
    
    ECRformerModel only imports `to_2tuple`, `trunc_normal_`, and `DropPath`
    from timm.layers. In eval/inference mode, DropPath is a no-op identity pass-through.
    Providing an in-process shim avoids downloading or upgrading heavy global packages.
    """
    if "timm" not in sys.modules or "timm.layers" not in sys.modules:
        timm_mod = types.ModuleType("timm")
        layers_mod = types.ModuleType("timm.layers")

        def _to_2tuple(x: Any) -> Tuple[Any, Any]:
            if isinstance(x, (list, tuple)):
                return tuple(x)  # type: ignore[return-value]
            return (x, x)

        class _DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
                super().__init__()
                self.drop_prob = drop_prob
                self.scale_by_keep = scale_by_keep

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # In eval mode (training=False), DropPath is strictly the identity function
                return x

        layers_mod.to_2tuple = _to_2tuple  # type: ignore[attr-defined]
        layers_mod.trunc_normal_ = init.trunc_normal_  # type: ignore[attr-defined]
        layers_mod.DropPath = _DropPath  # type: ignore[attr-defined]

        timm_mod.layers = layers_mod  # type: ignore[attr-defined]

        sys.modules["timm"] = timm_mod
        sys.modules["timm.layers"] = layers_mod


# Apply shim and import ECRformerModel
_ensure_timm_shim()

_REPO_ROOT = Path(__file__).resolve().parents[2] / "models" / "ECRformer" / "ECRformer-master"
if str(_REPO_ROOT) not in sys.path and _REPO_ROOT.exists():
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from models.ecrformer_model import ECRformerModel
except ImportError as exc:
    # Handle import if working directory differs
    raise ImportError(
        f"Could not import ECRformerModel from {_REPO_ROOT}. Ensure repository exists."
    ) from exc


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class ECRformerInferenceResult:
    """Structured output returned by the ECRformer adapter."""
    reconstructed_multispectral: np.ndarray  # Shape: (13, H, W), float32 in [0.0, 1.0]
    reconstructed_rgb_preview: np.ndarray    # Shape: (H, W, 3), uint8 in [0, 255]
    input_cloudy_rgb_preview: np.ndarray     # Shape: (H, W, 3), uint8 in [0, 255]
    input_sar_preview: np.ndarray            # Shape: (H, W, 3), uint8 false-color composite
    output_scale: str                        # Description of numerical scale
    bands: List[str]                         # 13 Sentinel-2 band names
    device: str                              # Device used for inference ('cpu' or 'cuda')
    checkpoint_path: str                     # Checkpoint path loaded
    variant: str                             # 'light' or 'full'
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ECRformer Adapter
# ---------------------------------------------------------------------------

class ECRformerAdapter:
    """Dedicated inference adapter for ECRformer models."""

    # Default relative paths from workspace root
    DEFAULT_LIGHT_CKPT = "models/ECRformer/ECRformer-master/pretrained/light.ckpt"
    DEFAULT_FULL_CKPT = "models/ECRformer/ECRformer-master/pretrained/full.ckpt"

    def __init__(
        self,
        variant: str = "light",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize and load the ECRformer model.

        Args:
            variant: Model variant, either 'light' (default, ~3.7M params) or 'full' (~11.3M params).
            checkpoint_path: Optional path to checkpoint. Defaults to official pretrained weights.
            device: 'cpu' or 'cuda'. If None, auto-selects cuda if available, else cpu.

        Raises:
            ValueError: If variant is invalid or device is unavailable.
            FileNotFoundError: If the specified checkpoint file does not exist.
        """
        variant = variant.lower()
        if variant not in {"light", "full"}:
            raise ValueError(f"Invalid variant: '{variant}'. Must be 'light' or 'full'.")
        self.variant = variant

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
            rel_ckpt = self.DEFAULT_LIGHT_CKPT if variant == "light" else self.DEFAULT_FULL_CKPT
            self.checkpoint_path = str(base_ws / rel_ckpt)
        else:
            self.checkpoint_path = checkpoint_path

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"ECRformer checkpoint not found at: '{self.checkpoint_path}'."
            )

        # Build architecture
        self.model = self._build_model(variant)

        # Load weights
        self._load_checkpoint()

        self.model.to(self.device)
        self.model.eval()

    def _build_model(self, variant: str) -> nn.Module:
        """Construct the ECRformerModel with matching configuration."""
        if variant == "light":
            cfg = dict(
                in_chans=[2, 13],
                out_chans=13,
                num_layers=4,
                num_blocks=[2, 2, 1, 1],
                features_start=32,
                block_type=["ecrformer", "ecrformer"],
                cbam="1ca2+1sa2",
                bottle_neck="tsa",
                num_refine=2,
                pos_encoding=None,
                drop_path_rate=0.0,
            )
        else:  # 'full'
            cfg = dict(
                in_chans=[2, 13],
                out_chans=13,
                num_layers=4,
                num_blocks=[2, 3, 2, 2],
                features_start=48,
                block_type=["ecrformer", "ecrformer"],
                cbam="1ca2+1sa2",
                bottle_neck="tsa",
                num_refine=4,
                pos_encoding=None,
                drop_path_rate=0.0,
            )

        return ECRformerModel(**cfg)

    def _load_checkpoint(self) -> None:
        """Load state dict from PyTorch Lightning checkpoint file onto CPU."""
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        raw_state_dict = ckpt.get("state_dict", ckpt)

        # Strip 'net.' prefix used in PyTorch Lightning module
        clean_state_dict = {}
        for key, value in raw_state_dict.items():
            if key.startswith("net."):
                clean_state_dict[key[4:]] = value
            elif not key.startswith("down_proj.") and not key.startswith("up_proj."):
                clean_state_dict[key] = value

        missing, unexpected = self.model.load_state_dict(clean_state_dict, strict=False)

        # Ensure no core model parameters are missing
        core_missing = [k for k in missing if not k.startswith("down_proj") and not k.startswith("up_proj")]
        if core_missing:
            raise RuntimeError(
                f"Checkpoint '{self.checkpoint_path}' is missing essential parameters: {core_missing[:5]}..."
            )

    def preprocess(
        self,
        sar: Union[np.ndarray, torch.Tensor],
        optical: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Validate and apply official normalization to SAR and optical inputs.

        Args:
            sar: Sentinel-1 SAR array/tensor. Expected channels: 2 (VV, VH). Values in dB.
            optical: Sentinel-2 optical array/tensor. Expected channels: 13 (B1-B12, B8A). Values in [0, 10000].

        Returns:
            Preprocessed merged tensor of shape (1, 15, H, W) on the configured device.

        Raises:
            ValueError: If channels, dimensions, or data validity checks fail.
        """
        # Convert to numpy float32 for initial validation
        if isinstance(sar, torch.Tensor):
            sar_np = sar.detach().cpu().numpy().astype(np.float32)
        else:
            sar_np = np.asarray(sar, dtype=np.float32)

        if isinstance(optical, torch.Tensor):
            opt_np = optical.detach().cpu().numpy().astype(np.float32)
        else:
            opt_np = np.asarray(optical, dtype=np.float32)

        # Standardize to (C, H, W)
        if sar_np.ndim == 4 and sar_np.shape[0] == 1:
            sar_np = sar_np[0]
        if opt_np.ndim == 4 and opt_np.shape[0] == 1:
            opt_np = opt_np[0]

        if sar_np.ndim != 3:
            raise ValueError(
                f"SAR input must have 3 dimensions (C, H, W), got shape {sar_np.shape}."
            )
        if opt_np.ndim != 3:
            raise ValueError(
                f"Optical input must have 3 dimensions (C, H, W), got shape {opt_np.shape}."
            )

        # Validate channel counts
        if sar_np.shape[0] != 2:
            raise ValueError(
                f"SAR input must contain exactly 2 channels (VV, VH), got {sar_np.shape[0]} channels."
            )
        if opt_np.shape[0] != 13:
            raise ValueError(
                f"Optical input must contain exactly 13 Sentinel-2 channels [B1-B12, B8A], "
                f"got {opt_np.shape[0]} channels."
            )

        # Validate spatial dimension matching
        h_sar, w_sar = sar_np.shape[1], sar_np.shape[2]
        h_opt, w_opt = opt_np.shape[1], opt_np.shape[2]

        if (h_sar, w_sar) != (h_opt, w_opt):
            raise ValueError(
                f"Spatial dimension mismatch: SAR is {h_sar}x{w_sar}, but Optical is {h_opt}x{w_opt}."
            )

        # Check minimum size and divisibility by 32
        if h_sar < 32 or w_sar < 32:
            raise ValueError(
                f"Spatial dimensions must be at least 32x32 pixels, got {h_sar}x{w_sar}."
            )

        if h_sar % 32 != 0 or w_sar % 32 != 0:
            raise ValueError(
                f"ECRformer requires image height and width to be divisible by 32; "
                f"got {h_sar}x{w_sar}. Please crop or pad the input patches to a multiple of 32."
            )

        # Sanitize NaNs and Infs (official preprocessing nan_to_num)
        sar_np = np.nan_to_num(sar_np, nan=0.0, posinf=0.0, neginf=-25.0)
        opt_np = np.nan_to_num(opt_np, nan=0.0, posinf=10000.0, neginf=0.0)

        # Official SAR Preprocessing: clip to [-25.0, 0.0] and scale to [0, 1]
        sar_clipped = np.clip(sar_np, -25.0, 0.0)
        sar_norm = (sar_clipped + 25.0) / 25.0

        # Official Optical Preprocessing: clip to [0.0, 10000.0] and scale to [0, 1]
        opt_clipped = np.clip(opt_np, 0.0, 10000.0)
        opt_norm = opt_clipped / 10000.0

        # Concatenate along channel dimension: [SAR (2), Optical (13)] -> 15 channels
        merged_np = np.concatenate([sar_norm, opt_norm], axis=0)  # Shape: (15, H, W)
        merged_tensor = torch.from_numpy(merged_np).unsqueeze(0).to(self.device)  # Shape: (1, 15, H, W)

        return merged_tensor

    @torch.no_grad()
    def predict(
        self,
        sar: Union[np.ndarray, torch.Tensor],
        optical: Union[np.ndarray, torch.Tensor],
    ) -> ECRformerInferenceResult:
        """Run cloud-removal inference on Sentinel-1 SAR and Sentinel-2 optical inputs.

        Args:
            sar: Sentinel-1 SAR array or tensor (2 channels: VV, VH).
            optical: Sentinel-2 optical array or tensor (13 bands: B1-B12, B8A).

        Returns:
            ECRformerInferenceResult containing the reconstructed 13-band output,
            RGB preview, input previews, and execution metadata.
        """
        start_time = time.perf_counter()

        # Preprocess and validate inputs
        merged_tensor = self.preprocess(sar, optical)

        # Model forward pass
        raw_output = self.model(merged_tensor)
        pred_tensor = raw_output[0] if isinstance(raw_output, tuple) else raw_output

        # Clamp output to valid normalized reflectance range [0.0, 1.0]
        pred_tensor = torch.clamp(pred_tensor, 0.0, 1.0)

        # Numerical validity assertion
        if torch.isnan(pred_tensor).any():
            raise RuntimeError("Model produced NaN values during inference.")
        if torch.isinf(pred_tensor).any():
            raise RuntimeError("Model produced Infinite values during inference.")

        # Detach to numpy
        recon_ms = pred_tensor.squeeze(0).cpu().float().numpy()  # (13, H, W)

        # Optical input normalized to [0, 1] for preview extraction
        if isinstance(optical, torch.Tensor):
            opt_raw = optical.detach().cpu().numpy().astype(np.float32)
        else:
            opt_raw = np.asarray(optical, dtype=np.float32)
        if opt_raw.ndim == 4:
            opt_raw = opt_raw[0]

        opt_clean = np.nan_to_num(opt_raw, nan=0.0, posinf=10000.0, neginf=0.0)
        opt_norm = np.clip(opt_clean, 0.0, 10000.0) / 10000.0

        # SAR input for preview
        if isinstance(sar, torch.Tensor):
            sar_raw = sar.detach().cpu().numpy().astype(np.float32)
        else:
            sar_raw = np.asarray(sar, dtype=np.float32)
        if sar_raw.ndim == 4:
            sar_raw = sar_raw[0]

        sar_clean = np.nan_to_num(sar_raw, nan=0.0, posinf=0.0, neginf=-25.0)

        # Generate True-Color RGB previews (B4=idx 3, B3=idx 2, B2=idx 1)
        recon_rgb = self._make_rgb_preview(recon_ms, bands=(3, 2, 1), brightness=3.0)
        cloudy_rgb = self._make_rgb_preview(opt_norm, bands=(3, 2, 1), brightness=3.0)
        sar_preview = self._make_sar_preview(sar_clean)

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = {
            "inference_time_ms": round(inference_time_ms, 2),
            "input_shape": list(merged_tensor.shape),
            "output_shape": list(recon_ms.shape),
            "output_min": float(recon_ms.min()),
            "output_max": float(recon_ms.max()),
            "output_mean": float(recon_ms.mean()),
            "finite_values_verified": True,
            "device": str(self.device),
            "variant": self.variant,
            "brightness_stretch": 3.0,
        }

        return ECRformerInferenceResult(
            reconstructed_multispectral=recon_ms,
            reconstructed_rgb_preview=recon_rgb,
            input_cloudy_rgb_preview=cloudy_rgb,
            input_sar_preview=sar_preview,
            output_scale="[0.0, 1.0] normalized surface reflectance (multiply by 10000 for standard Sentinel-2 DN)",
            bands=list(S2_BANDS_13),
            device=str(self.device),
            checkpoint_path=self.checkpoint_path,
            variant=self.variant,
            metadata=metadata,
        )

    @staticmethod
    def _make_rgb_preview(
        image_ms: np.ndarray,
        bands: Tuple[int, int, int] = (3, 2, 1),
        brightness: float = 3.0,
    ) -> np.ndarray:
        """Extract True-Color RGB preview with official brightness multiplier.
        
        Args:
            image_ms: (13, H, W) normalized array in [0, 1].
            bands: RGB band indices (default B4=3, B3=2, B2=1).
            brightness: Contrast/brightness multiplier (official default 3.0).

        Returns:
            (H, W, 3) uint8 image in [0, 255].
        """
        r, g, b = bands
        clean_ms = np.nan_to_num(image_ms, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = np.stack([clean_ms[r], clean_ms[g], clean_ms[b]], axis=-1)
        rgb_stretched = np.clip(rgb * brightness, 0.0, 1.0)
        return (rgb_stretched * 255.0).round().astype(np.uint8)

    @staticmethod
    def _make_sar_preview(sar_raw: np.ndarray) -> np.ndarray:
        """Create a 3-channel composite preview for 2-band SAR (VV, VH).
        
        R: VV, G: VH, B: (VV - VH) ratio.
        """
        clean_sar = np.nan_to_num(sar_raw, nan=0.0, posinf=0.0, neginf=-25.0)
        vv = np.clip((clean_sar[0] + 25.0) / 25.0, 0.0, 1.0)
        vh = np.clip((clean_sar[1] + 32.5) / 32.5, 0.0, 1.0)
        diff = np.clip((vv - vh + 1.0) / 2.0, 0.0, 1.0)

        composite = np.stack([vv, vh, diff], axis=-1)
        return (composite * 255.0).round().astype(np.uint8)
