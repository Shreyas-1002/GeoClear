"""I/O utilities for geospatial raster imagery and numpy arrays.

Provides robust reading, validation, coregistration checking, and exporting
for GeoTIFF, NPZ, and NPY files using rasterio and Pillow.
"""

from __future__ import annotations

import io
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Windows PROJ environment isolation: PostgreSQL/PostGIS can set a system-wide PROJ_LIB
# that breaks rasterio's PROJ database lookup. Point to rasterio's internal proj_data if needed.
if "PROJ_LIB" in os.environ and "PostgreSQL" in os.environ["PROJ_LIB"]:
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    rasterio_proj = os.path.join(site_packages, "rasterio", "proj_data")
    if os.path.isdir(rasterio_proj):
        os.environ["PROJ_LIB"] = rasterio_proj
        os.environ["PROJ_DATA"] = rasterio_proj

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from PIL import Image


class RasterValidationError(ValueError):
    """Raised when an uploaded raster fails channel, dimension, or numerical validation."""
    pass


def read_raster_file(
    file_bytes_or_path: Union[bytes, str],
    expected_channels: Optional[int] = None,
    name: str = "Input",
    require_divisible_by: Optional[int] = 32,
    check_finite: bool = True,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Read a raster file (GeoTIFF, NPZ, or NPY) into a float32 NumPy array (C, H, W).

    Args:
        file_bytes_or_path: Raw bytes from upload or file path string.
        expected_channels: Optional channel count validation.
        name: Name for descriptive error messages.
        require_divisible_by: Optional spatial dimension divisibility check (default 32).
        check_finite: Whether to verify all pixel values are finite (no NaN / Inf).

    Returns:
        Tuple of (array of shape (C, H, W) in float32, metadata dictionary or None).

    Raises:
        ValueError: If file format is invalid, corrupted, channel count is wrong,
            values are non-finite, or dimensions are not divisible by require_divisible_by.
    """
    if isinstance(file_bytes_or_path, str):
        with open(file_bytes_or_path, "rb") as f:
            data_bytes = f.read()
    else:
        data_bytes = file_bytes_or_path

    # 1. Try reading as GeoTIFF via rasterio
    try:
        with MemoryFile(data_bytes) as memfile:
            with memfile.open() as src:
                arr = src.read().astype(np.float32)  # Shape: (C, H, W)
                meta = {
                    "format": "GeoTIFF",
                    "driver": src.driver,
                    "crs": str(src.crs) if src.crs else None,
                    "transform": list(src.transform) if src.transform else None,
                    "bounds": list(src.bounds) if src.bounds else None,
                    "res": list(src.res) if src.res else None,
                    "nodata": src.nodata,
                    "channels": src.count,
                    "height": src.height,
                    "width": src.width,
                    "dtype": str(src.dtypes[0]),
                }
                _validate_array_properties(arr, expected_channels, name, require_divisible_by, check_finite)
                return arr, meta
    except RasterValidationError:
        raise
    except Exception:
        pass  # Not a GeoTIFF, try NPZ / NPY

    # 2. Try reading as NumPy NPZ archive
    try:
        buffer = io.BytesIO(data_bytes)
        npz = np.load(buffer)
        # Look for standard keys
        target_key = None
        for key in ["data", "arr_0", "image", "array", "sar", "optical", "pred"]:
            if key in npz.files:
                target_key = key
                break
        if target_key is None and len(npz.files) > 0:
            target_key = npz.files[0]

        if target_key is not None:
            arr = npz[target_key].astype(np.float32)
            arr = _standardize_shape(arr)
            meta = {
                "format": "NumPy NPZ",
                "key_used": target_key,
                "channels": arr.shape[0],
                "height": arr.shape[1],
                "width": arr.shape[2],
            }
            _validate_array_properties(arr, expected_channels, name, require_divisible_by, check_finite)
            return arr, meta
    except RasterValidationError:
        raise
    except Exception:
        pass

    # 3. Try reading as raw NPY
    try:
        buffer = io.BytesIO(data_bytes)
        arr = np.load(buffer).astype(np.float32)
        arr = _standardize_shape(arr)
        meta = {
            "format": "NumPy NPY",
            "channels": arr.shape[0],
            "height": arr.shape[1],
            "width": arr.shape[2],
        }
        _validate_array_properties(arr, expected_channels, name, require_divisible_by, check_finite)
        return arr, meta
    except RasterValidationError:
        raise
    except Exception:
        pass

    raise ValueError(
        f"Could not read {name}. The file must be a valid GeoTIFF (.tif/.tiff), "
        f"NumPy NPZ (.npz), or NumPy NPY (.npy) file."
    )


def _standardize_shape(arr: np.ndarray) -> np.ndarray:
    """Ensure array has shape (C, H, W)."""
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        # If shape is (H, W, C) where C in {2, 3, 13} and H, W > C
        if arr.shape[2] in {2, 3, 13} and arr.shape[0] > arr.shape[2]:
            return np.transpose(arr, (2, 0, 1))
        return arr
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    raise RasterValidationError(f"Unexpected array dimension: {arr.ndim}. Expected 2D or 3D array.")


def _validate_array_properties(
    arr: np.ndarray,
    expected_channels: Optional[int],
    name: str,
    require_divisible_by: Optional[int] = 32,
    check_finite: bool = True,
) -> None:
    """Validate channel count, spatial dimension divisibility, and finite values."""
    if expected_channels is not None and arr.shape[0] != expected_channels:
        raise RasterValidationError(
            f"{name} channel mismatch: expected exactly {expected_channels} channels, "
            f"got {arr.shape[0]} channels (shape: {arr.shape})."
        )

    if check_finite:
        if not np.isfinite(arr).all():
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            raise RasterValidationError(
                f"{name} contains non-finite values ({nan_count} NaNs, {inf_count} Infs). "
                f"Input imagery must contain valid finite numerical values."
            )

    if require_divisible_by is not None and require_divisible_by > 1:
        h, w = arr.shape[1], arr.shape[2]
        if h % require_divisible_by != 0 or w % require_divisible_by != 0:
            raise RasterValidationError(
                f"{name} spatial dimensions ({h}x{w}) must be divisible by {require_divisible_by} "
                f"for encoder-decoder patch processing (e.g. 128x128, 256x256, 512x512)."
            )


def validate_spatial_alignment(
    sar_meta: Optional[Dict[str, Any]],
    opt_meta: Optional[Dict[str, Any]],
    sar_shape: Optional[Tuple[int, ...]] = None,
    opt_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[bool, List[str]]:
    """Validate spatial co-registration between SAR and Optical inputs.

    Checks spatial grid shape, CRS, and geotransform metadata when available.

    Args:
        sar_meta: Metadata dictionary from SAR file (if GeoTIFF).
        opt_meta: Metadata dictionary from Optical file (if GeoTIFF).
        sar_shape: Shape tuple of SAR array (e.g. (2, H, W)).
        opt_shape: Shape tuple of Optical array (e.g. (13, H, W)).

    Returns:
        Tuple of (is_aligned: bool, notes_or_warnings: List[str]).
    """
    notes: List[str] = []

    # 1. Check array spatial dimensions
    if sar_shape is not None and opt_shape is not None:
        sar_hw = sar_shape[-2:]
        opt_hw = opt_shape[-2:]
        if sar_hw != opt_hw:
            notes.append(
                f"Spatial dimension mismatch: SAR has resolution {sar_hw[0]}x{sar_hw[1]} "
                f"while Optical has {opt_hw[0]}x{opt_hw[1]}."
            )
            return False, notes

    # 2. Check metadata
    if sar_meta is None or opt_meta is None:
        notes.append(
            "Geospatial CRS/transform metadata is unavailable (NumPy upload format). "
            "Matching pixel dimensions confirmed, but spatial co-registration cannot be "
            "geometrically verified from coordinates alone. Ensure upstream co-registration."
        )
        return True, notes

    if sar_meta.get("format") != "GeoTIFF" or opt_meta.get("format") != "GeoTIFF":
        notes.append("One or both inputs lack standard GeoTIFF spatial references.")
        return True, notes

    # Check CRS
    sar_crs = sar_meta.get("crs")
    opt_crs = opt_meta.get("crs")
    if sar_crs and opt_crs:
        if sar_crs != opt_crs:
            notes.append(
                f"CRS Mismatch: SAR CRS is '{sar_crs}' while Optical CRS is '{opt_crs}'."
            )
            return False, notes
        else:
            notes.append(f"Matching CRS confirmed: {sar_crs}")

    # Check pixel resolution
    sar_res = sar_meta.get("res")
    opt_res = opt_meta.get("res")
    if sar_res and opt_res and sar_res != opt_res:
        notes.append(
            f"Pixel Resolution Discrepancy: SAR resolution is {sar_res}, while Optical resolution is {opt_res}."
        )

    # Check bounds / geotransform
    sar_bounds = sar_meta.get("bounds")
    opt_bounds = opt_meta.get("bounds")
    if sar_bounds and opt_bounds and sar_bounds != opt_bounds:
        notes.append(
            f"Bounding Box Warning: SAR bounds {sar_bounds} differ from Optical bounds {opt_bounds}."
        )

    if not any("Mismatch" in n for n in notes):
        notes.append("Spatial alignment verified: matching dimensions and georeference coordinates.")

    return True, notes


def export_npz_bytes(array: np.ndarray, array_name: str = "reconstructed") -> bytes:
    """Export NumPy array into compressed .npz archive bytes for download."""
    buf = io.BytesIO()
    np.savez_compressed(buf, **{array_name: array})
    buf.seek(0)
    return buf.getvalue()


def export_png_bytes(rgb_uint8: np.ndarray) -> bytes:
    """Export uint8 RGB image array (H, W, 3) into PNG bytes for download."""
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {rgb_uint8.shape}")
    img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
