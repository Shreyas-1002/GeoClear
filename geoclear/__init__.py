import os
import sys

# Windows PROJ environment isolation: PostgreSQL/PostGIS can set a system-wide PROJ_LIB
# that breaks rasterio's PROJ database lookup. Point to rasterio's internal proj_data if needed.
if "PROJ_LIB" in os.environ and "PostgreSQL" in os.environ["PROJ_LIB"]:
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    rasterio_proj = os.path.join(site_packages, "rasterio", "proj_data")
    if os.path.isdir(rasterio_proj):
        os.environ["PROJ_LIB"] = rasterio_proj
        os.environ["PROJ_DATA"] = rasterio_proj

from geoclear.types import (
    Modality,
    SensorType,
    AuditStatus,
    ConfigurationStatus,
    ImplementationStatus,
    VerificationStatus,
    ReadinessStatus,
    DeclaredData,
    InputRequirements,
    ModelCard,
    CompatibilityResult,
    CompatibilityReport,
    S2_BANDS_13,
    RGB_BANDS,
    S1_SAR_POLARIZATIONS,
)
from geoclear.registry import ModelRegistry
from geoclear.compatibility import CompatibilityEngine

__all__ = [
    "Modality",
    "SensorType",
    "AuditStatus",
    "ConfigurationStatus",
    "ImplementationStatus",
    "VerificationStatus",
    "ReadinessStatus",
    "DeclaredData",
    "InputRequirements",
    "ModelCard",
    "CompatibilityResult",
    "CompatibilityReport",
    "S2_BANDS_13",
    "RGB_BANDS",
    "S1_SAR_POLARIZATIONS",
    "ModelRegistry",
    "CompatibilityEngine",
]
