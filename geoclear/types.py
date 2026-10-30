"""Type definitions and data structures for the GeoClear platform.

Defines enums, dataclasses, and validation logic for model statuses,
modalities, sensors, user-declared data, requirements, and compatibility results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


# Standard spectral bands and sensor channels
S2_BANDS_13: Tuple[str, ...] = (
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
)
RGB_BANDS: Tuple[str, ...] = ("B4", "B3", "B2")  # Red, Green, Blue in Sentinel-2 naming
GENERIC_RGB_BANDS: Tuple[str, ...] = ("R", "G", "B")
S1_SAR_POLARIZATIONS: Tuple[str, ...] = ("VV", "VH")


class AuditStatus(str, Enum):
    """Status of code and repository inspection."""
    NOT_AUDITED = "NOT_AUDITED"
    AUDITED = "AUDITED"


class ConfigurationStatus(str, Enum):
    """Status of model configuration representation within GeoClear."""
    NOT_CONFIGURED = "NOT_CONFIGURED"
    CONFIGURED = "CONFIGURED"


class ImplementationStatus(str, Enum):
    """Status of dedicated adapter and runtime execution implementation."""
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    IN_PROGRESS = "IN_PROGRESS"
    IMPLEMENTED = "IMPLEMENTED"


class VerificationStatus(str, Enum):
    """Status of actual inference verification.
    
    Statuses must only be assigned based on verified evidence:
    - UNVERIFIED: No execution verified.
    - SYNTHETIC_INFERENCE_VERIFIED: Execution tested using synthetic multi-band verification tensors.
    - REAL_DATA_INFERENCE_VERIFIED: Inference succeeded on suitable real satellite inputs.
    - FULLY_INTEGRATED: Complete supported workflow tested end-to-end.
    """
    UNVERIFIED = "UNVERIFIED"
    SYNTHETIC_INFERENCE_VERIFIED = "SYNTHETIC_INFERENCE_VERIFIED"
    REAL_DATA_INFERENCE_VERIFIED = "REAL_DATA_INFERENCE_VERIFIED"
    FULLY_INTEGRATED = "FULLY_INTEGRATED"


class ReadinessStatus(str, Enum):
    """High-level operational readiness of a model."""
    READY_FOR_INFERENCE = "READY_FOR_INFERENCE"
    BLOCKED_MISSING_WEIGHTS = "BLOCKED_MISSING_WEIGHTS"
    BLOCKED_LEGACY_FRAMEWORK = "BLOCKED_LEGACY_FRAMEWORK"
    BLOCKED_CODE_BUG = "BLOCKED_CODE_BUG"
    BLOCKED_UNIMPLEMENTED_ADAPTER = "BLOCKED_UNIMPLEMENTED_ADAPTER"


class Modality(str, Enum):
    """Input Earth observation modality."""
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    MULTISPECTRAL = "MULTISPECTRAL"
    MULTI_TEMPORAL = "MULTI_TEMPORAL"
    MASK = "MASK"


class SensorType(str, Enum):
    """Satellite sensor or platform."""
    SENTINEL_1 = "SENTINEL_1"
    SENTINEL_2 = "SENTINEL_2"
    GENERIC_RGB = "GENERIC_RGB"
    LANDSAT_8 = "LANDSAT_8"
    OTHER = "OTHER"


@dataclass(frozen=True)
class DeclaredData:
    """User-declared available dataset characteristics.
    
    Represents the exact modalities, sensors, spectral bands, and dimensions
    that a user actually possesses for processing.
    """
    modalities: Set[Modality] = field(default_factory=set)
    sensors: Set[SensorType] = field(default_factory=set)
    optical_bands: Set[str] = field(default_factory=set)
    sar_polarizations: Set[str] = field(default_factory=set)
    temporal_observations: int = 1
    has_cloud_mask: bool = False
    has_ground_truth_reference: bool = False
    image_height: Optional[int] = None
    image_width: Optional[int] = None

    def validate(self) -> None:
        """Validate declaration against logical contradictions and invalid values."""
        if self.temporal_observations < 1:
            raise ValueError(
                f"temporal_observations must be >= 1, got {self.temporal_observations}"
            )

        if self.image_height is not None and self.image_height <= 0:
            raise ValueError(f"image_height must be > 0, got {self.image_height}")

        if self.image_width is not None and self.image_width <= 0:
            raise ValueError(f"image_width must be > 0, got {self.image_width}")

        # If SAR modality declared, polarizations or sensor should be consistent
        if Modality.SAR in self.modalities and not self.sar_polarizations:
            if SensorType.SENTINEL_1 in self.sensors:
                # Default to missing polarizations rather than silently creating them
                raise ValueError(
                    "SAR modality declared for Sentinel-1 but no polarizations specified (e.g. {'VV', 'VH'})."
                )

        # If Optical or Multispectral modality declared, optical bands should be declared
        if (Modality.OPTICAL in self.modalities or Modality.MULTISPECTRAL in self.modalities) and not self.optical_bands:
            raise ValueError(
                "Optical/Multispectral modality declared but optical_bands set is empty."
            )

        # Multi-temporal modality declared without >1 observations
        if Modality.MULTI_TEMPORAL in self.modalities and self.temporal_observations < 2:
            raise ValueError(
                f"MULTI_TEMPORAL modality declared but temporal_observations is {self.temporal_observations} (< 2)."
            )


@dataclass(frozen=True)
class InputRequirements:
    """Rigorous input requirements for a model candidate."""
    required_modalities: Set[Modality]
    required_sensors: Set[SensorType]
    required_optical_bands: Set[str] = field(default_factory=set)
    required_sar_polarizations: Set[str] = field(default_factory=set)
    min_temporal_observations: int = 1
    expected_channels: int = 3
    channel_ordering_description: str = ""
    spatial_resolution_meters: Optional[float] = None
    required_spatial_dimensions: Optional[Tuple[int, int]] = None
    dimension_multiple_of: Optional[int] = None
    requires_cloud_mask: bool = False
    accepts_cloud_mask: bool = False
    requires_coregistration: bool = False


@dataclass
class ModelCard:
    """Comprehensive metadata card for an audited or configured model."""
    model_id: str
    display_name: str
    description: str
    paper_citation: str
    paper_url: str
    license_type: str
    has_explicit_repo_license: bool
    license_notes: str

    # Status tracking
    audit_status: AuditStatus
    configuration_status: ConfigurationStatus
    implementation_status: ImplementationStatus
    verification_status: VerificationStatus
    readiness_status: ReadinessStatus

    # Checkpoint information
    checkpoint_present_locally: bool
    checkpoint_files: List[str]
    local_weights_directory: Optional[str] = None

    # Technical environment
    framework: str = "PyTorch"
    framework_version: str = ">= 2.0"
    python_version_support: str = ">= 3.10"
    dependency_constraints: List[str] = field(default_factory=list)

    # Inputs and outputs
    input_requirements: InputRequirements = field(
        default_factory=lambda: InputRequirements(
            required_modalities=set(), required_sensors=set()
        )
    )
    output_description: str = ""
    output_bands: List[str] = field(default_factory=list)
    output_range: str = "[0, 1]"
    output_format: str = "GeoTIFF / NumPy"

    # Limitations & blockers
    known_limitations: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)


@dataclass
class CompatibilityResult:
    """Detailed compatibility outcome for a specific model."""
    model_id: str
    model_name: str
    is_compatible: bool
    is_executable: bool
    missing_inputs: List[str] = field(default_factory=list)
    exclusion_reasons: List[str] = field(default_factory=list)
    execution_blockers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class CompatibilityReport:
    """Comprehensive report returned by the CompatibilityEngine."""
    declared_data: DeclaredData
    compatible_models: List[CompatibilityResult] = field(default_factory=list)
    incompatible_models: List[CompatibilityResult] = field(default_factory=list)
    executable_models: List[CompatibilityResult] = field(default_factory=list)

    def get_result(self, model_id: str) -> Optional[CompatibilityResult]:
        for res in self.compatible_models + self.incompatible_models:
            if res.model_id == model_id:
                return res
        return None

    @property
    def total_evaluated(self) -> int:
        return len(self.compatible_models) + len(self.incompatible_models)
