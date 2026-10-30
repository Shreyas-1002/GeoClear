"""Model registry implementation for GeoClear.

Registers the operational models (ECRformer-Light and DSen2-CR) with their exact documented
modality, sensor, spectral band, dimensional, framework, licensing, and operational
characteristics.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from geoclear.types import (
    AuditStatus,
    ConfigurationStatus,
    ImplementationStatus,
    InputRequirements,
    Modality,
    ModelCard,
    ReadinessStatus,
    SensorType,
    VerificationStatus,
    S2_BANDS_13,
    RGB_BANDS,
    S1_SAR_POLARIZATIONS,
)


def _build_ecrformer_card() -> ModelCard:
    return ModelCard(
        model_id="ecrformer",
        display_name="ECRformer-Light",
        description=(
            "Efficient Cloud Removal Transformer with Semantic-Decoupled Learning "
            "for Multimodal Satellite Imagery."
        ),
        paper_citation=(
            "Zhang, Z., Li, J., Liang, Y., Yan, J., Xiao, Y., Su, X., & Yuan, Q. (2026). "
            "ECRformer: An efficient cloud removal Transformer with semantic-decoupled learning "
            "for multimodal satellite imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 237, 323-338."
        ),
        paper_url="https://doi.org/10.1016/j.isprsjprs.2026.04.009",
        license_type="Unspecified",
        has_explicit_repo_license=False,
        license_notes=(
            "No explicit LICENSE file is present in the repository root or codebase. "
            "The absence of an explicit license means copyright remains fully with the authors; "
            "unrestricted commercial or redistribution rights cannot be assumed without explicit grant."
        ),
        audit_status=AuditStatus.AUDITED,
        configuration_status=ConfigurationStatus.CONFIGURED,
        implementation_status=ImplementationStatus.IMPLEMENTED,
        verification_status=VerificationStatus.SYNTHETIC_INFERENCE_VERIFIED,
        readiness_status=ReadinessStatus.READY_FOR_INFERENCE,
        checkpoint_present_locally=True,
        checkpoint_files=[
            "models/ECRformer/ECRformer-master/pretrained/full.ckpt",
            "models/ECRformer/ECRformer-master/pretrained/light.ckpt",
        ],
        local_weights_directory="models/ECRformer/ECRformer-master/pretrained",
        framework="PyTorch",
        framework_version=">= 2.0",
        python_version_support=">= 3.10",
        dependency_constraints=[
            "torch>=2.0",
            "einops>=0.8",
            "timm (helper layers only; DropPath/to_2tuple)",
        ],
        input_requirements=InputRequirements(
            required_modalities={Modality.SAR, Modality.MULTISPECTRAL},
            required_sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            required_optical_bands=set(S2_BANDS_13),
            required_sar_polarizations=set(S1_SAR_POLARIZATIONS),
            min_temporal_observations=1,
            expected_channels=15,
            channel_ordering_description=(
                "[S1_VV, S1_VH, S2_B1, S2_B2, S2_B3, S2_B4, S2_B5, S2_B6, S2_B7, "
                "S2_B8, S2_B8A, S2_B9, S2_B10, S2_B11, S2_B12] (2 SAR + 13 Optical MS)"
            ),
            spatial_resolution_meters=10.0,
            required_spatial_dimensions=None,
            dimension_multiple_of=32,
            requires_cloud_mask=False,
            accepts_cloud_mask=False,
            requires_coregistration=True,
        ),
        output_description="Reconstructed 13-band Sentinel-2 optical surface reflectance",
        output_bands=list(S2_BANDS_13),
        output_range="[0.0, 1.0]",
        output_format="13-band GeoTIFF / NumPy array, with true-color RGB preview (B4, B3, B2)",
        known_limitations=[
            "Requires co-registered Sentinel-1 SAR imagery; cannot run on optical data alone.",
            "Requires all 13 Sentinel-2 spectral bands; cannot run on standard 3-band RGB imagery.",
            "License is unstated in the repository; commercial/unrestricted use requires author clarification.",
            "Outputs are model-generated optical estimates informed by SAR and optical inputs.",
        ],
        blockers=[],
    )


def _build_dsen2_cr_card() -> ModelCard:
    return ModelCard(
        model_id="dsen2_cr",
        display_name="DSen2-CR",
        description=(
            "Deep residual neural network for cloud removal in Sentinel-2 imagery "
            "using SAR-optical data fusion."
        ),
        paper_citation=(
            "Meraner, A., Ebel, P., Zhu, X. X., & Schmitt, M. (2020). "
            "Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion. "
            "ISPRS Journal of Photogrammetry and Remote Sensing, 166, 333-346."
        ),
        paper_url="https://doi.org/10.1016/j.isprsjprs.2020.05.013",
        license_type="GNU General Public License v3.0 (GPLv3)",
        has_explicit_repo_license=True,
        license_notes=(
            "Licensed under GNU GPL v3.0. Official CARL checkpoint converted to PyTorch state dict "
            "with verified mathematical equivalence."
        ),
        audit_status=AuditStatus.AUDITED,
        configuration_status=ConfigurationStatus.CONFIGURED,
        implementation_status=ImplementationStatus.IMPLEMENTED,
        verification_status=VerificationStatus.SYNTHETIC_INFERENCE_VERIFIED,
        readiness_status=ReadinessStatus.READY_FOR_INFERENCE,
        checkpoint_present_locally=True,
        checkpoint_files=[
            "models/DSen2-CR/checkpoints/dsen2cr_carl.pth",
            "models/DSen2-CR/checkpoints/model_SARcarl.hdf5",
        ],
        local_weights_directory="models/DSen2-CR/checkpoints",
        framework="PyTorch",
        framework_version="PyTorch >= 2.0 (Native PyTorch architecture converted from Keras)",
        python_version_support=">= 3.10",
        dependency_constraints=[
            "torch>=2.0",
            "numpy",
            "h5py",
        ],
        input_requirements=InputRequirements(
            required_modalities={Modality.SAR, Modality.MULTISPECTRAL},
            required_sensors={SensorType.SENTINEL_1, SensorType.SENTINEL_2},
            required_optical_bands=set(S2_BANDS_13),
            required_sar_polarizations=set(S1_SAR_POLARIZATIONS),
            min_temporal_observations=1,
            expected_channels=15,
            channel_ordering_description="[S2 (13 channels: B1..B12), S1 (2 channels: VV, VH)]",
            spatial_resolution_meters=10.0,
            required_spatial_dimensions=None,
            dimension_multiple_of=None,
            requires_cloud_mask=False,
            accepts_cloud_mask=True,
            requires_coregistration=True,
        ),
        output_description="Reconstructed 13-band Sentinel-2 optical surface reflectance",
        output_bands=list(S2_BANDS_13),
        output_range="[0.0, 1.0]",
        output_format="13-band GeoTIFF / NumPy array, with true-color RGB preview (B4, B3, B2)",
        known_limitations=[
            "Requires co-registered Sentinel-1 SAR imagery (VV/VH in dB).",
            "Requires all 13 Sentinel-2 spectral bands.",
            "GPLv3 copyleft license.",
            "Outputs are model-generated optical estimates informed by SAR and optical inputs.",
        ],
        blockers=[],
    )


class ModelRegistry:
    """Central registry of available operational models in GeoClear.
    
    Provides query methods, metadata inspection, and status tracking.
    """

    _MODELS: Dict[str, ModelCard] = {
        "ecrformer": _build_ecrformer_card(),
        "dsen2_cr": _build_dsen2_cr_card(),
    }

    @classmethod
    def list_models(cls) -> List[ModelCard]:
        """Return a copy of all registered model cards."""
        return [copy.deepcopy(model) for model in cls._MODELS.values()]

    @classmethod
    def get_model(cls, model_id: str) -> Optional[ModelCard]:
        """Retrieve a deep copy of a specific model card by ID, or None if not found."""
        model = cls._MODELS.get(model_id)
        return copy.deepcopy(model) if model is not None else None

    @classmethod
    def list_model_ids(cls) -> List[str]:
        """Return all registered model IDs."""
        return list(cls._MODELS.keys())

    @classmethod
    def list_audited_models(cls) -> List[ModelCard]:
        """Return all models marked as AUDITED."""
        return [
            copy.deepcopy(m)
            for m in cls._MODELS.values()
            if m.audit_status == AuditStatus.AUDITED
        ]

    @classmethod
    def list_configured_models(cls) -> List[ModelCard]:
        """Return all models marked as CONFIGURED."""
        return [
            copy.deepcopy(m)
            for m in cls._MODELS.values()
            if m.configuration_status == ConfigurationStatus.CONFIGURED
        ]
