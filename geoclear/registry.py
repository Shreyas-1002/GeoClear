"""Model registry implementation for GeoClear.

Registers the four audited external candidate models with their exact documented
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
        display_name="ECRformer",
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
        output_description="Recovered 13-band Sentinel-2 surface reflectance",
        output_bands=list(S2_BANDS_13),
        output_range="[0.0, 1.0]",
        output_format="13-band GeoTIFF / NumPy array, with true-color RGB preview (B4, B3, B2)",
        known_limitations=[
            "Requires co-registered Sentinel-1 SAR imagery; cannot run on optical data alone.",
            "Requires all 13 Sentinel-2 spectral bands; cannot run on standard 3-band RGB imagery.",
            "License is unstated in the repository; commercial/unrestricted use requires author clarification.",
            "Verification status is SYNTHETIC_INFERENCE_VERIFIED; real-data verification is pending availability of co-registered SEN12MS-CR imagery.",
        ],
        blockers=[],
    )


def _build_diffcr_card() -> ModelCard:
    return ModelCard(
        model_id="diffcr",
        display_name="DiffCR",
        description=(
            "A Fast Conditional Diffusion Framework for Cloud Removal from "
            "Multi-Temporal Optical Satellite Images."
        ),
        paper_citation=(
            "Zou, X., Li, K., Xing, J., Zhang, Y., Wang, S., Jin, L., & Tao, P. (2024). "
            "DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal From Optical Satellite Images. "
            "IEEE Transactions on Geoscience and Remote Sensing, 62, 1-14."
        ),
        paper_url="https://doi.org/10.1109/TGRS.2024.3374465",
        license_type="Unspecified",
        has_explicit_repo_license=False,
        license_notes=(
            "No LICENSE file is present in the repository root or codebase. "
            "Unrestricted use or redistribution rights are not explicitly granted."
        ),
        audit_status=AuditStatus.AUDITED,
        configuration_status=ConfigurationStatus.CONFIGURED,
        implementation_status=ImplementationStatus.NOT_IMPLEMENTED,
        verification_status=VerificationStatus.UNVERIFIED,
        readiness_status=ReadinessStatus.BLOCKED_CODE_BUG,
        checkpoint_present_locally=True,
        checkpoint_files=[
            "models/DiffCR/DiffCR-main/pretrained/diffcr_new.pth",
            "models/DiffCR/DiffCR-main/pretrained/diffcr_old.pth",
        ],
        local_weights_directory="models/DiffCR/DiffCR-main/pretrained",
        framework="PyTorch",
        framework_version=">= 1.6",
        python_version_support=">= 3.8",
        dependency_constraints=[
            "torch>=1.6",
            "torchvision",
            "numpy",
            "pandas",
            "tqdm",
            "scipy",
            "clean-fid",
        ],
        input_requirements=InputRequirements(
            required_modalities={Modality.OPTICAL, Modality.MULTI_TEMPORAL},
            required_sensors={SensorType.SENTINEL_2, SensorType.GENERIC_RGB},
            required_optical_bands={"B4", "B3", "B2"},
            required_sar_polarizations=set(),
            min_temporal_observations=3,
            expected_channels=9,
            channel_ordering_description=(
                "[T0_R, T0_G, T0_B, T1_R, T1_G, T1_B, T2_R, T2_G, T2_B] "
                "(3 co-registered temporal cloudy RGB observations, 9 channels total)"
            ),
            spatial_resolution_meters=10.0,
            required_spatial_dimensions=None,
            dimension_multiple_of=16,
            requires_cloud_mask=False,
            accepts_cloud_mask=False,
            requires_coregistration=True,
        ),
        output_description="Recovered cloud-free RGB optical image",
        output_bands=["R", "G", "B"],
        output_range="[-1.0, 1.0] (rescaled to [0, 255] uint8)",
        output_format="3-band RGB PNG / GeoTIFF",
        known_limitations=[
            "Strictly requires 3 temporal cloudy optical acquisitions of the exact same scene.",
            "Does not support or benefit from SAR imagery in these checkpoints.",
            "Multi-step diffusion sampling (20 steps DPM-Solver) incurs high CPU latency.",
            "Upstream import path bug prevents official run.py from executing out of the box.",
        ],
        blockers=[
            (
                "Upstream code bug in models/network_x0_dpm_solver.py: line 47 imports from "
                "models.ours.ours_double_encoder_splitcaCond_splitcaUnet instead of "
                "models.ours.nafnet_double_encoder_splitcaCond_splitcaUnet."
            ),
            "Adapter not yet implemented.",
        ],
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
        output_description="Recovered 13-band Sentinel-2 surface reflectance",
        output_bands=list(S2_BANDS_13),
        output_range="[0.0, 1.0]",
        output_format="13-band GeoTIFF / NumPy array, with true-color RGB preview (B4, B3, B2)",
        known_limitations=[
            "Requires co-registered Sentinel-1 SAR imagery (VV/VH in dB).",
            "Requires all 13 Sentinel-2 spectral bands.",
            "GPLv3 copyleft license.",
        ],
        blockers=[],
    )


def _build_cloudgan_card() -> ModelCard:
    return ModelCard(
        model_id="cloudgan",
        display_name="CloudGAN",
        description=(
            "Detecting and Removing Clouds from RGB Satellite Images using "
            "an Auto-Encoder and SN-PatchGAN Image Inpainting."
        ),
        paper_citation="Jerry Schonenberg (2022). CloudGAN: Detecting and Removing Clouds from RGB-images using Image Inpainting.",
        paper_url="https://github.com/JerrySchonenberg/CloudGAN",
        license_type="MIT (repo) / Apache 2.0 (SN-PatchGAN)",
        has_explicit_repo_license=True,
        license_notes="MIT License applies to repository wrapper; Apache 2.0 applies to cloud_removal (SN-PatchGAN).",
        audit_status=AuditStatus.AUDITED,
        configuration_status=ConfigurationStatus.CONFIGURED,
        implementation_status=ImplementationStatus.NOT_IMPLEMENTED,
        verification_status=VerificationStatus.UNVERIFIED,
        readiness_status=ReadinessStatus.BLOCKED_MISSING_WEIGHTS,
        checkpoint_present_locally=False,
        checkpoint_files=[],
        local_weights_directory=None,
        framework="TensorFlow",
        framework_version="TensorFlow 1.15.0 (`tf.contrib` graph/session API)",
        python_version_support="Python 3.6.13",
        dependency_constraints=[
            "tensorflow-gpu==1.15.0",
            "neuralgym",
            "h5py==2.10.0",
            "numpy==1.19.5",
            "scikit-image==0.17.2",
            "opencv-python==4.6.0.66",
        ],
        input_requirements=InputRequirements(
            required_modalities={Modality.OPTICAL},
            required_sensors={SensorType.GENERIC_RGB, SensorType.SENTINEL_2},
            required_optical_bands={"B4", "B3", "B2"},
            required_sar_polarizations=set(),
            min_temporal_observations=1,
            expected_channels=3,
            channel_ordering_description="[R, G, B] (256x256 RGB image)",
            spatial_resolution_meters=None,
            required_spatial_dimensions=(256, 256),
            dimension_multiple_of=None,
            requires_cloud_mask=False,
            accepts_cloud_mask=True,
            requires_coregistration=False,
        ),
        output_description="Inpainted 3-channel RGB image",
        output_bands=["R", "G", "B"],
        output_range="[0, 255] uint8",
        output_format="3-band RGB PNG / JPEG (256x256)",
        known_limitations=[
            "No pretrained weights exist locally in the repository (external Google Drive links only).",
            "Strictly hardcoded to 256x256 pixel images.",
            "Requires deprecated TensorFlow 1.15 and tf.contrib (permanently removed in TF 2.0+).",
            "Inpainting approach hallucinates missing textures from cloud-free regions rather than recovering physical surface reflectance.",
        ],
        blockers=[
            "No pretrained weights found on disk (missing local checkpoints).",
            "Code relies on tf.contrib.framework and TensorFlow 1.15 Session API, incompatible with modern Python.",
        ],
    )


class ModelRegistry:
    """Central registry of all candidate models in GeoClear.
    
    Provides query methods, metadata inspection, and status tracking.
    """

    _MODELS: Dict[str, ModelCard] = {
        "ecrformer": _build_ecrformer_card(),
        "diffcr": _build_diffcr_card(),
        "dsen2_cr": _build_dsen2_cr_card(),
        "cloudgan": _build_cloudgan_card(),
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
