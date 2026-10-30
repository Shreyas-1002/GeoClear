"""Strict data-compatibility engine for GeoClear.

Determines candidate model eligibility strictly based on user-declared data,
separating input compatibility from operational execution readiness.
"""

from __future__ import annotations

from typing import List, Optional, Set

from geoclear.types import (
    CompatibilityReport,
    CompatibilityResult,
    DeclaredData,
    ImplementationStatus,
    Modality,
    ModelCard,
    ReadinessStatus,
    SensorType,
    S2_BANDS_13,
)
from geoclear.registry import ModelRegistry


class CompatibilityEngine:
    """Deterministic compatibility evaluator for Earth-observation models."""

    @classmethod
    def evaluate(
        cls,
        declared_data: DeclaredData,
        models: Optional[List[ModelCard]] = None,
    ) -> CompatibilityReport:
        """Evaluate declared data against all registered models.

        Args:
            declared_data: The exact dataset characteristics declared by the user.
            models: Optional list of model cards to evaluate. If None, evaluates all models in ModelRegistry.

        Returns:
            CompatibilityReport containing compatible, incompatible, and executable models.

        Raises:
            ValueError: If declared_data contains logical contradictions or invalid parameters.
        """
        declared_data.validate()

        if models is None:
            models = ModelRegistry.list_models()

        report = CompatibilityReport(declared_data=declared_data)

        for model in models:
            result = cls._evaluate_single_model(declared_data, model)
            if result.is_compatible:
                report.compatible_models.append(result)
                if result.is_executable:
                    report.executable_models.append(result)
            else:
                report.incompatible_models.append(result)

        return report

    @classmethod
    def _evaluate_single_model(
        cls, declared_data: DeclaredData, model: ModelCard
    ) -> CompatibilityResult:
        missing_inputs: List[str] = []
        exclusion_reasons: List[str] = []
        execution_blockers: List[str] = []
        notes: List[str] = []

        req = model.input_requirements

        # 1. Modality evaluation
        for mod in req.required_modalities:
            if mod not in declared_data.modalities:
                if mod == Modality.SAR:
                    missing_inputs.append("Sentinel-1 SAR imagery (VV/VH polarizations)")
                    exclusion_reasons.append(
                        "Requires co-registered Sentinel-1 SAR imagery; SAR data was not declared."
                    )
                elif mod == Modality.MULTISPECTRAL:
                    missing_inputs.append("13-band multispectral optical data (Sentinel-2)")
                    exclusion_reasons.append(
                        "Requires full multispectral optical imagery; multispectral data was not declared."
                    )
                elif mod == Modality.MULTI_TEMPORAL:
                    missing_inputs.append(
                        f"Multi-temporal optical observations ({req.min_temporal_observations} dates required)"
                    )
                    exclusion_reasons.append(
                        f"Requires at least {req.min_temporal_observations} multi-temporal acquisitions of the same scene."
                    )
                elif mod == Modality.OPTICAL:
                    missing_inputs.append("Optical imagery")
                    exclusion_reasons.append("Requires optical imagery; no optical data was declared.")

        # 2. Sensor compatibility evaluation
        if req.required_sensors:
            # Check if declared sensors intersect with required sensors
            declared_sensor_match = any(s in req.required_sensors for s in declared_data.sensors)
            if not declared_sensor_match:
                missing_inputs.append(
                    f"Sensor: {', '.join(sorted(s.value for s in req.required_sensors))}"
                )
                exclusion_reasons.append(
                    f"Sensor mismatch: requires {', '.join(sorted(s.value for s in req.required_sensors))}, "
                    f"but declared sensor set is {[s.value for s in declared_data.sensors]}."
                )

        # 3. Spectral bands evaluation (strictly rule-based; never infer missing bands)
        if req.required_optical_bands:
            declared_bands = set(declared_data.optical_bands)
            required_bands = set(req.required_optical_bands)

            # Map generic RGB aliases if model accepts generic RGB
            if required_bands == {"B4", "B3", "B2"} and {"R", "G", "B"}.issubset(declared_bands):
                pass
            else:
                missing_bands = required_bands - declared_bands
                if missing_bands:
                    missing_inputs.append(
                        f"Missing optical bands: {', '.join(sorted(missing_bands))}"
                    )
                    if required_bands == set(S2_BANDS_13):
                        exclusion_reasons.append(
                            f"Requires all 13 Sentinel-2 spectral bands [B1-B12, B8A]; missing {len(missing_bands)} "
                            f"bands: {', '.join(sorted(missing_bands))}. (Converting RGB to TIFF does not synthesize "
                            f"missing physical NIR, SWIR, or red-edge bands)."
                        )
                    else:
                        exclusion_reasons.append(
                            f"Requires spectral bands {', '.join(sorted(required_bands))}; missing {', '.join(sorted(missing_bands))}."
                        )

        # 4. SAR polarizations evaluation
        if req.required_sar_polarizations:
            missing_pols = req.required_sar_polarizations - declared_data.sar_polarizations
            if missing_pols:
                missing_inputs.append(
                    f"Missing SAR polarizations: {', '.join(sorted(missing_pols))}"
                )
                exclusion_reasons.append(
                    f"Requires SAR polarizations {', '.join(sorted(req.required_sar_polarizations))}; "
                    f"missing {', '.join(sorted(missing_pols))}."
                )

        # 5. Temporal observation count evaluation
        if req.min_temporal_observations > 1:
            if declared_data.temporal_observations < req.min_temporal_observations:
                missing_inputs.append(
                    f"{req.min_temporal_observations} co-registered temporal observations"
                )
                exclusion_reasons.append(
                    f"Requires at least {req.min_temporal_observations} temporal observations of the same scene; "
                    f"{declared_data.temporal_observations} observation declared."
                )

        # 6. Cloud mask requirement evaluation
        if req.requires_cloud_mask and not declared_data.has_cloud_mask:
            missing_inputs.append("Pre-computed cloud mask")
            exclusion_reasons.append("Model requires a pre-computed cloud mask input.")

        # 7. Spatial dimensions evaluation
        if req.required_spatial_dimensions is not None:
            req_h, req_w = req.required_spatial_dimensions
            if declared_data.image_height is not None and declared_data.image_width is not None:
                if (declared_data.image_height, declared_data.image_width) != (req_h, req_w):
                    missing_inputs.append(f"Image dimensions of exactly {req_h}x{req_w} pixels")
                    exclusion_reasons.append(
                        f"Requires fixed dimensions of {req_h}x{req_w} pixels; declared dimensions are "
                        f"{declared_data.image_height}x{declared_data.image_width}."
                    )

        if req.dimension_multiple_of is not None:
            mult = req.dimension_multiple_of
            if declared_data.image_height is not None and declared_data.image_height % mult != 0:
                exclusion_reasons.append(
                    f"Height ({declared_data.image_height}) is not divisible by required multiple of {mult}."
                )
            if declared_data.image_width is not None and declared_data.image_width % mult != 0:
                exclusion_reasons.append(
                    f"Width ({declared_data.image_width}) is not divisible by required multiple of {mult}."
                )

        # Determine input compatibility
        is_compatible = len(exclusion_reasons) == 0

        # Determine operational execution readiness
        if not model.checkpoint_present_locally:
            execution_blockers.append(
                "Pretrained weights are not present on disk (external download required)."
            )

        if model.readiness_status == ReadinessStatus.BLOCKED_MISSING_WEIGHTS:
            if "Pretrained weights are not present on disk (external download required)." not in execution_blockers:
                execution_blockers.append("Blocked: missing pretrained weights.")
        elif model.readiness_status == ReadinessStatus.BLOCKED_CODE_BUG:
            execution_blockers.append(
                "Blocked: upstream code bug in repository (import path discrepancy)."
            )
        elif model.readiness_status == ReadinessStatus.BLOCKED_LEGACY_FRAMEWORK:
            execution_blockers.append(
                f"Blocked: requires legacy framework ({model.framework} {model.framework_version}) "
                f"incompatible with the current runtime environment."
            )
        elif model.readiness_status == ReadinessStatus.BLOCKED_UNIMPLEMENTED_ADAPTER:
            execution_blockers.append("Blocked: dedicated model adapter not yet implemented.")

        # Copy any specific blockers from model card
        for blk in model.blockers:
            if blk not in execution_blockers:
                execution_blockers.append(blk)

        is_executable = (
            is_compatible
            and model.checkpoint_present_locally
            and model.readiness_status == ReadinessStatus.READY_FOR_INFERENCE
            and model.implementation_status == ImplementationStatus.IMPLEMENTED
        )

        # Informational notes
        if not model.has_explicit_repo_license:
            notes.append(
                f"Licensing advisory: {model.display_name} has no explicit repository LICENSE file. "
                "Verify usage permissions before commercial deployment."
            )

        has_ref = declared_data.has_ground_truth_reference or getattr(declared_data, "has_clear_optical_reference", False)
        if not has_ref:
            notes.append(
                "No clear optical reference for evaluation provided. Reference-dependent metrics (PSNR, SSIM, MAE, RMSE) "
                "cannot be computed and will not be displayed."
            )

        return CompatibilityResult(
            model_id=model.model_id,
            model_name=model.display_name,
            is_compatible=is_compatible,
            is_executable=is_executable,
            missing_inputs=missing_inputs,
            exclusion_reasons=exclusion_reasons,
            execution_blockers=execution_blockers,
            notes=notes,
        )
