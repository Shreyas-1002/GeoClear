"""GeoClear model adapters.

Provides dedicated, dependency-isolated model execution adapters.
"""

from geoclear.adapters.ecrformer_adapter import ECRformerAdapter, ECRformerInferenceResult

__all__ = [
    "ECRformerAdapter",
    "ECRformerInferenceResult",
]
