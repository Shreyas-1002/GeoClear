#!/usr/bin/env python3
"""Run real-data inference with DSen2-CR on data/real_patch and evaluate against temporal reference."""

import os
import sys
import time
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from geoclear.adapters.dsen2cr_adapter import DSen2CRAdapter
from geoclear.ingestion import discover_local_patch, load_local_patch, save_inference_outputs
from geoclear.metrics import compute_evaluation_metrics


def main():
    print("=" * 70)
    print("DSEN2-CR REAL-DATA INFERENCE PIPELINE")
    print("=" * 70)

    patch_dir = PROJECT_ROOT / "data" / "real_patch"
    discovered = discover_local_patch(patch_dir)
    print(f"Patch Directory:    {patch_dir}")
    print(f"Discovered SAR:     {discovered.sar_path.name if discovered.sar_path else 'None'}")
    print(f"Discovered Optical: {discovered.opt_path.name if discovered.opt_path else 'None'}")
    print(f"Discovered Ref:     {discovered.ref_path.name if discovered.ref_path else 'None'}")

    sar_arr, opt_arr, ref_arr, patch_metadata = load_local_patch(discovered, require_divisible_by=16)
    print(f"SAR array shape:     {sar_arr.shape}, range: [{sar_arr.min():.2f}, {sar_arr.max():.2f}] dB")
    print(f"Optical array shape: {opt_arr.shape}, range: [{opt_arr.min():.1f}, {opt_arr.max():.1f}] DN")

    # Instantiate adapter on CPU
    adapter = DSen2CRAdapter(device="cpu")
    print("\nExecuting forward pass on CPU...")
    t0 = time.perf_counter()
    result = adapter.predict(sar_arr, opt_arr)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Forward pass completed in {latency_ms:.2f} ms")

    recon = result.reconstructed_multispectral
    print(f"Reconstructed Shape: {recon.shape}")
    print(f"Reconstructed Range: [{recon.min():.4f}, {recon.max():.4f}] (reflectance)")
    print(f"Finite Values:       {np.all(np.isfinite(recon))}")

    # Quantitative comparison with temporal reference
    eval_metrics = None
    if ref_arr is not None:
        ref_norm = np.clip(ref_arr / 10000.0, 0.0, 1.0)
        eval_metrics = compute_evaluation_metrics(recon, ref_norm)
        print("\nQuantitative Comparison against Temporal Reference (2023-06-25, 17 days prior):")
        print(f"  Overall PSNR: {eval_metrics.psnr:.2f} dB")
        print(f"  Overall SSIM: {eval_metrics.ssim:.4f}")
        print(f"  Overall MAE:  {eval_metrics.mae:.4f}")
        print(f"  Overall RMSE: {eval_metrics.rmse:.4f}")
        print(f"  RGB PSNR:     {eval_metrics.rgb_psnr:.2f} dB")
        print(f"  RGB SSIM:     {eval_metrics.rgb_ssim:.4f}")
        print("  NOTE: This comparison reflects seasonal/phenological changes over 17 days, not simultaneous ground truth.")

    # Save output artifacts
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / f"run_dsen2cr_real_{timestamp_str}"
    saved_paths = save_inference_outputs(
        output_dir=out_dir,
        result=result,
        patch_metadata=patch_metadata,
        patch_files=discovered,
        evaluation_metrics=eval_metrics,
    )
    print(f"\nArtifacts successfully written to: {out_dir}")
    for k, v in saved_paths.items():
        print(f"  - {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
