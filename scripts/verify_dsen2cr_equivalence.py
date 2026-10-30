#!/usr/bin/env python3
"""DSen2-CR Numerical Equivalence Verification Script.

Compares:
1. An independent reference evaluator that extracts the raw Keras HDF5 kernels,
   computes the 34 convolutions with PyTorch functional ops according to the
   original Keras DSen2CR_model computational graph.
2. The converted native PyTorch architecture (ResnetStackedArchitecture) loaded
   from dsen2cr_carl.pth.

Test conditions:
- Deterministic random input: shape (1, 15, 256, 256)
- Normalized optical channels (0..12) in [0.0, 5.0]
- Normalized SAR channels (13..14) in [0.0, 2.0]
- Seed: 42
- Metrics: Output shapes, Mean Absolute Error (MAE), Maximum Absolute Error (MaxAE),
  finite value checks (NaN/Inf), numerical stability.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from scripts.convert_dsen2cr_weights import ResnetStackedArchitecture


def run_equivalence_test(
    h5_path: str | Path = "models/DSen2-CR/checkpoints/model_SARcarl.hdf5",
    pth_path: str | Path = "models/DSen2-CR/checkpoints/dsen2cr_carl.pth",
    patch_size: int = 256,
    seed: int = 42,
    mae_tol: float = 1e-6,
    maxae_tol: float = 1e-5,
) -> dict:
    h5_path = Path(h5_path)
    pth_path = Path(pth_path)

    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 checkpoint not found: {h5_path}")
    if not pth_path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {pth_path}")

    print("=" * 70)
    print("DSEN2-CR NUMERICAL EQUIVALENCE TEST")
    print("=" * 70)
    print(f"HDF5 Checkpoint:    {h5_path}")
    print(f"PyTorch Checkpoint: {pth_path}")
    print(f"Test Tensor Grid:   (1, 15, {patch_size}, {patch_size})")
    print(f"Fixed Random Seed:  {seed}")
    print(f"Tolerances:         MAE <= {mae_tol}, MaxAE <= {maxae_tol}")
    print("-" * 70)

    # 1. Synthesize deterministic input
    np.random.seed(seed)
    torch.manual_seed(seed)

    x_opt_np = np.random.uniform(0.0, 5.0, size=(1, 13, patch_size, patch_size)).astype(np.float32)
    x_sar_np = np.random.uniform(0.0, 2.0, size=(1, 2, patch_size, patch_size)).astype(np.float32)

    x_opt_torch = torch.from_numpy(x_opt_np)
    x_sar_torch = torch.from_numpy(x_sar_np)
    x_combined = torch.cat([x_opt_torch, x_sar_torch], dim=1)

    print(f"Input Optical range: [{x_opt_np.min():.4f}, {x_opt_np.max():.4f}]")
    print(f"Input SAR range:     [{x_sar_np.min():.4f}, {x_sar_np.max():.4f}]")
    print(f"Combined Input shape: {tuple(x_combined.shape)}")

    # 2. Run forward pass with converted PyTorch model
    print("\n[1/3] Executing converted PyTorch model...")
    net = ResnetStackedArchitecture()
    state_dict = torch.load(pth_path, map_location="cpu")
    net.load_state_dict(state_dict)
    net.eval()

    with torch.no_grad():
        out_pytorch_tensor = net(x_combined)
        out_pytorch = out_pytorch_tensor.numpy()

    # 3. Run forward pass with independent reference evaluator using raw HDF5 weights
    print("[2/3] Executing independent reference evaluator from raw HDF5 tensors...")
    with h5py.File(h5_path, "r") as h5:
        # Layer 1: conv2d_1 + ReLU
        w1_raw = h5["conv2d_1"]["conv2d_1_1"]["kernel:0"][()]
        b1_raw = h5["conv2d_1"]["conv2d_1_1"]["bias:0"][()]
        w1 = torch.from_numpy(w1_raw.transpose(3, 2, 0, 1)).float()
        b1 = torch.from_numpy(b1_raw).float()

        cur = F.conv2d(x_combined, w1, b1, padding=1)
        cur = F.relu(cur)

        # 16 Residual blocks
        for i in range(16):
            c_a = 2 * i + 2
            c_b = 2 * i + 3

            wa_raw = h5[f"conv2d_{c_a}"][f"conv2d_{c_a}_1"]["kernel:0"][()]
            ba_raw = h5[f"conv2d_{c_a}"][f"conv2d_{c_a}_1"]["bias:0"][()]
            wb_raw = h5[f"conv2d_{c_b}"][f"conv2d_{c_b}_1"]["kernel:0"][()]
            bb_raw = h5[f"conv2d_{c_b}"][f"conv2d_{c_b}_1"]["bias:0"][()]

            wa = torch.from_numpy(wa_raw.transpose(3, 2, 0, 1)).float()
            ba = torch.from_numpy(ba_raw).float()
            wb = torch.from_numpy(wb_raw.transpose(3, 2, 0, 1)).float()
            bb = torch.from_numpy(bb_raw).float()

            tmp = F.conv2d(cur, wa, ba, padding=1)
            tmp = F.relu(tmp)
            tmp = F.conv2d(tmp, wb, bb, padding=1)
            tmp = tmp * 0.1  # res_scale
            cur = cur + tmp

        # Layer 34: conv2d_34
        w34_raw = h5["conv2d_34"]["conv2d_34_1"]["kernel:0"][()]
        b34_raw = h5["conv2d_34"]["conv2d_34_1"]["bias:0"][()]
        w34 = torch.from_numpy(w34_raw.transpose(3, 2, 0, 1)).float()
        b34 = torch.from_numpy(b34_raw).float()

        res = F.conv2d(cur, w34, b34, padding=1)
        # Global long-skip connection (add optical input)
        out_ref_tensor = x_combined[:, :-2, ...] + res
        out_ref = out_ref_tensor.numpy()

    # 4. Numerical comparison and verification
    print("[3/3] Computing discrepancy metrics...")
    diff = np.abs(out_pytorch - out_ref)
    mae = float(np.mean(diff))
    max_ae = float(np.max(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    is_finite_pyt = bool(np.all(np.isfinite(out_pytorch)))
    is_finite_ref = bool(np.all(np.isfinite(out_ref)))
    shapes_match = out_pytorch.shape == out_ref.shape

    mae_passed = mae <= mae_tol
    maxae_passed = max_ae <= maxae_tol
    overall_passed = shapes_match and is_finite_pyt and is_finite_ref and mae_passed and maxae_passed

    print("-" * 70)
    print("VERIFICATION RESULTS:")
    print(f"PyTorch Output Shape:    {out_pytorch.shape}")
    print(f"Reference Output Shape:  {out_ref.shape}")
    print(f"Shapes Identical:        {shapes_match}")
    print(f"PyTorch Output Finite:   {is_finite_pyt}")
    print(f"Reference Output Finite: {is_finite_ref}")
    print(f"PyTorch Value Range:     [{out_pytorch.min():.5f}, {out_pytorch.max():.5f}]")
    print(f"Reference Value Range:   [{out_ref.min():.5f}, {out_ref.max():.5f}]")
    print(f"Mean Absolute Error:     {mae:.10e} (threshold <= {mae_tol}) -> {'PASS' if mae_passed else 'FAIL'}")
    print(f"Maximum Absolute Error:  {max_ae:.10e} (threshold <= {maxae_tol}) -> {'PASS' if maxae_passed else 'FAIL'}")
    print(f"Root Mean Squared Error: {rmse:.10e}")
    print(f"Overall Status:          {'PASS' if overall_passed else 'FAIL'}")
    print("=" * 70)

    if not overall_passed:
        raise AssertionError("Numerical equivalence verification failed!")

    return {
        "shapes_match": shapes_match,
        "is_finite": is_finite_pyt and is_finite_ref,
        "mae": mae,
        "max_ae": max_ae,
        "rmse": rmse,
        "overall_passed": overall_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify DSen2-CR converted checkpoint numerical equivalence")
    parser.add_argument("--h5-path", default="models/DSen2-CR/checkpoints/model_SARcarl.hdf5")
    parser.add_argument("--pth-path", default="models/DSen2-CR/checkpoints/dsen2cr_carl.pth")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_equivalence_test(
        h5_path=args.h5_path,
        pth_path=args.pth_path,
        patch_size=args.patch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
