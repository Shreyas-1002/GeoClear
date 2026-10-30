#!/usr/bin/env python3
"""DSen2-CR Checkpoint Converter: Keras HDF5 -> PyTorch State Dict.

Reads the official Keras HDF5 checkpoint from DSen2-CR (model_SARcarl.hdf5),
inspects all 68 trainable weight and bias tensors, transposes Conv2D weights
from Keras (H, W, In, Out) to PyTorch (Out, In, H, W), copies biases directly,
populates the PyTorch ResnetStackedArchitecture state dictionary, and saves
the converted weights to dsen2cr_carl.pth.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """Residual block matching DSen2-CR definition."""
    def __init__(self, dim: int, use_dropout: bool = False, use_bias: bool = True, res_scale: float = 0.1, padding_size: int = 1):
        super(ResnetBlock, self).__init__()
        self.res_scale = res_scale
        self.padding_size = padding_size
        conv_block = [
            nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=self.padding_size, bias=use_bias),
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.conv_block(x)


class ResnetStackedArchitecture(nn.Module):
    """Native PyTorch implementation of DSen2-CR architecture."""
    def __init__(self, in_channels: int = 15, out_channels: int = 13, F: int = 256, B: int = 16, res_scale: float = 0.1):
        super(ResnetStackedArchitecture, self).__init__()
        self.F = F
        self.B = B
        self.scale_res = res_scale
        model = [
            nn.Conv2d(in_channels, self.F, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
        ]
        for i in range(self.B):
            model.append(ResnetBlock(self.F, use_dropout=False, use_bias=True, res_scale=self.scale_res, padding_size=1))
        model.append(nn.Conv2d(self.F, out_channels, kernel_size=3, padding=1, bias=True))
        self.model = nn.Sequential(*model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Long skip-connection: add cloudy MS optical input (first 13 channels) to model output
        return input[:, :-2, ...] + self.model(input)


def build_layer_mapping() -> Dict[str, Tuple[str, str]]:
    """Build the exact mapping between Keras conv2d layers and PyTorch parameter names.
    
    Returns:
        dict: keras_layer_name -> (pytorch_weight_param, pytorch_bias_param)
    """
    mapping = {}
    
    # 1. Input head convolution
    mapping["conv2d_1"] = ("model.0.weight", "model.0.bias")
    
    # 2. Body residual blocks (16 blocks, each with 2 convs)
    # Block index 0 corresponds to model.2 (model.1 is ReLU)
    # Block i (i=0..15) corresponds to model.{i+2}
    for b in range(16):
        block_module_idx = b + 2
        conv_first = 2 * b + 2
        conv_second = 2 * b + 3
        
        mapping[f"conv2d_{conv_first}"] = (
            f"model.{block_module_idx}.conv_block.0.weight",
            f"model.{block_module_idx}.conv_block.0.bias",
        )
        mapping[f"conv2d_{conv_second}"] = (
            f"model.{block_module_idx}.conv_block.2.weight",
            f"model.{block_module_idx}.conv_block.2.bias",
        )
        
    # 3. Output tail convolution (conv2d_34 -> model.18)
    mapping["conv2d_34"] = ("model.18.weight", "model.18.bias")
    
    return mapping


def convert_checkpoint(h5_path: str | Path, output_pth_path: str | Path) -> None:
    """Convert Keras HDF5 checkpoint to PyTorch state dict with strict validation."""
    h5_path = Path(h5_path)
    output_pth_path = Path(output_pth_path)

    if not h5_path.is_file():
        raise FileNotFoundError(f"Input HDF5 checkpoint not found: {h5_path}")
    if output_pth_path.exists():
        raise FileExistsError(f"Target checkpoint already exists: {output_pth_path}. Will not overwrite.")

    print(f"Loading Keras checkpoint from: {h5_path}")
    print(f"File size: {h5_path.stat().st_size:,} bytes")

    # Instantiate PyTorch target architecture
    net = ResnetStackedArchitecture()
    pyt_state = net.state_dict()
    pyt_param_names = set(pyt_state.keys())
    print(f"Target PyTorch model parameter count: {len(pyt_param_names)}")

    mapping = build_layer_mapping()
    new_state_dict: Dict[str, torch.Tensor] = {}

    with h5py.File(h5_path, "r") as h5:
        print("Inspecting HDF5 groups...")
        tensors_converted = 0

        for conv_name, (w_name, b_name) in mapping.items():
            if conv_name not in h5:
                raise KeyError(f"Expected layer group '{conv_name}' not found in HDF5 file!")

            grp = h5[conv_name]
            # Handle Keras nested subgroup if present (e.g. conv2d_1/conv2d_1_1/kernel:0)
            subgroups = list(grp.keys())
            if len(subgroups) == 1 and isinstance(grp[subgroups[0]], h5py.Group):
                target_grp = grp[subgroups[0]]
            else:
                target_grp = grp

            if "kernel:0" not in target_grp or "bias:0" not in target_grp:
                raise KeyError(f"Missing kernel:0 or bias:0 in {conv_name} (keys: {list(target_grp.keys())})")

            kernel_np = target_grp["kernel:0"][()]  # (H, W, In, Out)
            bias_np = target_grp["bias:0"][()]      # (Out,)

            # Permute kernel dimensions: (H, W, In, Out) -> (Out, In, H, W)
            kernel_transposed = np.transpose(kernel_np, (3, 2, 0, 1))

            w_tensor = torch.from_numpy(kernel_transposed).float()
            b_tensor = torch.from_numpy(bias_np).float()

            # Shape validation against target PyTorch model
            expected_w_shape = pyt_state[w_name].shape
            expected_b_shape = pyt_state[b_name].shape

            if w_tensor.shape != expected_w_shape:
                raise ValueError(
                    f"Kernel shape mismatch for {w_name}: got {w_tensor.shape}, expected {expected_w_shape} "
                    f"(original Keras shape {kernel_np.shape})"
                )
            if b_tensor.shape != expected_b_shape:
                raise ValueError(
                    f"Bias shape mismatch for {b_name}: got {b_tensor.shape}, expected {expected_b_shape}"
                )

            new_state_dict[w_name] = w_tensor
            new_state_dict[b_name] = b_tensor
            tensors_converted += 2

    # Comprehensive verification
    converted_names = set(new_state_dict.keys())
    missing_params = pyt_param_names - converted_names
    extra_params = converted_names - pyt_param_names

    if missing_params:
        raise RuntimeError(f"Conversion incomplete! Missing PyTorch parameters: {sorted(missing_params)}")
    if extra_params:
        raise RuntimeError(f"Unexpected extra parameters converted: {sorted(extra_params)}")

    # Load into PyTorch model to ensure state_dict compatibility
    net.load_state_dict(new_state_dict)
    print(f"Successfully populated all {tensors_converted} PyTorch model parameter tensors.")

    # Save state dict
    output_pth_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_state_dict, output_pth_path)
    print(f"Saved converted PyTorch checkpoint to: {output_pth_path}")
    print(f"Converted file size: {output_pth_path.stat().st_size:,} bytes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DSen2-CR Keras HDF5 checkpoint to PyTorch state dict")
    parser.add_argument(
        "--h5-path",
        default="models/DSen2-CR/checkpoints/model_SARcarl.hdf5",
        help="Path to official model_SARcarl.hdf5",
    )
    parser.add_argument(
        "--output-pth",
        default="models/DSen2-CR/checkpoints/dsen2cr_carl.pth",
        help="Target output path for PyTorch checkpoint",
    )
    args = parser.parse_args()

    convert_checkpoint(args.h5_path, args.output_pth)


if __name__ == "__main__":
    main()
