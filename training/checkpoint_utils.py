# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for checkpoint saving, loading, model warm-start, and training resumption."""

import os
from typing import Any, Dict, Optional, Tuple
import torch


def load_checkpoint_for_training(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    resume: bool = False,
    expected_in_features: Optional[int] = None,
    expected_out_features: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Load a checkpoint for either continuing training (resume=True) or warm-starting (resume=False).

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model: PyTorch model to load weights into.
        optimizer: PyTorch optimizer (required if resume=True).
        scheduler: PyTorch learning rate scheduler (optional, for resume=True).
        device: Target torch.device.
        resume: If True, restores optimizer and scheduler states and returns iteration + 1.
                If False, only loads model weights and returns 0.
        expected_in_features: Expected input dimensionality (for architecture verification).
        expected_out_features: Expected output dimensionality.

    Returns:
        Tuple of (start_iter, metadata_dict).
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    else:
        # Bare state_dict
        state_dict = checkpoint

    # Validate model configuration if present in metadata
    model_config = metadata.get("model_config")
    if model_config and isinstance(model_config, dict):
        ckpt_in = model_config.get("in_features")
        ckpt_out = model_config.get("out_features")
        if expected_in_features is not None and ckpt_in is not None and ckpt_in != expected_in_features:
            raise ValueError(
                f"Checkpoint in_features ({ckpt_in}) does not match expected in_features ({expected_in_features})"
            )
        if expected_out_features is not None and ckpt_out is not None and ckpt_out != expected_out_features:
            raise ValueError(
                f"Checkpoint out_features ({ckpt_out}) does not match expected out_features ({expected_out_features})"
            )

    # Load model parameters
    model.load_state_dict(state_dict)
    print("Successfully loaded model weights.")

    if not resume:
        # Pretrained initialization only (warm start)
        print("Warm-start mode: Starting fresh training run from iteration 0.")
        return 0, metadata

    # Resume training: Restore optimizer and scheduler states
    ckpt_iter = int(metadata.get("iteration", 0))
    start_iter = ckpt_iter + 1

    if optimizer is not None and "optimizer_state_dict" in metadata:
        optimizer.load_state_dict(metadata["optimizer_state_dict"])
        # Ensure optimizer state tensors are on the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print(f"Restored optimizer state from checkpoint (checkpoint iteration: {ckpt_iter}).")

    if scheduler is not None:
        if "scheduler_state_dict" in metadata and metadata["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(metadata["scheduler_state_dict"])
            print(f"Restored scheduler state_dict (last_epoch={scheduler.last_epoch}).")
        else:
            # Backward-compatibility for older checkpoints without scheduler_state_dict
            scheduler.last_epoch = start_iter
            # Update learning rate in optimizer param_groups to match schedule at start_iter
            for param_group in optimizer.param_groups:
                initial_lr = param_group.get("initial_lr", param_group["lr"])
                # Apply decay if scheduler has lr_lambdas
                if hasattr(scheduler, "lr_lambdas") and scheduler.lr_lambdas:
                    factor = scheduler.lr_lambdas[0](start_iter)
                    param_group["lr"] = initial_lr * factor
            print(f"Synchronized scheduler step to iteration {start_iter} (LR={optimizer.param_groups[0]['lr']:.2e}).")

    print(f"Resuming training starting at iteration {start_iter}.")
    return start_iter, metadata


def save_training_checkpoint(
    save_path: str,
    iteration: int,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a comprehensive training checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    checkpoint_data = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }
    if extra_metadata:
        checkpoint_data.update(extra_metadata)
    torch.save(checkpoint_data, save_path)
