"""Shared training setup and logging."""
from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from data.schema import get_device, load_config, save_json, set_seed


def arguments(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true", help="use the separate smoke configuration and artifacts")
    return parser


def setup(args: argparse.Namespace, *, device_override: str | None = None) -> tuple[dict, torch.device]:
    cfg = load_config(args.config, smoke=args.smoke)
    device = torch.device(device_override) if device_override else get_device(cfg)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    set_seed(cfg["seed"])
    return cfg, device


def mixed_precision(cfg: dict, device: torch.device):
    """BF16 where available; use FP32 on CPU or older GPUs."""
    if cfg["training"]["amp"] and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext


def batch_slices(n: int, batch_size: int) -> Iterator[slice]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, n, batch_size):
        yield slice(start, start + batch_size)


def writer_for(cfg: dict, stage: str) -> SummaryWriter:
    return SummaryWriter(str(Path(cfg["paths"]["output_dir"]) / "tensorboard" / stage))


def log_epoch(writer: SummaryWriter, history: list, epoch: int, metrics: dict) -> None:
    for name, value in metrics.items():
        writer.add_scalar(name, value, epoch)
    history.append({"epoch": epoch, **metrics})
    writer.flush()


def write_history(cfg: dict, stage: str, history: list) -> None:
    save_json(Path(cfg["paths"]["output_dir"]) / f"{stage}_history.json", history)


def optimize(loss: torch.Tensor, optimizer: torch.optim.Optimizer, parameters: list, clip: float) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(parameters, clip, error_if_nonfinite=True)
    optimizer.step()


def mixed_indices(n: int, generation_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Visit each edit once and sample enough generation examples for the requested fraction."""
    if not 0 <= generation_ratio < 1:
        raise ValueError("generation_ratio must be in [0, 1)")
    n_gen = round(n * generation_ratio / (1 - generation_ratio))
    indices = np.concatenate((np.arange(n), rng.choice(n, size=n_gen, replace=n_gen > n)))
    edit = np.arange(len(indices)) < n
    order = rng.permutation(len(indices))
    return indices[order], edit[order]
