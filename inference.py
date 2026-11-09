"""Structured-token inference for one or more 28x28 RGB images."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data.dataset import denormalize_images, normalize_images
from data.schema import OP_EDIT, OP_GENERATE, edit_prompt, generate_prompt, get_device, load_config, understand_prompt
from evaluation.metrics import SceneOracle
from models.checkpoints import checkpoint_path, load_flow, load_understanding
from models.flow import FlowCondition, sample_flow
from models.tiny_umm import TinyUMM


@dataclass
class EditResult:
    raw: np.ndarray
    constrained: np.ndarray
    region: np.ndarray


def constrain_edit(source: np.ndarray, edited: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Copy bytes from the source outside the caller's chosen region."""
    return np.where(region[..., None], edited, source)


class Pipeline:
    def __init__(self, cfg: dict, device: torch.device, checkpoint: str | Path | None = None):
        self.cfg = cfg
        self.device = device
        self.checkpoint = Path(checkpoint or checkpoint_path(cfg, "connector"))
        self.model, self.metadata = load_flow(cfg, self.checkpoint, device)

    @cached_property
    def oracle(self) -> SceneOracle:
        return SceneOracle(self.cfg)

    def regions(self, images: np.ndarray) -> np.ndarray:
        return np.stack([self.oracle.region(self.oracle.parse(image)) for image in images])

    def _tokens(self, prompts: np.ndarray, operation: int) -> torch.Tensor:
        prompts = np.asarray(prompts)
        if prompts.ndim != 2 or prompts.shape[1] != 4 or not len(prompts) or np.any(prompts[:, 0] != operation):
            raise ValueError("expected a nonempty batch of four-slot prompts for this operation")
        return torch.as_tensor(prompts, dtype=torch.long, device=self.device)

    def _sample(
        self, condition: FlowCondition, steps: int | None, seed: int | None, gain: float, *, prompt: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> np.ndarray:
        settings = self.cfg["sampling"]
        seed = self.cfg["seed"] if seed is None else seed
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if self.model.bottleneck is not None:
            latent_noise = torch.Generator(device=self.device).manual_seed(seed ^ 0x5EED)
            condition, _, _ = self.model.bottleneck(condition, latent_noise, prompt=prompt, active=active)
        condition.instruction = gain * condition.instruction
        result = sample_flow(
            self.model.flow, condition, steps=settings["steps"] if steps is None else steps,
            solver=settings["solver"], generator=generator,
        )
        return denormalize_images(result)

    @torch.inference_mode()
    def generate(
        self, prompts: np.ndarray, *, steps: int | None = None, seed: int | None = None,
        gain: float | None = None,
    ) -> np.ndarray:
        tokens = self._tokens(prompts, OP_GENERATE)
        if gain is None:
            gain = self.cfg["sampling"]["generation_instruction_gain"]
        return self._sample(self.model.condition(tokens), steps, seed, gain, prompt=tokens)

    @torch.inference_mode()
    def edit(
        self, images: np.ndarray, prompts: np.ndarray, *, region: np.ndarray | None = None,
        steps: int | None = None, seed: int | None = None, gain: float | None = None,
        direct: bool | None = None,
    ) -> EditResult:
        tokens = self._tokens(prompts, OP_EDIT)
        size = self.cfg["image_size"]
        images = np.asarray(images)
        if images.shape != (len(tokens), size, size, 3) or images.dtype != np.uint8:
            raise ValueError(f"expected one uint8 {size}x{size} RGB image per edit prompt")
        if region is None:
            region = self.regions(images)
        else:
            region = np.asarray(region)
            if region.shape == (size, size):
                region = np.broadcast_to(region, images.shape[:3])
            if region.shape != images.shape[:3] or not np.isin(region, (0, 1, 255)).all():
                raise ValueError(f"expected a binary {size}x{size} region (or one per image)")
            region = region.astype(bool)
        if direct is None:
            direct = self.metadata["direct_source"]
        if gain is None:
            gain = self.cfg["sampling"]["edit_instruction_gain"]
        source = torch.as_tensor(normalize_images(images), device=self.device)
        condition = self.model.condition(tokens, source, direct=direct)
        active = None
        if self.metadata.get("posttrain", {}).get("same_color_edits"):
            active = self.model.keeps_source_color(tokens, source)
        raw = self._sample(condition, steps, seed, gain, prompt=tokens, active=active)
        return EditResult(raw, constrain_edit(images, raw, region), region)

    def chain(
        self, images: np.ndarray, rounds: list[np.ndarray], *, region: np.ndarray | None = None,
        steps: int | None = None, seed: int | None = None, use_raw: bool = False,
    ) -> list[EditResult]:
        """Edit in sequence, reusing the first source's region on every round."""
        seed = self.cfg["seed"] if seed is None else seed
        if region is None:
            region = self.regions(images)
        results = []
        for turn, prompt in enumerate(rounds):
            result = self.edit(images, prompt, region=region, steps=steps, seed=seed + turn)
            images = result.raw if use_raw else result.constrained
            results.append(result)
        return results


@torch.inference_mode()
def predict_understanding(
    model, images: np.ndarray, cfg: dict, device: torch.device, *, seed: int | None = None,
) -> dict[str, np.ndarray]:
    batch_size = cfg["evaluation"]["batch_size"]
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be positive")
    token = torch.as_tensor(understand_prompt(cfg), dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] if seed is None else seed)
    predictions = {"count": [], "color": [], "shape": []}
    for start in range(0, len(images), batch_size):
        x = torch.as_tensor(normalize_images(images[start:start + batch_size]), device=device)
        if isinstance(model, TinyUMM):
            logits = model.classify(x, generator=generator)
        else:
            logits = model(x, token[None].expand(len(x), -1))
        for name, value in logits.items():
            predictions[name].append(value.argmax(dim=-1).cpu().numpy())
    return {name: np.concatenate(values) for name, values in predictions.items()}


def read_image(path: str | Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (size, size):
            raise ValueError(f"{path}: expected a {size}x{size} image, got {image.size[0]}x{image.size[1]}")
        return np.array(image.convert("RGB"))


def save_png(path: str | Path, image: Image.Image | np.ndarray) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"output must use the .png extension: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)
    return path


def save_grid(path: str | Path, rows: list[list[tuple[np.ndarray, str]]], scale: int = 4) -> Path:
    """Write a simple contact sheet; keep individual edits as lossless 28x28 PNGs."""
    height, width = rows[0][0][0].shape[:2]
    tile_w, tile_h = width * scale, height * scale
    gap, label_h = 8, 16
    ncols = max(map(len, rows))
    canvas = Image.new("RGB", (gap + ncols * (tile_w + gap), gap + len(rows) * (tile_h + gap + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for c, (image, label) in enumerate(row):
            x, y = gap + c * (tile_w + gap), gap + r * (tile_h + gap + label_h)
            draw.text((x, y), label, fill="#172236")
            canvas.paste(Image.fromarray(image).resize((tile_w, tile_h), Image.Resampling.NEAREST), (x, y + label_h))
    return save_png(path, canvas)


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _named_index(cfg: dict, field: str, name: str) -> int:
    names = cfg[field]["names"]
    try:
        return names.index(name.strip().casefold())
    except ValueError as exc:
        raise ValueError(f"unknown {field[:-1]} {name!r}; choose from {', '.join(names)}") from exc


def _edit_prompt(cfg: dict, color: str, shape: str) -> np.ndarray:
    return edit_prompt(cfg, _named_index(cfg, "colors", color), _named_index(cfg, "shapes", shape))


def _broadcast_list(value: str, n: int) -> list[str]:
    names = value.split(",")
    if len(names) == 1:
        return names * n
    if len(names) != n:
        raise ValueError(f"expected either one value or {n} comma-separated values")
    return names


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=positive_int)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("understand").add_argument("--image", required=True, type=Path)
    generate = sub.add_parser("generate")
    generate.add_argument("--count", required=True, type=positive_int)
    generate.add_argument("--color", required=True)
    generate.add_argument("--shape", required=True)
    generate.add_argument("--num", type=positive_int, default=1)
    generate.add_argument("--out", type=Path)
    edit = sub.add_parser("edit")
    edit.add_argument("--image", required=True, type=Path)
    edit.add_argument("--color", required=True)
    edit.add_argument("--shape", required=True)
    edit.add_argument("--region", type=Path, help="optional binary HxW NumPy .npy mask")
    edit.add_argument("--out", type=Path)
    batch = sub.add_parser("batch-edit")
    batch.add_argument("--images", required=True, nargs="+", type=Path)
    batch.add_argument("--colors", required=True)
    batch.add_argument("--shapes", required=True)
    batch.add_argument("--out-dir", type=Path)
    chain = sub.add_parser("chain")
    chain.add_argument("--image", required=True, type=Path)
    chain.add_argument("--instructions", required=True, help="color:shape pairs separated by semicolons")
    chain.add_argument("--out-dir", type=Path)
    return parser


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config, args.smoke)
    device = get_device(cfg)
    out = Path(cfg["paths"]["output_dir"])
    size = cfg["image_size"]
    seed = cfg["seed"] if args.seed is None else args.seed
    batch_size = cfg["evaluation"]["batch_size"]
    if args.command == "understand":
        model = load_understanding(cfg, device, args.checkpoint)
        prediction = predict_understanding(model, read_image(args.image, size)[None], cfg, device, seed=seed)
        return {
            "count": int(prediction["count"][0]) + 1,
            "color": cfg["colors"]["names"][prediction["color"][0]],
            "shape": cfg["shapes"]["names"][prediction["shape"][0]],
        }

    pipeline = Pipeline(cfg, device, args.checkpoint)
    if args.command == "generate":
        token = generate_prompt(cfg, args.count, _named_index(cfg, "colors", args.color), _named_index(cfg, "shapes", args.shape))
        images = []
        for start in range(0, args.num, batch_size):
            n = min(batch_size, args.num - start)
            images.extend(pipeline.generate(np.repeat(token[None], n, axis=0), steps=args.steps, seed=seed + start))
        path = args.out or out / "generate.png"
        if len(images) == 1:
            save_png(path, images[0])
        else:
            columns = min(8, math.ceil(math.sqrt(len(images))))
            tiles = [(img, str(i + 1)) for i, img in enumerate(images)]
            save_grid(path, [tiles[i:i + columns] for i in range(0, len(tiles), columns)])
        return {"output": str(path), "samples": len(images)}

    if args.command == "edit":
        source = read_image(args.image, size)
        token = _edit_prompt(cfg, args.color, args.shape)
        region = np.load(args.region, allow_pickle=False) if args.region is not None else None
        result = pipeline.edit(source[None], token[None], region=region, steps=args.steps, seed=seed)
        path = save_png(args.out or out / "edit.png", result.constrained[0])
        return {"output": str(path), "editable_pixels": int(result.region.sum())}

    if args.command == "batch-edit":
        n = len(args.images)
        prompts = np.stack([_edit_prompt(cfg, c, s) for c, s in zip(_broadcast_list(args.colors, n), _broadcast_list(args.shapes, n))])
        images = np.stack([read_image(path, size) for path in args.images])
        directory = args.out_dir or out / "batch"
        paths = []
        for start in range(0, n, batch_size):
            part = slice(start, start + batch_size)
            result = pipeline.edit(images[part], prompts[part], steps=args.steps, seed=seed + start)
            for i, image in enumerate(result.constrained, start):
                paths.append(str(save_png(directory / f"{i + 1:03d}_{args.images[i].stem}.png", image)))
        return {"outputs": paths}

    rounds = []
    for item in args.instructions.split(";"):
        pair = item.split(":")
        if len(pair) != 2:
            raise ValueError("chain instructions must use color:shape pairs separated by semicolons")
        rounds.append(_edit_prompt(cfg, *pair)[None])
    source = read_image(args.image, size)
    results = pipeline.chain(source[None], rounds, steps=args.steps, seed=seed)
    directory = args.out_dir or out / "chain"
    return {"outputs": [
        str(save_png(directory / f"round_{i:02d}.png", result.constrained[0]))
        for i, result in enumerate(results, 1)
    ]}


def main() -> None:
    parser = _parser()
    try:
        print(json.dumps(run(parser.parse_args()), indent=2, ensure_ascii=False))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
