"""Assemble the training stages around one Flow conditioning interface."""
from __future__ import annotations

import torch
from torch import nn

from data.schema import OP_EDIT, OP_GENERATE, understand_prompt

from .bottleneck import GaussianBottleneck
from .conditioning import InstructionEncoder, SourceEncoder
from .connector import DiscreteBridge, QueryConnector
from .flow import FlowCondition, PixelFlow
from .understanding import SharedUnderstandingHead, UnderstandingMLP


class TinyUMM(nn.Module):
    def __init__(
        self, cfg: dict, bridge: str | None = None, *, bottleneck: bool = False, shared_understanding: bool = False,
    ):
        super().__init__()
        self.flow = PixelFlow(cfg)
        self.instruction = InstructionEncoder(cfg)
        self.source_encoder = SourceEncoder(cfg)
        self.understanding = UnderstandingMLP(cfg)
        if bridge is None:
            self.bridge = None
        elif bridge == "connector":
            self.bridge = QueryConnector(cfg)
        elif bridge == "discrete":
            self.bridge = DiscreteBridge(cfg)
        else:
            raise ValueError(f"unknown bridge: {bridge}")
        self.bottleneck = GaussianBottleneck(cfg) if bottleneck else None
        self.register_buffer("understand_tokens", torch.from_numpy(understand_prompt(cfg)).long(), persistent=False)
        self.text_aux = SharedUnderstandingHead(cfg) if shared_understanding else None
        if shared_understanding:
            if bridge != "connector":
                raise ValueError("shared understanding requires the learnable Query Connector")
            size = cfg["image_size"]
            probe = torch.randn((1, size, size, 3), generator=torch.Generator().manual_seed(37))
            self.register_buffer("understand_probe", probe, persistent=False)

    def classify(
        self, source: torch.Tensor, *, generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        prompt = self.understand_tokens[None].expand(len(source), -1)
        if self.text_aux is None:
            return self.understanding(source, prompt)
        condition = self.condition(prompt, source)
        if self.bottleneck is not None:
            if generator is None:
                generator = torch.Generator(device=source.device).manual_seed(37)
            condition, _, _ = self.bottleneck(condition, generator, prompt=prompt)
        return self.read_text(condition)

    def read_text(self, condition: FlowCondition) -> dict[str, torch.Tensor]:
        return self.text_aux(self.query_features(condition))

    def query_features(self, condition: FlowCondition) -> torch.Tensor:
        """Read the shared decoder using only its sampled query and instruction inputs."""
        if self.text_aux is None:
            raise ValueError("model has no shared understanding head")
        probe = self.understand_probe.expand(len(condition.instruction), -1, -1, -1)
        return self.flow.probe_features(probe, condition)

    @torch.no_grad()
    def keeps_source_color(self, prompt: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=source.device.type, enabled=False):
            return self.classify(source)["color"].argmax(dim=1) == prompt[:, 2]

    def condition(
        self, prompt: torch.Tensor, source: torch.Tensor | None = None, *,
        direct: bool | torch.Tensor = False, instruction_gain: float = 1.0, train_features: bool = False,
    ) -> FlowCondition:
        instruction = self.instruction(prompt)
        null = self.flow.null_visual[None].expand(len(prompt), -1, -1)
        if self.text_aux is not None:
            if direct is not False:
                raise ValueError("shared queries do not accept direct source pixels")
            generation = prompt[:, 0] == OP_GENERATE
            if source is None:
                features = instruction.new_zeros((len(prompt), self.understanding.feature_dim))
            else:
                features = self.understanding.encode_image(source)
                features = torch.where(generation[:, None], 0.0, features)
            queries = self.bridge(features, instruction)
            # Start the new text-only query path near the pretrained null condition.
            visual = torch.where(generation[:, None, None], null + 0.05 * queries, queries)
            return FlowCondition(instruction_gain * instruction, visual)
        if source is None:
            return FlowCondition(instruction_gain * instruction, null)
        edit = prompt[:, 0] == OP_EDIT
        source = torch.where(edit[:, None, None, None], source, 0.0)
        if self.bridge is None:
            visual = self.source_encoder(source)
        else:
            # Stage C freezes U by default; the bridge and Flow still propagate gradients.
            with torch.set_grad_enabled(train_features and torch.is_grad_enabled()):
                features = self.classify(source) if isinstance(self.bridge, DiscreteBridge) else self.understanding.encode_image(source)
            visual = self.bridge(features, instruction)
        visual = torch.where(edit[:, None, None], visual, null.to(visual.dtype))
        if direct is False:
            return FlowCondition(instruction_gain * instruction, visual)
        return FlowCondition(instruction_gain * instruction, visual, source, edit & direct)

    def train_bridge(self, *, joint_instruction: bool = False, adapters: bool = False) -> None:
        if self.bridge is None:
            raise ValueError("model has no bridge to train")
        self.eval().requires_grad_(False)
        self.bridge.train().requires_grad_(True)
        if joint_instruction:
            self.instruction.train().requires_grad_(True)
        if adapters:
            for adapter in self.flow.adapters():
                adapter.train().requires_grad_(True)
