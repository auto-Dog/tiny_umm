"""Assemble the training stages around one Flow conditioning interface."""
from __future__ import annotations

import torch
from torch import nn

from data.schema import OP_EDIT, understand_prompt

from .bottleneck import GaussianBottleneck
from .conditioning import InstructionEncoder, SourceEncoder
from .connector import DiscreteBridge, QueryConnector
from .flow import FlowCondition, PixelFlow
from .understanding import UnderstandingMLP


class TinyUMM(nn.Module):
    def __init__(self, cfg: dict, bridge: str | None = None, *, bottleneck: bool = False):
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

    def classify(self, source: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.understanding(source, self.understand_tokens[None].expand(len(source), -1))

    def condition(
        self, prompt: torch.Tensor, source: torch.Tensor | None = None, *,
        direct: bool | torch.Tensor = False, instruction_gain: float = 1.0,
    ) -> FlowCondition:
        instruction = self.instruction(prompt)
        null = self.flow.null_visual[None].expand(len(prompt), -1, -1)
        if source is None:
            return FlowCondition(instruction_gain * instruction, null)
        edit = prompt[:, 0] == OP_EDIT
        source = torch.where(edit[:, None, None, None], source, 0.0)
        if self.bridge is None:
            visual = self.source_encoder(source)
        else:
            # Only U is detached. Both the bridge and the frozen Flow's
            # operations must be in the graph during connector training.
            with torch.no_grad():
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
