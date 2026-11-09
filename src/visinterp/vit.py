"""Pretrained ViTs (DINOv2-small, CLIP ViT-B/32) behind the same interface as the tiny CNN.

`PretrainedViT` takes float images in [0, 1] at 224x224, normalises them for the
backbone, and returns the final CLS embedding. `layer_names` lists the transformer
blocks in order, so `cache_activations(vit, x, vit.layer_names)` gives the CLS token
of the residual stream after every block.
"""

from __future__ import annotations

import torch
from torch import nn

BACKBONES = {
    "dinov2": dict(
        repo="facebook/dinov2-small",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    "clip": dict(
        repo="openai/clip-vit-base-patch32",
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
}


class PretrainedViT(nn.Module):
    def __init__(self, name: str = "dinov2"):
        super().__init__()
        cfg = BACKBONES[name]
        if name == "clip":
            from transformers import CLIPVisionModel

            self.model = CLIPVisionModel.from_pretrained(cfg["repo"])
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(cfg["repo"])
        self.name = name
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1))
        # The transformer blocks are the largest ModuleList in the backbone.
        lists = [(n, m) for n, m in self.named_modules() if isinstance(m, nn.ModuleList)]
        prefix, blocks = max(lists, key=lambda nm: len(nm[1]))
        self.layer_names = [f"{prefix}.{i}" for i in range(len(blocks))]
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] in [0, 1] -> final CLS embedding [B, D]."""
        out = self.model(pixel_values=(x - self.mean) / self.std)
        return out.last_hidden_state[:, 0]
