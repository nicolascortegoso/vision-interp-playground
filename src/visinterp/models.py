"""The tiny CNN trained from scratch."""

from __future__ import annotations

import torch
from torch import nn

from .data import ARM_POSITIONS, CHARACTERS

# ImageNet statistics are as good as any for a white-background dataset;
# the point is just to centre the inputs.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

INPUT_SIZE = 96  # images are downsampled from 224 to this for the CNN


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),  # not in-place, so hooks on the block see an unaliased tensor
        nn.MaxPool2d(2),
    )


class TinyCNN(nn.Module):
    """Four conv blocks -> global average pool -> two linear heads.

    Takes float images in [0, 1] of shape [B, 3, H, W]; normalisation happens inside.
    Hookable layers (see `LAYERS`): block1..block4 output [B, C, H, W].
    `embed` (not in LAYERS) turns block4 into the vector the heads read.

    Variants (used in notebook 09):
    - head="gap" (default): `embed` is the spatial mean of block4, i.e. exactly what
      `cache_activations` returns for block4.
    - head="flatten": `embed` flattens the whole block4 map, so the heads see spatial layout.
    - angle_head=True: an extra linear head regresses the viewing angle as (cos, sin).
    """

    LAYERS = ["block1", "block2", "block3", "block4"]

    def __init__(
        self,
        widths=(32, 64, 128, 256),
        n_characters=len(CHARACTERS),
        n_poses=len(ARM_POSITIONS),
        dropout=0.2,
        head: str = "gap",
        angle_head: bool = False,
        input_size: int = INPUT_SIZE,
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        chans = (3, *widths)
        self.block1 = conv_block(chans[0], chans[1])
        self.block2 = conv_block(chans[1], chans[2])
        self.block3 = conv_block(chans[2], chans[3])
        self.block4 = conv_block(chans[3], chans[4])
        if head == "gap":
            self.embed = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
            feat = widths[-1]
        elif head == "flatten":
            self.embed = nn.Flatten()
            feat = widths[-1] * (input_size // 16) ** 2
        else:
            raise ValueError(f"unknown head {head!r}")
        self.dropout = nn.Dropout(dropout)
        self.character_head = nn.Linear(feat, n_characters)
        self.arm_position_head = nn.Linear(feat, n_poses)
        self.angle_head = nn.Linear(feat, 2) if angle_head else None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        x = self.block4(self.block3(self.block2(self.block1(x))))
        return self.embed(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._heads(self.features(x))

    def forward_from(self, layer: str, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the rest of the network starting from the output `h` of `layer`.

        Useful after editing activations: `model.forward_from("block3", edited)` gives the
        heads' outputs as if `edited` had been block3's output.
        """
        names = self.LAYERS
        for name in names[names.index(layer) + 1 :]:
            h = getattr(self, name)(h)
        return self._heads(self.embed(h))

    def _heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.dropout(h)
        out = {"character": self.character_head(h), "arm_position": self.arm_position_head(h)}
        if self.angle_head is not None:
            out["angle_cos_sin"] = self.angle_head(h)
        return out


class NarrowCNN(nn.Module):
    """A deliberately small CNN for the basics notebooks, built to be read by hand.

    Four stages, each conv -> ReLU -> 2x2 max-pool, then global average pooling and two
    linear heads. No BatchNorm, and one convolution per stage, so every channel of a stage
    is computed directly from the previous stage's channels.

    Shapes for a batch of B images at 96x96 (the default):

        input   [B,  3, 96, 96]
        conv1   [B,  8, 96, 96]   5x5 kernels, padding 2   relu1 same   pool1 [B,  8, 48, 48]
        conv2   [B, 16, 48, 48]   3x3 kernels, padding 1   relu2 same   pool2 [B, 16, 24, 24]
        conv3   [B, 32, 24, 24]   3x3 kernels, padding 1   relu3 same   pool3 [B, 32, 12, 12]
        conv4   [B, 64, 12, 12]   3x3 kernels, padding 1   relu4 same   pool4 [B, 64,  6,  6]
        gap     [B, 64]           mean over the 6x6 positions
        heads   character [B, 11], arm_position [B, 9]

    Every stage is its own module ("conv2", "relu2", "pool2", ...), so hooks can read any of them.
    """

    STAGES = ["conv1", "relu1", "pool1", "conv2", "relu2", "pool2",
              "conv3", "relu3", "pool3", "conv4", "relu4", "pool4", "gap"]

    def __init__(self, widths=(8, 16, 32, 64), n_characters=len(CHARACTERS), n_poses=len(ARM_POSITIONS)):
        super().__init__()
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))
        chans = (3, *widths)
        for i in range(4):
            k = 5 if i == 0 else 3
            setattr(self, f"conv{i + 1}", nn.Conv2d(chans[i], chans[i + 1], k, padding=k // 2))
            setattr(self, f"relu{i + 1}", nn.ReLU())
            setattr(self, f"pool{i + 1}", nn.MaxPool2d(2))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.character_head = nn.Linear(widths[-1], n_characters)
        self.arm_position_head = nn.Linear(widths[-1], n_poses)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = (x - self.mean) / self.std
        for name in self.STAGES:
            h = getattr(self, name)(h)
        return {"character": self.character_head(h), "arm_position": self.arm_position_head(h)}
