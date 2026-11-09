"""Decoders that turn TinyCNN activations back into images (notebook 08).

A decoder is trained to reconstruct the input image from one frozen layer of the CNN.
Once trained, we can edit activations (add directions, mix images) and decode the
result to *see* what the edit does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import PlaymobilData
from .hooks import record
from .models import INPUT_SIZE
from .train import DEFAULT_ARTIFACTS, augment, get_device, set_seed

# (channels, spatial size) of each TinyCNN block output for 96x96 inputs.
LAYER_SHAPES = {"block2": (64, 24), "block3": (128, 12), "block4": (256, 6)}


def _up(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Decoder(nn.Module):
    """Maps a TinyCNN layer's output back to a [3, 96, 96] image in [0, 1].

    pooled=False: input is the full feature map [C, H, W].
    pooled=True:  input is the spatially averaged vector [C], i.e. what the heads see
                  when layer="block4". A linear layer first spreads it back onto a 6x6 grid.
    """

    def __init__(self, layer: str = "block3", pooled: bool = False, out_size: int = INPUT_SIZE):
        super().__init__()
        self.layer, self.pooled = layer, pooled
        c, size = LAYER_SHAPES[layer]
        if pooled:
            self.stem = nn.Sequential(nn.Linear(c, 256 * 6 * 6), nn.Unflatten(1, (256, 6, 6)), nn.ReLU())
            c, size = 256, 6
        else:
            self.stem = nn.Identity()
        ups = []
        widths = {6: 192, 12: 128, 24: 64, 48: 32}
        while size < out_size:
            ups.append(_up(c, widths[size]))
            c, size = widths[size], size * 2
        self.ups = nn.Sequential(*ups)
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.ups(self.stem(h))))


@torch.no_grad()
def encode(cnn: nn.Module, x: torch.Tensor, layer: str, pooled: bool = False, batch_size: int = 128) -> torch.Tensor:
    """Activations of `layer` for images x (full maps, or spatial means if pooled), on CPU."""
    device = next(cnn.parameters()).device
    cnn.eval()
    outs = []
    for i in range(0, len(x), batch_size):
        with record(cnn, [layer]) as acts:
            cnn(x[i : i + batch_size].to(device))
        h = acts[layer]
        outs.append((h.mean((2, 3)) if pooled else h).cpu())
    return torch.cat(outs)


@torch.no_grad()
def decode(decoder: Decoder, h: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    """Images [N, 3, 96, 96] for activations h, on CPU."""
    device = next(decoder.parameters()).device
    decoder.eval()
    return torch.cat([decoder(h[i : i + batch_size].to(device)).cpu() for i in range(0, len(h), batch_size)])


def train_decoder(
    cnn: nn.Module,
    data: PlaymobilData,
    train_idx: np.ndarray,
    layer: str = "block3",
    pooled: bool = False,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 2e-3,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[Decoder, list[dict]]:
    """Train a decoder on images `train_idx`. The CNN stays frozen.

    Each step augments the images (shifts, brightness), runs them through the CNN, and
    asks the decoder to reconstruct the *augmented* image from the activations (L1 loss).
    Augmentation gives the decoder more varied examples than the 693 originals.
    """
    set_seed(seed)
    device = get_device()
    cnn.eval().to(device)
    for p in cnn.parameters():
        p.requires_grad_(False)
    x = data.float_images(INPUT_SIZE)
    dec = Decoder(layer, pooled).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(train_idx) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    history = []
    for epoch in range(epochs):
        dec.train()
        perm = np.random.permutation(train_idx)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            xb = augment(x[perm[i : i + batch_size]]).to(device)
            with torch.no_grad(), record(cnn, [layer]) as acts:
                cnn(xb)
            h = acts[layer].mean((2, 3)) if pooled else acts[layer]
            loss = F.l1_loss(dec(h), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 25 == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "train_l1": total / len(train_idx)})
            if verbose:
                print(f"epoch {epoch + 1}: train L1 {total / len(train_idx):.4f}", flush=True)
    for p in cnn.parameters():
        p.requires_grad_(True)
    dec.eval()
    return dec, history


def load_or_train_decoder(
    cnn: nn.Module,
    data: PlaymobilData,
    layer: str = "block3",
    pooled: bool = False,
    test_angle: int = 45,
    artifacts: str | Path = DEFAULT_ARTIFACTS,
    **kwargs,
) -> Decoder:
    """Load a saved decoder for the baseline CNN, or train one (a few minutes on a GPU) and save it."""
    name = f"decoder_{layer}{'_pooled' if pooled else ''}_angle{test_angle}.pt"
    path = Path(artifacts) / name
    device = get_device()
    dec = Decoder(layer, pooled).to(device)
    if path.exists():
        dec.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return dec.eval()
    train_idx, _ = data.split_by_angle(test_angle)
    dec, _ = train_decoder(cnn, data, train_idx, layer=layer, pooled=pooled, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dec.state_dict(), path)
    return dec
