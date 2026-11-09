"""Plain PyTorch forward hooks: record, cache, patch and steer.

Layers are addressed by their dotted module name (`model.get_submodule(name)`),
e.g. "block3" for the tiny CNN or "model.encoder.layer.5" for DINOv2.
A layer output may be a tensor or a tuple whose first element is the hidden
state (as in Hugging Face transformer blocks); both are handled.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable

import torch
from torch import nn


def _main(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_main(out, new):
    return (new, *out[1:]) if isinstance(out, tuple) else new


def default_reduce(h: torch.Tensor) -> torch.Tensor:
    """Pool a layer output to [B, D]: spatial mean for conv maps, CLS token for token sequences."""
    if h.ndim == 4:  # [B, C, H, W]
        return h.mean(dim=(2, 3))
    if h.ndim == 3:  # [B, T, D]
        return h[:, 0]
    return h


@contextmanager
def record(model: nn.Module, layers: Iterable[str]):
    """Capture the (unreduced) outputs of `layers` during forward passes.

    with record(model, ["block2"]) as acts:
        model(x)
    acts["block2"]  # [B, C, H, W]
    """
    acts: dict[str, torch.Tensor] = {}
    handles = []
    for name in layers:
        def hook(_mod, _inp, out, name=name):
            acts[name] = _main(out).detach()
        handles.append(model.get_submodule(name).register_forward_hook(hook))
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def cache_activations(
    model: nn.Module,
    images: torch.Tensor,
    layers: Iterable[str],
    reduce: Callable[[torch.Tensor], torch.Tensor] = default_reduce,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Run all `images` through `model` and return {layer: [N, D]} reduced activations (on CPU)."""
    layers = list(layers)
    model.eval().to(device)
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in layers}
    for i in range(0, len(images), batch_size):
        with record(model, layers) as acts:
            model(images[i : i + batch_size].to(device))
        for name in layers:
            chunks[name].append(reduce(acts[name]).float().cpu())
    return {name: torch.cat(c) for name, c in chunks.items()}


@contextmanager
def patch(model: nn.Module, layer: str, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Replace the output `h` of `layer` with `fn(h)` while the context is active."""
    def hook(_mod, _inp, out):
        return _replace_main(out, fn(_main(out)))

    handle = model.get_submodule(layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _broadcast(v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    v = v.to(h.device, h.dtype)
    if h.ndim == 4:  # conv map: add at every spatial position
        return v.view(1, -1, 1, 1)
    return v.view(*([1] * (h.ndim - 1)), -1)  # [1, D] or [1, 1, D]: every token


@contextmanager
def steer(model: nn.Module, layer: str, vector: torch.Tensor, alpha: float = 1.0, token: int | None = None):
    """Add `alpha * vector` to the output of `layer`.

    Conv maps get it at every spatial position. Token sequences [B, T, D] get it at every
    token, or only at position `token` if given (e.g. `token=0` for a ViT's CLS token).
    """
    def fn(h):
        v = alpha * _broadcast(vector, h)
        if token is None or h.ndim != 3:
            return h + v
        h = h.clone()
        h[:, token] += v[0, 0]
        return h

    with patch(model, layer, fn):
        yield
