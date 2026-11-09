"""Small plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_images(x: torch.Tensor, titles=None, ncols: int = 8, size: float = 1.6, suptitle: str | None = None):
    """Grid of float [N, 3, H, W] images in [0, 1]."""
    n = len(x)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.1), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(x[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            if titles is not None:
                ax.set_title(titles[i], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def heatmap(M, row_labels, col_labels, ax=None, cmap="RdBu_r", vmin=-1, vmax=1, fmt=".2f", annotate=True, title=None):
    """Annotated matrix plot."""
    M = M.numpy() if isinstance(M, torch.Tensor) else np.asarray(M)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.55 * len(col_labels) + 2, 0.45 * len(row_labels) + 1.5))
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, format(M[i, j], fmt), ha="center", va="center", fontsize=6)
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax
