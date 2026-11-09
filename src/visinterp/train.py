"""Training and evaluation for the tiny CNN, plus a load-or-train helper for notebooks."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import PROJECT_ROOT, PlaymobilData
from .models import INPUT_SIZE, NarrowCNN, TinyCNN

TARGETS = ("character", "arm_position")
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"

# Model variants compared in notebook 09. Each maps to (TinyCNN kwargs, training kwargs).
VARIANTS = {
    "baseline": ({}, {}),
    "flatten": ({"head": "flatten"}, {}),
    "angle_head": ({"angle_head": True}, {}),
    "mirror_aug": ({}, {"mirror_aug": True}),
    # Controls for notebook 09: also leave out 315°, the mirror image of the 45° test view.
    "baseline_minus315": ({}, {"exclude_angles": (315,)}),
    "mirror_aug_minus315": ({}, {"mirror_aug": True, "exclude_angles": (315,)}),
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(x: torch.Tensor, max_shift: int = 6, brightness: float = 0.15) -> torch.Tensor:
    """Small random translations (white padding) and brightness jitter. No flips:
    mirroring an image swaps the figure's left and right arms, which changes the label."""
    b, _, h, w = x.shape
    padded = F.pad(x, [max_shift] * 4, value=1.0)
    dx = torch.randint(0, 2 * max_shift + 1, (b,)).tolist()
    dy = torch.randint(0, 2 * max_shift + 1, (b,)).tolist()
    x = torch.stack([padded[i, :, dy[i] : dy[i] + h, dx[i] : dx[i] + w] for i in range(b)])
    scale = 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * brightness
    return (x * scale).clamp(0, 1)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 128, device=None) -> dict[str, torch.Tensor]:
    """Logits for every head, on CPU."""
    device = device or next(model.parameters()).device
    model.eval()
    outs: dict[str, list[torch.Tensor]] = {}
    for i in range(0, len(x), batch_size):
        for k, v in model(x[i : i + batch_size].to(device)).items():
            outs.setdefault(k, []).append(v.float().cpu())
    return {k: torch.cat(v) for k, v in outs.items()}


def accuracy(logits: dict[str, torch.Tensor], data: PlaymobilData, idx) -> dict[str, float]:
    """Accuracy of the classification heads (`character`, `arm_position`) on images `idx`."""
    return {t: (logits[t][idx].argmax(1) == data.y(t)[idx]).float().mean().item() for t in TARGETS if t in logits}


def mirror_pose_map(data: PlaymobilData) -> torch.Tensor:
    """[n_poses] tensor: index of each arm position's mirror image (left <-> right, ...)."""
    pairs = data.labels[["arm_position_idx", "mirror_arm_position"]].drop_duplicates()
    from .data import ARM_POSITIONS

    m = torch.arange(len(ARM_POSITIONS))
    for row in pairs.itertuples():
        m[row.arm_position_idx] = ARM_POSITIONS.index(row.mirror_arm_position)
    return m


@dataclass
class TrainResult:
    model: TinyCNN
    train_idx: np.ndarray
    test_idx: np.ndarray
    test_angle: int
    history: list[dict] = field(default_factory=list)


def train_tiny_cnn(
    data: PlaymobilData,
    test_angle: int = 45,
    size: int = INPUT_SIZE,
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 3e-3,
    weight_decay: float = 5e-4,
    seed: int = 0,
    device=None,
    verbose: bool = True,
    mirror_aug: bool = False,
    angle_weight: float = 1.0,
    exclude_angles: tuple = (),
    model: nn.Module | None = None,
    **model_kwargs,
) -> TrainResult:
    """Train on all angles except `test_angle`; evaluate on the held-out angle every few epochs.

    mirror_aug: flip half of each batch horizontally *and* swap the arm-position label for its
        mirror image (left <-> right, ...). If the model has an angle head, the angle becomes 360 - angle.
    exclude_angles: further angles left out of training (but not added to the test set).
    model: an untrained model to train instead of a new TinyCNN (e.g. a NarrowCNN).
    model_kwargs: passed to TinyCNN (e.g. head="flatten", angle_head=True).
    """
    set_seed(seed)
    device = device or get_device()
    x = data.float_images(size)
    ys = {t: data.y(t) for t in TARGETS}
    theta = torch.tensor(data.labels["angle"].to_numpy(), dtype=torch.float32)  # degrees
    mirror = mirror_pose_map(data)
    train_idx, test_idx = data.split_by_angle(test_angle)
    train_idx = train_idx[~data.labels["angle"].iloc[train_idx].isin(exclude_angles).to_numpy()]

    if model is None:
        model = TinyCNN(input_size=size, **model_kwargs)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)

    result = TrainResult(model, train_idx, test_idx, test_angle)
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            xb = augment(x[b])
            yb = {t: ys[t][b] for t in TARGETS}
            tb = theta[b]
            if mirror_aug:
                flip = torch.rand(len(b)) < 0.5
                xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
                yb["arm_position"] = torch.where(flip, mirror[yb["arm_position"]], yb["arm_position"])
                tb = torch.where(flip, (360 - tb) % 360, tb)
            out = model(xb.to(device))
            loss = sum(F.cross_entropy(out[t], yb[t].to(device)) for t in TARGETS)
            if "angle_cos_sin" in out:
                rad = torch.deg2rad(tb)
                target = torch.stack([rad.cos(), rad.sin()], 1).to(device)
                loss = loss + angle_weight * F.mse_loss(out["angle_cos_sin"], target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logits = predict(model, x, device=device)
            row = {"epoch": epoch + 1, "loss": total / len(train_idx)}
            row.update({f"train_{k}": v for k, v in accuracy(logits, data, train_idx).items()})
            row.update({f"test_{k}": v for k, v in accuracy(logits, data, test_idx).items()})
            result.history.append(row)
            if verbose:
                print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    model.eval()
    return result


def load_or_train(
    data: PlaymobilData,
    test_angle: int = 45,
    seed: int = 0,
    variant: str = "baseline",
    artifacts: str | Path = DEFAULT_ARTIFACTS,
    **kwargs,
) -> TrainResult:
    """Load a saved checkpoint, or train (about a minute on a GPU) and save it.

    variant: one of VARIANTS ("baseline", "flatten", "angle_head", "mirror_aug"); see notebook 09.
    """
    model_kwargs, train_kwargs = VARIANTS[variant]
    tag = "" if variant == "baseline" else f"{variant}_"
    path = Path(artifacts) / f"tiny_cnn_{tag}angle{test_angle}_seed{seed}.pt"
    device = get_device()
    if path.exists():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = TinyCNN(**model_kwargs).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        train_idx, test_idx = data.split_by_angle(test_angle)
        return TrainResult(model, train_idx, test_idx, test_angle, ckpt["history"])
    result = train_tiny_cnn(data, test_angle=test_angle, seed=seed, device=device, **train_kwargs, **model_kwargs, **kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result.model.state_dict(), "history": result.history}, path)
    return result


BUNDLED_WEIGHTS = Path(__file__).parent / "weights"


def load_or_train_narrow(
    data: PlaymobilData,
    test_angle: int = 45,
    seed: int = 0,
    artifacts: str | Path = DEFAULT_ARTIFACTS,
    retrain: bool = False,
    **kwargs,
) -> TrainResult:
    """Load the NarrowCNN used in the basics notebooks, or train it (under a minute on a laptop).

    The weights for test_angle=45, seed=0 ship with the package, so every reader studies the same
    network and the channel numbers quoted in the notebooks match. Training on other hardware gives a
    different network (channels in a different order, with somewhat different roles). retrain=True
    trains a fresh one anyway and saves it under `artifacts`.
    """
    name = f"narrow_cnn_angle{test_angle}_seed{seed}.pt"
    path = Path(artifacts) / name
    if not retrain and not path.exists() and (BUNDLED_WEIGHTS / name).exists():
        path = BUNDLED_WEIGHTS / name
    device = get_device()
    if path.exists() and not retrain:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = NarrowCNN().to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        train_idx, test_idx = data.split_by_angle(test_angle)
        return TrainResult(model, train_idx, test_idx, test_angle, ckpt["history"])
    set_seed(seed)
    result = train_tiny_cnn(data, test_angle=test_angle, seed=seed, device=device, model=NarrowCNN(), **kwargs)
    path = Path(artifacts) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": result.model.state_dict(), "history": result.history}, path)
    return result
