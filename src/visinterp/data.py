"""Download, preprocess and index the playmobil-dc-poses dataset.

The raw images are 1200x1600 transparent PNG cutouts. We composite them onto
white, take a 1200x1200 square crop centred vertically on the figure's bounding
box (same scale for every image), resize, and cache the result as one uint8
tensor. The HF train/test split is random, so we ignore it and keep all 792
images in a fixed order; experiments choose their own split (usually by angle).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

REPO_ID = "nicolashr/playmobil-dc-poses"

# Class order defines the integer labels and the order of the cached images. It follows the
# dataset's original label names (renamed on the Hub in Nov 2026), so it is not alphabetical
# for the new names. Keeping it fixed keeps saved models and decoders valid.
CHARACTERS = [
    "aquaman", "batman", "catwoman", "cyborg", "harleyquinn", "joker",
    "riddler", "robin", "blackadam", "supergirl", "wonderwoman",
]
ARM_POSITIONS = [
    "both_side", "both_up", "both_down", "left_side", "left_high",
    "right_side", "right_high", "left_up", "right_up",
]
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

FACTORS = {
    "character": CHARACTERS,
    "arm_position": ARM_POSITIONS,
    "angle": ANGLES,
}



def _project_root() -> Path:
    """The repo root if we are inside it (so notebooks/ and the root share one cache), else the cwd."""
    here = Path.cwd().resolve()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists() and (d / "src" / "visinterp").exists():
            return d
    return here


PROJECT_ROOT = _project_root()
DEFAULT_ROOT = PROJECT_ROOT / "data_cache"


def download(root: str | Path = DEFAULT_ROOT) -> Path:
    """Download the raw dataset from the Hugging Face Hub (~200 MB). Returns its directory."""
    from huggingface_hub import snapshot_download

    raw_dir = Path(root) / "raw"
    snapshot_download(REPO_ID, repo_type="dataset", local_dir=raw_dir)
    return raw_dir


def _load_image(path: Path, row: pd.Series, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("RGB")
    w, h = img.size
    side = min(w, h)
    cy = row.bbox_y + row.bbox_h / 2
    top = int(round(np.clip(cy - side / 2, 0, h - side)))
    left = (w - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img)


@dataclass
class PlaymobilData:
    """All 792 images plus labels, in a fixed order (character, arm_position, angle).

    images: uint8 tensor [N, 3, H, W]
    labels: DataFrame with string labels and integer `*_idx` columns
    """

    images: torch.Tensor
    labels: pd.DataFrame

    def __len__(self) -> int:
        return len(self.labels)

    def y(self, factor: str) -> torch.Tensor:
        """Integer class labels for `character`, `arm_position` or `angle`."""
        return torch.as_tensor(self.labels[f"{factor}_idx"].to_numpy().copy(), dtype=torch.long)

    def float_images(self, size: int | None = None) -> torch.Tensor:
        """Images as float in [0, 1], optionally resized (antialiased) to `size`."""
        x = self.images.float() / 255.0
        if size is not None and size != x.shape[-1]:
            x = F.interpolate(x, size=(size, size), mode="bilinear", antialias=True, align_corners=False).clamp(0, 1)
        return x

    def split_by_angle(self, test_angle: int) -> tuple[np.ndarray, np.ndarray]:
        """Indices of (train, test), holding out every image at `test_angle`."""
        test = (self.labels["angle"] == test_angle).to_numpy()
        return np.flatnonzero(~test), np.flatnonzero(test)

    def index(self, **query) -> np.ndarray:
        """Indices matching e.g. `character="batman", angle=0`."""
        mask = np.ones(len(self), dtype=bool)
        for k, v in query.items():
            mask &= (self.labels[k] == v).to_numpy()
        return np.flatnonzero(mask)


def load_dataset(root: str | Path = DEFAULT_ROOT, size: int = 224) -> PlaymobilData:
    """Load the preprocessed dataset, downloading and preprocessing on first use."""
    root = Path(root)
    cache = root / f"images_{size}.pt"
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        return PlaymobilData(blob["images"], blob["labels"])

    raw_dir = root / "raw"
    if not (raw_dir / "annotations" / "labels.csv").exists():
        download(root)

    df = pd.read_csv(raw_dir / "annotations" / "labels.csv")
    df["character"] = pd.Categorical(df["character"], CHARACTERS, ordered=True)
    df["arm_position"] = pd.Categorical(df["arm_position"], ARM_POSITIONS, ordered=True)
    df = df.sort_values(["character", "arm_position", "angle"]).reset_index(drop=True)
    assert len(df) == len(CHARACTERS) * len(ARM_POSITIONS) * len(ANGLES), "dataset is not fully factorial"

    images = np.stack(
        [_load_image(raw_dir / row.path, row, size) for row in tqdm(df.itertuples(), total=len(df), desc="preprocess")]
    )
    images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()

    labels = df[["character", "arm_position", "angle", "mirror_arm_position", "path"]].copy()
    labels["character"] = labels["character"].astype(str)
    labels["arm_position"] = labels["arm_position"].astype(str)
    for factor, values in FACTORS.items():
        labels[f"{factor}_idx"] = labels[factor].map({v: i for i, v in enumerate(values)}).astype(int)

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, cache)
    return PlaymobilData(images, labels)
