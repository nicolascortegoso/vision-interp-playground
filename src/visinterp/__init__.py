"""Mechanistic-interpretability experiments on playmobil-dc-poses."""

from .data import ANGLES, ARM_POSITIONS, CHARACTERS, FACTORS, PlaymobilData, load_dataset
from .hooks import cache_activations, patch, record, steer
from .models import INPUT_SIZE, NarrowCNN, TinyCNN
from .train import get_device, load_or_train, load_or_train_narrow, predict, set_seed, train_tiny_cnn

__all__ = [
    "ANGLES", "ARM_POSITIONS", "CHARACTERS", "FACTORS", "PlaymobilData", "load_dataset",
    "cache_activations", "patch", "record", "steer",
    "INPUT_SIZE", "NarrowCNN", "TinyCNN",
    "get_device", "load_or_train", "load_or_train_narrow", "predict", "set_seed", "train_tiny_cnn",
]
