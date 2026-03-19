from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as a float64 RGB array in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def save_image(img: np.ndarray, path: str | Path) -> None:
    """Save a float64 RGB array in [0, 1] to disk."""
    img_clipped = np.clip(img, 0.0, 1.0)
    Image.fromarray((img_clipped * 255).astype(np.uint8)).save(path)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] to CIELAB. Returns (H, W, 3) with L in [0, 100]."""
    return color.rgb2lab(rgb)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to RGB [0,1]."""
    return np.clip(color.lab2rgb(lab), 0.0, 1.0)
