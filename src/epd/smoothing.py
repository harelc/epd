"""Edge-preserving smoothing operators.

Provides a unified interface for multiple smoothing methods:
- WLS (Farbman et al. 2008)
- Bilateral filter
- Guided filter (He et al. 2010)
"""

from enum import Enum
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from scipy.ndimage import uniform_filter
from scipy.sparse.linalg import spsolve


class SmoothingMethod(str, Enum):
    WLS = "wls"
    BILATERAL = "bilateral"
    GUIDED = "guided"


class EdgePreservingSmoother(Protocol):
    def __call__(self, image: np.ndarray, strength: float, guide: np.ndarray | None = None) -> np.ndarray:
        """Smooth a 2D image with given strength. Optional guide image for edge awareness."""
        ...


# ---------------------------------------------------------------------------
# WLS
# ---------------------------------------------------------------------------


def _compute_smoothness_weights(
    log_luminance: np.ndarray, alpha: float, eps: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatially-varying smoothness weights (Eq. 6)."""
    dl_dx = np.diff(log_luminance, axis=1)
    dl_dx = np.pad(dl_dx, ((0, 0), (0, 1)), mode="constant")

    dl_dy = np.diff(log_luminance, axis=0)
    dl_dy = np.pad(dl_dy, ((0, 1), (0, 0)), mode="constant")

    ax = 1.0 / (np.abs(dl_dx) ** alpha + eps)
    ay = 1.0 / (np.abs(dl_dy) ** alpha + eps)

    return ax, ay


def _build_wls_system(ax: np.ndarray, ay: np.ndarray, lam: float, h: int, w: int) -> sp.csc_matrix:
    """Build the sparse matrix (I + λ·L_g) for the WLS system."""
    n = h * w
    ax_flat = ax.ravel()
    ay_flat = ay.ravel()

    idx = np.arange(n).reshape(h, w)
    horiz_left = idx[:, :-1].ravel()
    horiz_right = idx[:, 1:].ravel()
    horiz_w = ax_flat[horiz_left]

    vert_top = idx[:-1, :].ravel()
    vert_bot = idx[1:, :].ravel()
    vert_w = ay_flat[vert_top]

    diag = np.ones(n, dtype=np.float64)
    np.add.at(diag, horiz_left, lam * horiz_w)
    np.add.at(diag, horiz_right, lam * horiz_w)
    np.add.at(diag, vert_top, lam * vert_w)
    np.add.at(diag, vert_bot, lam * vert_w)

    off_rows = np.concatenate([horiz_left, horiz_right, vert_top, vert_bot])
    off_cols = np.concatenate([horiz_right, horiz_left, vert_bot, vert_top])
    off_vals = np.concatenate([-lam * horiz_w, -lam * horiz_w, -lam * vert_w, -lam * vert_w])

    rows = np.concatenate([np.arange(n), off_rows])
    cols = np.concatenate([np.arange(n), off_cols])
    vals = np.concatenate([diag, off_vals])

    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def wls_smooth(image: np.ndarray, lam: float, alpha: float, log_luminance: np.ndarray | None = None) -> np.ndarray:
    """Apply WLS edge-preserving smoothing to a 2D image."""
    h, w = image.shape
    if log_luminance is None:
        log_luminance = np.log(np.maximum(image, 1e-6))
    ax, ay = _compute_smoothness_weights(log_luminance, alpha)
    A = _build_wls_system(ax, ay, lam, h, w)
    u = spsolve(A, image.ravel())
    return u.reshape(h, w)


# ---------------------------------------------------------------------------
# Bilateral filter
# ---------------------------------------------------------------------------


def bilateral_smooth(image: np.ndarray, sigma_s: float, sigma_r: float, guide: np.ndarray | None = None) -> np.ndarray:
    """Apply bilateral filter to a 2D image.

    Pure numpy implementation using a truncated spatial kernel.
    """
    if guide is None:
        guide = image

    h, w = image.shape
    radius = min(int(np.ceil(2.0 * sigma_s)), min(h, w) - 1)
    result = np.zeros_like(image)
    weight_sum = np.zeros_like(image)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            spatial_w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_s * sigma_s))

            # Compute overlapping regions between shifted and unshifted images
            y1s, y1e = max(0, dy), min(h, h + dy)
            y2s, y2e = max(0, -dy), min(h, h - dy)
            x1s, x1e = max(0, dx), min(w, w + dx)
            x2s, x2e = max(0, -dx), min(w, w - dx)

            if y1e <= y1s or x1e <= x1s:
                continue

            diff = guide[y1s:y1e, x1s:x1e] - guide[y2s:y2e, x2s:x2e]
            range_w = np.exp(-(diff * diff) / (2.0 * sigma_r * sigma_r))

            w_total = spatial_w * range_w
            result[y2s:y2e, x2s:x2e] += w_total * image[y1s:y1e, x1s:x1e]
            weight_sum[y2s:y2e, x2s:x2e] += w_total

    return result / np.maximum(weight_sum, 1e-10)


# ---------------------------------------------------------------------------
# Guided filter (He et al. 2010)
# ---------------------------------------------------------------------------


def guided_smooth(image: np.ndarray, radius: int, eps: float, guide: np.ndarray | None = None) -> np.ndarray:
    """Apply guided filter to a 2D image.

    O(N) implementation using box (mean) filters.
    """
    if guide is None:
        guide = image

    size = 2 * radius + 1

    mean_g = uniform_filter(guide, size=size, mode="reflect")
    mean_p = uniform_filter(image, size=size, mode="reflect")
    mean_gp = uniform_filter(guide * image, size=size, mode="reflect")
    mean_gg = uniform_filter(guide * guide, size=size, mode="reflect")

    cov_gp = mean_gp - mean_g * mean_p
    var_g = mean_gg - mean_g * mean_g

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g

    mean_a = uniform_filter(a, size=size, mode="reflect")
    mean_b = uniform_filter(b, size=size, mode="reflect")

    return mean_a * guide + mean_b


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def _auto_scale_factor(image: np.ndarray) -> float:
    """Compute a scale factor based on image size relative to a 1MP reference."""
    h, w = image.shape
    megapixels = (h * w) / 1e6
    return max(0.25, np.sqrt(megapixels))


def smooth(
    image: np.ndarray,
    method: SmoothingMethod,
    strength: float,
    alpha: float = 1.2,
    log_luminance: np.ndarray | None = None,
    sigma_r: float | None = None,
    guided_eps: float | None = None,
) -> np.ndarray:
    """Unified smoothing interface with auto-adaptation to image size.

    Args:
        image: 2D array to smooth.
        method: Which smoothing operator to use.
        strength: Controls the spatial scale of smoothing.
            - WLS: used directly as λ
            - Bilateral: mapped to σ_s, scaled by image size
            - Guided: mapped to radius, scaled by image size
        alpha: Gradient sensitivity (WLS only).
        log_luminance: Guide image for edge awareness.
        sigma_r: Range parameter for bilateral filter.
            If None, auto-computed from image value range.
        guided_eps: Regularization for guided filter.
            If None, auto-computed from image variance.
    """
    scale = _auto_scale_factor(image)

    if method == SmoothingMethod.WLS:
        return wls_smooth(image, lam=strength, alpha=alpha, log_luminance=log_luminance)

    elif method == SmoothingMethod.BILATERAL:
        sigma_s = max(1.0, strength * 30.0 * scale)
        if sigma_r is None:
            value_range = float(np.ptp(image)) if np.ptp(image) > 0 else 1.0
            sigma_r = 0.1 * value_range * (0.5 + strength)
        return bilateral_smooth(image, sigma_s=sigma_s, sigma_r=sigma_r, guide=log_luminance)

    elif method == SmoothingMethod.GUIDED:
        radius = max(1, int(strength * 40 * scale))
        if guided_eps is None:
            variance = float(np.var(image)) if np.var(image) > 0 else 1.0
            guided_eps = variance * 0.01 * (0.5 + strength * 5.0)
        return guided_smooth(image, radius=radius, eps=guided_eps, guide=log_luminance)

    else:
        raise ValueError(f"Unknown smoothing method: {method}")
