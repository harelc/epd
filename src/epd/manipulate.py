"""Multi-scale tone manipulation.

Inspired by Eq. 16 from Farbman et al. 2008:
    ĝ_p = μ + S(δ₀, η·b_p - μ) + Σ S(δᵢ, d^i_p)

The paper uses a sigmoid S(a, x) = 1/(1+exp(-ax)) "appropriately shifted
and normalized" to prevent hard clipping when detail layers are significantly
boosted. We implement this as: boost * x, with tanh soft-clipping scaled to
the lightness range, so that boost=1 is identity and boost>1 amplifies details.
"""

import numpy as np

from epd.decompose import PyramidMode, edge_preserving_decomposition
from epd.smoothing import SmoothingMethod
from epd.utils import lab_to_rgb, rgb_to_lab

L_RANGE = 100.0  # CIELAB lightness range


def _soft_boost(factor: float, x: np.ndarray, clip_range: float = L_RANGE) -> np.ndarray:
    """Boost signal by factor with soft clipping to prevent overflow.

    For factor=1, returns x unchanged. For factor>1, amplifies x
    linearly for small values and soft-clips near ±clip_range via tanh.
    """
    if abs(factor - 1.0) < 1e-6:
        return x.copy()
    scaled = factor * x
    # tanh soft-clip: linear near zero, saturates at ±clip_range
    return clip_range * np.tanh(scaled / clip_range)


def multi_scale_tone_manipulate(
    image_rgb: np.ndarray,
    exposure: float = 1.0,
    base_boost: float = 1.0,
    detail_boosts: list[float] | None = None,
    num_levels: int = 3,
    alpha_fine: float = 1.2,
    alpha_coarse: float = 1.4,
    lam_fine: float = 0.1,
    lam_coarse: float = 0.4,
    method: SmoothingMethod = SmoothingMethod.WLS,
    mode: PyramidMode = PyramidMode.NON_ITERATIVE,
) -> np.ndarray:
    """Apply multi-scale tone manipulation to an RGB image.

    Decomposes the CIELAB lightness channel into base + detail layers,
    applies per-level boosting, and reconstructs.

    Args:
        image_rgb: Input RGB image (H, W, 3) in [0, 1].
        exposure: Exposure adjustment η for base layer.
        base_boost: Boosting factor δ₀ for base layer contrast.
        detail_boosts: List of boosting factors for detail layers.
        num_levels: Number of decomposition levels.
        alpha_fine: α for finest level (WLS only).
        alpha_coarse: α for coarsest level (WLS only).
        lam_fine: Smoothing strength for finest level.
        lam_coarse: Smoothing strength for coarsest level.
        method: Smoothing operator (WLS, bilateral, guided).
        mode: Pyramid construction (non_iterative or iterative).

    Returns:
        Manipulated RGB image (H, W, 3) in [0, 1].
    """
    if detail_boosts is None:
        detail_boosts = [1.0] * num_levels

    if len(detail_boosts) != num_levels:
        raise ValueError(f"Expected {num_levels} detail boosts, got {len(detail_boosts)}")

    # Convert to LAB
    lab = rgb_to_lab(image_rgb)
    lightness = lab[:, :, 0]  # L channel, range [0, 100]

    # Compute log-luminance for smoothness weights
    log_lum = np.log(np.maximum(lightness, 1e-6))

    # Decompose
    base, details = edge_preserving_decomposition(
        lightness,
        num_levels=num_levels,
        alpha_fine=alpha_fine,
        alpha_coarse=alpha_coarse,
        lam_fine=lam_fine,
        lam_coarse=lam_coarse,
        log_luminance=log_lum,
        method=method,
        mode=mode,
    )

    # Manipulate
    mu = (lightness.max() + lightness.min()) / 2.0

    # Base layer: exposure + contrast boost
    result = mu + _soft_boost(base_boost, exposure * base - mu)

    # Detail layers: boosting
    for i, d in enumerate(details):
        result = result + _soft_boost(detail_boosts[i], d)

    # Clamp lightness to valid range
    result = np.clip(result, 0.0, 100.0)

    # Reconstruct LAB and convert back
    lab_out = lab.copy()
    lab_out[:, :, 0] = result

    return lab_to_rgb(lab_out)
