"""Multi-scale edge-preserving decomposition.

Implements Section 3.1 from Farbman et al. 2008.

Two construction methods:
- Non-iterative (Eq. 13): u^{i+1} = F_{c^i·λ}(g)      — each level from original
- Iterative (Eq. 14):     u^{i+1} = F_{c^i·λ}(u^i)     — each level from previous

Detail layers (Eq. 11): d^i = u^{i-1} - u^i
"""

from enum import Enum

import numpy as np

from epd.smoothing import SmoothingMethod, smooth


class PyramidMode(str, Enum):
    NON_ITERATIVE = "non_iterative"
    ITERATIVE = "iterative"


def edge_preserving_decomposition(
    image: np.ndarray,
    num_levels: int = 3,
    alpha_fine: float = 1.2,
    alpha_coarse: float = 1.4,
    lam_fine: float = 0.1,
    lam_coarse: float = 0.4,
    log_luminance: np.ndarray | None = None,
    method: SmoothingMethod = SmoothingMethod.WLS,
    mode: PyramidMode = PyramidMode.NON_ITERATIVE,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute a multi-scale edge-preserving decomposition.

    Args:
        image: 2D array (H, W), e.g. CIELAB lightness.
        num_levels: Number of detail levels (default 3).
        alpha_fine: α for finest level (WLS only).
        alpha_coarse: α for coarsest level (WLS only).
        lam_fine: Smoothing strength for finest level.
        lam_coarse: Smoothing strength for coarsest level.
        log_luminance: Optional log-luminance for smoothness weights.
        method: Smoothing operator to use (WLS, bilateral, guided).
        mode: Pyramid construction mode (non_iterative or iterative).

    Returns:
        (base, details) where base is the coarsest smoothed image (2D),
        and details is a list of num_levels detail layers from fine to coarse.
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")

    # Compute strength and α for each level
    if num_levels == 1:
        strengths = [lam_coarse]
        alphas = [alpha_coarse]
    else:
        strengths = np.geomspace(lam_fine, lam_coarse, num_levels).tolist()
        alphas = np.linspace(alpha_fine, alpha_coarse, num_levels).tolist()

    # Compute progressively smoother versions
    smoothed = []
    for i in range(num_levels):
        if mode == PyramidMode.NON_ITERATIVE:
            # Eq. 13: always smooth the original
            source = image
        else:
            # Eq. 14: smooth the previous level's output
            source = smoothed[-1] if smoothed else image

        u = smooth(source, method=method, strength=strengths[i], alpha=alphas[i], log_luminance=log_luminance)
        smoothed.append(u)

    # Detail layers: d^i = u^{i-1} - u^i, with u^0 = g
    details = []
    prev = image
    for u in smoothed:
        details.append(prev - u)
        prev = u

    base = smoothed[-1]
    return base, details
