"""Weighted Least Squares edge-preserving smoothing operator.

Implements Eq. 3-6 from Farbman et al. 2008:
    min_u  Σ (u_p - g_p)² + λ [ a_x (∂u/∂x)² + a_y (∂u/∂y)² ]

Solved via the linear system (Eq. 5):
    (I + λ·L_g) u = g
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _compute_smoothness_weights(
    log_luminance: np.ndarray, alpha: float, eps: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spatially-varying smoothness weights (Eq. 6).

    a_{x,p} = (|∂ℓ/∂x|^α + ε)^{-1}
    a_{y,p} = (|∂ℓ/∂y|^α + ε)^{-1}

    Returns flattened weight arrays (ax, ay) in row-major order.
    """
    # Forward differences; zero-pad the last column/row
    dl_dx = np.diff(log_luminance, axis=1)
    dl_dx = np.pad(dl_dx, ((0, 0), (0, 1)), mode="constant")

    dl_dy = np.diff(log_luminance, axis=0)
    dl_dy = np.pad(dl_dy, ((0, 1), (0, 0)), mode="constant")

    ax = 1.0 / (np.abs(dl_dx) ** alpha + eps)
    ay = 1.0 / (np.abs(dl_dy) ** alpha + eps)

    return ax, ay


def _build_system(g_flat: np.ndarray, ax: np.ndarray, ay: np.ndarray, lam: float, h: int, w: int) -> sp.csc_matrix:
    """Build the sparse matrix (I + λ·L_g) for the WLS system.

    L_g = Dx^T Ax Dx + Dy^T Ay Dy, where Dx/Dy are forward difference operators.
    Using the identity Dx^T Ax Dx, the resulting L_g is a 5-point stencil.
    """
    n = h * w
    ax_flat = ax.ravel()
    ay_flat = ay.ravel()

    # Horizontal connections: pixel p and p+1 (same row)
    # For forward difference Dx: (Dx u)_p = u_{p+1} - u_p
    # Dx^T Ax Dx contributes: ax_p on diagonal of p and p+1, -ax_p on off-diagonal
    # But we need to exclude the last column in each row.
    #
    # Indices of pixels that have a right neighbor
    idx = np.arange(n).reshape(h, w)
    horiz_left = idx[:, :-1].ravel()  # p
    horiz_right = idx[:, 1:].ravel()  # p+1
    horiz_w = ax_flat[horiz_left]  # weight at p

    # Vertical connections: pixel p and p+w (next row)
    vert_top = idx[:-1, :].ravel()  # p
    vert_bot = idx[1:, :].ravel()  # p+w
    vert_w = ay_flat[vert_top]  # weight at p

    # Build L_g as sum of contributions
    # Diagonal: identity + λ * (sum of weights from horizontal and vertical)
    diag = np.ones(n, dtype=np.float64)

    # Add horizontal contributions to diagonal
    np.add.at(diag, horiz_left, lam * horiz_w)
    np.add.at(diag, horiz_right, lam * horiz_w)

    # Add vertical contributions to diagonal
    np.add.at(diag, vert_top, lam * vert_w)
    np.add.at(diag, vert_bot, lam * vert_w)

    # Off-diagonal entries
    off_rows = np.concatenate([horiz_left, horiz_right, vert_top, vert_bot])
    off_cols = np.concatenate([horiz_right, horiz_left, vert_bot, vert_top])
    off_vals = np.concatenate([-lam * horiz_w, -lam * horiz_w, -lam * vert_w, -lam * vert_w])

    # Combine diagonal and off-diagonal
    rows = np.concatenate([np.arange(n), off_rows])
    cols = np.concatenate([np.arange(n), off_cols])
    vals = np.concatenate([diag, off_vals])

    return sp.csc_matrix((vals, (rows, cols)), shape=(n, n))


def wls_smooth(image: np.ndarray, lam: float, alpha: float, log_luminance: np.ndarray | None = None) -> np.ndarray:
    """Apply WLS edge-preserving smoothing to a 2D image.

    Args:
        image: 2D array (H, W) to smooth (e.g., CIELAB lightness channel).
        lam: Smoothness weight λ. Higher = smoother.
        alpha: Gradient sensitivity exponent (typically 1.2–2.0).
        log_luminance: Optional log-luminance for computing smoothness weights.
            If None, uses log(image + 1e-6).

    Returns:
        Smoothed 2D array of same shape.
    """
    h, w = image.shape

    if log_luminance is None:
        log_luminance = np.log(np.maximum(image, 1e-6))

    ax, ay = _compute_smoothness_weights(log_luminance, alpha)
    A = _build_system(image.ravel(), ax, ay, lam, h, w)
    u = spsolve(A, image.ravel())

    return u.reshape(h, w)
