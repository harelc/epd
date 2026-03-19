import numpy as np
import pytest

from epd.smoothing import SmoothingMethod, bilateral_smooth, guided_smooth, smooth


def test_bilateral_smooths_noise() -> None:
    np.random.seed(42)
    img = np.ones((30, 30)) * 50.0 + np.random.randn(30, 30) * 2.0
    result = bilateral_smooth(img, sigma_s=4.0, sigma_r=5.0)
    assert result.std() < img.std()


def test_bilateral_preserves_edges() -> None:
    img = np.zeros((30, 30))
    img[:, 15:] = 100.0
    result = bilateral_smooth(img, sigma_s=4.0, sigma_r=5.0)
    edge_diff = np.abs(result[15, 16] - result[15, 14])
    assert edge_diff > 50.0


def test_bilateral_output_shape() -> None:
    img = np.random.rand(20, 25)
    result = bilateral_smooth(img, sigma_s=3.0, sigma_r=0.1)
    assert result.shape == img.shape


def test_guided_smooths_noise() -> None:
    np.random.seed(42)
    img = np.ones((30, 30)) * 50.0 + np.random.randn(30, 30) * 2.0
    result = guided_smooth(img, radius=5, eps=10.0)
    assert result.std() < img.std()


def test_guided_preserves_edges() -> None:
    img = np.zeros((30, 30))
    img[:, 15:] = 100.0
    result = guided_smooth(img, radius=5, eps=1.0)
    edge_diff = np.abs(result[15, 16] - result[15, 14])
    assert edge_diff > 50.0


def test_guided_output_shape() -> None:
    img = np.random.rand(20, 25)
    result = guided_smooth(img, radius=3, eps=0.01)
    assert result.shape == img.shape


@pytest.mark.parametrize("method", [SmoothingMethod.WLS, SmoothingMethod.BILATERAL, SmoothingMethod.GUIDED])
def test_unified_smooth_runs(method: SmoothingMethod) -> None:
    np.random.seed(42)
    img = np.random.rand(20, 20) * 50.0 + 10.0
    log_lum = np.log(np.maximum(img, 1e-6))
    result = smooth(img, method=method, strength=0.2, alpha=1.2, log_luminance=log_lum)
    assert result.shape == img.shape
    assert np.isfinite(result).all()
