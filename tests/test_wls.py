import numpy as np

from epd.smoothing import _compute_smoothness_weights, wls_smooth


def test_smoothness_weights_shape() -> None:
    img = np.random.rand(10, 15)
    ax, ay = _compute_smoothness_weights(img, alpha=1.2)
    assert ax.shape == (10, 15)
    assert ay.shape == (10, 15)


def test_smoothness_weights_high_at_flat_regions() -> None:
    img = np.ones((10, 10)) * 0.5
    ax, ay = _compute_smoothness_weights(img, alpha=1.2)
    assert np.all(ax > 9000)
    assert np.all(ay > 9000)


def test_smoothness_weights_low_at_edges() -> None:
    img = np.zeros((10, 10))
    img[:, 5:] = 10.0
    ax, ay = _compute_smoothness_weights(img, alpha=1.2)
    assert ax[5, 4] < ax[5, 0]


def test_wls_identity_at_zero_lambda() -> None:
    img = np.random.rand(20, 20)
    result = wls_smooth(img, lam=0.0, alpha=1.2)
    np.testing.assert_allclose(result, img, atol=1e-10)


def test_wls_smooths_noise() -> None:
    np.random.seed(42)
    img = np.ones((30, 30)) * 50.0 + np.random.randn(30, 30) * 2.0
    result = wls_smooth(img, lam=1.0, alpha=1.2)
    assert result.std() < img.std()


def test_wls_preserves_edges() -> None:
    img = np.zeros((30, 30))
    img[:, 15:] = 100.0
    result = wls_smooth(img, lam=1.0, alpha=1.2)
    edge_diff = np.abs(result[15, 16] - result[15, 14])
    assert edge_diff > 80.0


def test_wls_output_shape() -> None:
    img = np.random.rand(25, 35)
    result = wls_smooth(img, lam=0.5, alpha=1.4)
    assert result.shape == img.shape


def test_wls_increasing_lambda_increases_smoothing() -> None:
    np.random.seed(42)
    img = np.random.rand(20, 20) * 50.0
    r1 = wls_smooth(img, lam=0.1, alpha=1.2)
    r2 = wls_smooth(img, lam=1.0, alpha=1.2)
    grad1 = np.sum(np.abs(np.diff(r1, axis=0))) + np.sum(np.abs(np.diff(r1, axis=1)))
    grad2 = np.sum(np.abs(np.diff(r2, axis=0))) + np.sum(np.abs(np.diff(r2, axis=1)))
    assert grad2 < grad1
