import numpy as np
import pytest

from epd.decompose import PyramidMode, edge_preserving_decomposition
from epd.smoothing import SmoothingMethod


def test_reconstruction() -> None:
    np.random.seed(42)
    img = np.random.rand(30, 30) * 80.0 + 10.0
    base, details = edge_preserving_decomposition(img, num_levels=3)
    reconstructed = base + sum(details)
    np.testing.assert_allclose(reconstructed, img, atol=1e-8)


def test_num_detail_layers() -> None:
    img = np.random.rand(20, 20) * 50.0
    base, details = edge_preserving_decomposition(img, num_levels=4)
    assert len(details) == 4


def test_base_is_smooth() -> None:
    np.random.seed(42)
    img = np.random.rand(30, 30) * 50.0
    base, _ = edge_preserving_decomposition(img, num_levels=3)
    grad_orig = np.sum(np.abs(np.diff(img, axis=0))) + np.sum(np.abs(np.diff(img, axis=1)))
    grad_base = np.sum(np.abs(np.diff(base, axis=0))) + np.sum(np.abs(np.diff(base, axis=1)))
    assert grad_base < grad_orig


def test_details_non_trivial() -> None:
    np.random.seed(42)
    img = np.random.rand(40, 40) * 80.0
    _, details = edge_preserving_decomposition(img, num_levels=3)
    for d in details:
        assert np.max(np.abs(d)) > 0


def test_single_level() -> None:
    img = np.random.rand(15, 15) * 50.0
    base, details = edge_preserving_decomposition(img, num_levels=1)
    assert len(details) == 1
    np.testing.assert_allclose(base + details[0], img, atol=1e-8)


def test_invalid_num_levels() -> None:
    img = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        edge_preserving_decomposition(img, num_levels=0)


def test_iterative_mode_reconstruction() -> None:
    np.random.seed(42)
    img = np.random.rand(25, 25) * 80.0 + 10.0
    base, details = edge_preserving_decomposition(img, num_levels=3, mode=PyramidMode.ITERATIVE)
    reconstructed = base + sum(details)
    np.testing.assert_allclose(reconstructed, img, atol=1e-8)


@pytest.mark.parametrize("method", [SmoothingMethod.WLS, SmoothingMethod.BILATERAL, SmoothingMethod.GUIDED])
def test_reconstruction_all_methods(method: SmoothingMethod) -> None:
    np.random.seed(42)
    img = np.random.rand(20, 20) * 80.0 + 10.0
    base, details = edge_preserving_decomposition(img, num_levels=2, method=method)
    reconstructed = base + sum(details)
    np.testing.assert_allclose(reconstructed, img, atol=1e-8)


@pytest.mark.parametrize("mode", [PyramidMode.NON_ITERATIVE, PyramidMode.ITERATIVE])
def test_reconstruction_both_modes(mode: PyramidMode) -> None:
    np.random.seed(42)
    img = np.random.rand(20, 20) * 80.0 + 10.0
    base, details = edge_preserving_decomposition(img, num_levels=3, mode=mode)
    reconstructed = base + sum(details)
    np.testing.assert_allclose(reconstructed, img, atol=1e-8)
