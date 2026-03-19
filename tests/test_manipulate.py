import numpy as np

from epd.manipulate import _soft_boost, multi_scale_tone_manipulate


def test_soft_boost_identity() -> None:
    """With factor=1, should return x unchanged."""
    x = np.linspace(-50, 50, 100)
    result = _soft_boost(1.0, x)
    np.testing.assert_allclose(result, x, atol=1e-10)


def test_soft_boost_zero_at_origin() -> None:
    assert abs(_soft_boost(2.0, np.array([0.0]))[0]) < 1e-10


def test_soft_boost_odd_function() -> None:
    x = np.linspace(0.1, 20, 50)
    for a in [0.5, 2.0, 5.0]:
        np.testing.assert_allclose(_soft_boost(a, -x), -_soft_boost(a, x), atol=1e-10)


def test_soft_boost_amplifies_small_values() -> None:
    """For small values, boost(2, x) ≈ 2*x."""
    x = np.array([0.1, 0.5, 1.0])
    result = _soft_boost(3.0, x)
    # For small x relative to clip_range=100, should be nearly linear
    np.testing.assert_allclose(result, 3.0 * x, atol=0.1)


def test_soft_boost_clips_large_values() -> None:
    """For very large boosted values, output should saturate."""
    x = np.array([80.0])
    result = _soft_boost(10.0, x, clip_range=100.0)
    assert abs(result[0]) < 100.0  # should be soft-clipped below range


def test_manipulate_no_change() -> None:
    """With default parameters (all boosts=1, exposure=1), output ≈ input."""
    np.random.seed(42)
    img = np.random.rand(30, 30, 3) * 0.8 + 0.1
    result = multi_scale_tone_manipulate(img, exposure=1.0, base_boost=1.0, detail_boosts=[1.0, 1.0, 1.0])
    np.testing.assert_allclose(result, img, atol=0.02)


def test_manipulate_detail_boost_changes_output() -> None:
    """Boosting details should produce a visibly different result."""
    np.random.seed(42)
    img = np.random.rand(30, 30, 3) * 0.8 + 0.1
    result_neutral = multi_scale_tone_manipulate(img, detail_boosts=[1.0, 1.0, 1.0])
    result_boosted = multi_scale_tone_manipulate(img, detail_boosts=[5.0, 5.0, 5.0])
    diff = np.abs(result_boosted - result_neutral).mean()
    assert diff > 0.01, f"Boosted result too similar to neutral: mean diff = {diff}"


def test_manipulate_output_range() -> None:
    np.random.seed(42)
    img = np.random.rand(20, 20, 3)
    result = multi_scale_tone_manipulate(img, exposure=1.5, base_boost=2.0, detail_boosts=[3.0, 3.0, 3.0])
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_manipulate_output_shape() -> None:
    img = np.random.rand(25, 35, 3)
    result = multi_scale_tone_manipulate(img, num_levels=2, detail_boosts=[1.0, 1.0])
    assert result.shape == img.shape


def test_manipulate_wrong_boosts_count() -> None:
    img = np.random.rand(10, 10, 3)
    try:
        multi_scale_tone_manipulate(img, num_levels=3, detail_boosts=[1.0, 1.0])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
