import numpy as np
from pytest import approx
from scipy import stats
from scipy.interpolate import RegularGridInterpolator

from starlord import cy_tools


def test_lpdf():
    assert cy_tools.uniform_lpdf(3.5, 3.1, 6.6) == approx(-np.log(3.5))
    assert cy_tools.uniform_lpdf(3.0, 3.1, 6.6) == -np.inf
    assert cy_tools.uniform_lpdf(6.7, 3.1, 6.6) == -np.inf
    for x in stats.uniform.rvs(-4, 4, 100):
        assert stats.norm.logpdf(x, 1., .5) == approx(cy_tools.normal_lpdf(x, 1., .5), rel=1e-12)
        assert stats.norm.logpdf(x, -10.1, 2.5) == approx(cy_tools.normal_lpdf(x, -10.1, 2.5), rel=1e-12)
        assert stats.norm.logpdf(x, 1e3, 1e2) == approx(cy_tools.normal_lpdf(x, 1e3, 1e2), rel=1e-12)
    assert np.isnan(cy_tools.normal_lpdf(5., 2., -1.5))
    for x in stats.uniform.rvs(0., 1., 100):
        assert stats.beta.logpdf(x, 15., 20.) == approx(cy_tools.beta_lpdf(x, 15., 20.), rel=1e-12)
        assert stats.beta.logpdf(x, 500., 300.) == approx(cy_tools.beta_lpdf(x, 500., 300.), rel=1e-12)
        assert stats.beta.logpdf(x, 53.2, 48.5) == approx(cy_tools.beta_lpdf(x, 53.2, 48.5), rel=1e-12)
    assert cy_tools.beta_lpdf(0., 5., 2.) == -np.inf
    assert cy_tools.beta_lpdf(1., 15., 2.3) == -np.inf
    assert np.isnan(cy_tools.beta_lpdf(1.01, 23., 15.3))
    assert np.isnan(cy_tools.beta_lpdf(-3., 23., 15.3))
    for x in stats.uniform.rvs(0., 10., 100):
        assert stats.gamma.logpdf(x, 15., scale=1. / 20.) == approx(cy_tools.gamma_lpdf(x, 15., 20.), rel=1e-12)
        assert stats.gamma.logpdf(x, 500., scale=1. / 300.) == approx(cy_tools.gamma_lpdf(x, 500., 300.), rel=1e-12)
        assert stats.gamma.logpdf(x, 53.2, scale=1. / 48.5) == approx(cy_tools.gamma_lpdf(x, 53.2, 48.5), rel=1e-12)


def test_ppf():
    for x in stats.uniform.rvs(0., 1., 100):
        assert cy_tools.normal_ppf(x, 1.3e4, 5.2e3) == approx(stats.norm.ppf(x, loc=1.3e4, scale=5.2e3), rel=1e-12)
        assert cy_tools.normal_ppf(x, -1.2e-3, 5.2e-3) == approx(
            stats.norm.ppf(x, loc=-1.2e-3, scale=5.2e-3), rel=1e-12)
        assert cy_tools.beta_ppf(x, 25.3, 12.2) == approx(stats.beta.ppf(x, 25.3, 12.2))
        assert cy_tools.beta_ppf(x, 230.3, 112.2) == approx(stats.beta.ppf(x, 230.3, 112.2))
        assert cy_tools.gamma_ppf(x, 25.3, 12.2) == approx(stats.gamma.ppf(x, 25.3, scale=1. / 12.2))
        assert cy_tools.gamma_ppf(x, 230.3, 112.2) == approx(stats.gamma.ppf(x, 230.3, scale=1. / 112.2))


def test_gridding1d():
    x = np.sin(np.linspace(0, np.pi / 2, 5))
    values = np.exp(x)
    f = cy_tools.GridInterpolator([x], values)
    # Check that we match RegularGridInterpolator
    g = RegularGridInterpolator([x], values)
    for xt in 0.9 * np.random.rand(50):
        assert f._interp1d(xt) == approx(g([xt])[0], rel=1e-12)
    assert f(.25) == approx(g([.25])[0], rel=1e-12)
    # Check bounds handling
    assert f._interp1d(1.) == approx(g([1.])[0], rel=1e-12)
    assert np.isnan(f._interp1d(-2))
    assert np.isnan(f._interp1d(1.1))
    assert np.isfinite(f._interp1d(0.))
    assert np.isfinite(f._interp1d(0.553))


def test_gridding2d():
    x = np.linspace(0, 10, 100)
    y = np.logspace(-1, 1, 75)
    values = np.sin(x[:, None]) + 1.2 * np.cos(.2 * y[None, :])
    f = cy_tools.GridInterpolator([x, y], values)
    # Check that we match RegularGridInterpolator
    g = RegularGridInterpolator([x, y], values)
    for xt in (0.1 + 9.9 * np.random.rand(50, 2)):
        assert f._interp2d(xt[0], xt[1]) == approx(g(xt)[0], rel=1e-12)
    assert f([4.32, 5.63]) == approx(g([4.32, 5.63])[0], rel=1e-12)
    # Check bounds handling
    assert np.isnan(f._interp2d(-5, -5))
    assert np.isnan(f._interp2d(5, 0.))
    assert np.isfinite(f._interp2d(10., 0.1))
    assert np.isfinite(f._interp2d(0., 10.))


def test_gridding3d():
    x = np.linspace(0, 10, 100)
    y = np.logspace(-1, 1, 75)
    z = np.linspace(-3, 13.5, 32)
    values = np.sin(x[:, None, None]) + 1.2 * np.cos(.2 * y[None, :, None]) + .25 * z[None, None, :]
    f = cy_tools.GridInterpolator([x, y, z], values)
    # Check that we match RegularGridInterpolator
    g = RegularGridInterpolator([x, y, z], values)
    for xt in (0.1 + 9.9 * np.random.rand(50, 3)):
        assert f._interp3d(xt[0], xt[1], xt[2]) == approx(g(xt)[0], rel=1e-12)
    assert f([4.32, 5.63, -2.5]) == approx(g([4.32, 5.63, -2.5])[0], rel=1e-12)
    # Check bounds handling
    assert np.isnan(f._interp3d(-5, -5, -5))
    assert np.isnan(f._interp3d(5, 0., 6.))
    assert np.isfinite(f._interp3d(10., 0.1, -3))
    assert np.isfinite(f._interp3d(0., 10., 13.5))


def test_gridding4d():
    x = np.linspace(0, 10, 100)
    y = np.logspace(-1, 1, 75)
    z = np.linspace(-3, 13.5, 32)
    u = np.logspace(-1, 1.5, 21)
    values = (
        np.sin(x[:, None, None, None]) + 1.2 * np.cos(.2 * y[None, :, None, None]) +
        (z[None, None, :, None] / 3.)**2 / u[None, None, None, :])
    f = cy_tools.GridInterpolator([x, y, z, u], values)
    # Check that we match RegularGridInterpolator
    g = RegularGridInterpolator([x, y, z, u], values)
    for xt in (0.1 + 9.9 * np.random.rand(50, 4)):
        assert f._interp4d(xt[0], xt[1], xt[2], xt[3]) == approx(g(xt)[0], rel=1e-12)
    assert f([4.32, 5.63, -2.5, 13.]) == approx(g([4.32, 5.63, -2.5, 13.])[0], rel=1e-12)
    # Check bounds handling
    assert np.isnan(f._interp4d(-5, -5, -5, -5))
    assert np.isnan(f._interp4d(5, 1., 6., 50.))
    assert np.isfinite(f._interp4d(10., 0.1, -3, .1))
    assert np.isfinite(f._interp4d(0., 10., 13.5, 10**1.5))
