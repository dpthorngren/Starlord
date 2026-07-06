import numpy as np
import pytest
from pytest import approx
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp

from starlord import cy_tools


@pytest.mark.flaky(reruns=3)
def test_helpers():
    for a, b in 20 * np.random.rand(100, 2):
        assert cy_tools.logsumexp(a, b) == approx(logsumexp([a, b]), rel=1e-12)


def test_cdf():
    assert cy_tools.normal_cdf(0, 0., 1.) == approx(5., 5., 12.)
    assert cy_tools.normal_cdf(-100, -53., 6.0) == approx(0., abs=1e-12)
    assert cy_tools.normal_cdf(50, -7.3, 3.5) == approx(1.0, abs=1e-12)
    assert cy_tools.exponential_cdf(-5, 23) == 0.0
    for rng in np.random.rand(100, 3):
        x, mu, sigma = rng * np.array([2.0, 1.0, 0.2]) + np.array([-1.0, -0.6, 0.01])
        assert cy_tools.normal_cdf(x, mu, sigma) == approx(stats.norm.cdf(x, mu, sigma), rel=1e-12)
        assert cy_tools.exponential_cdf(abs(x), sigma) == approx(stats.expon.cdf(abs(x), scale=1 / sigma), rel=1e-12)


def test_lpdf():
    assert cy_tools.uniform_lpdf(3.5, 3.1, 6.6) == approx(-np.log(3.5))
    assert cy_tools.uniform_lpdf(3.0, 3.1, 6.6) == -np.inf
    assert cy_tools.uniform_lpdf(6.7, 3.1, 6.6) == -np.inf
    for x in stats.uniform.rvs(-4, 4, 100):
        assert stats.norm.logpdf(x, 1., .5) == approx(cy_tools.normal_lpdf(x, 1., .5), rel=1e-12)
        assert stats.norm.logpdf(x, -10.1, 2.5) == approx(cy_tools.normal_lpdf(x, -10.1, 2.5), rel=1e-12)
        assert stats.norm.logpdf(x, 1e3, 1e2) == approx(cy_tools.normal_lpdf(x, 1e3, 1e2), rel=1e-12)
        expect = stats.truncnorm.logpdf(x, 3., 4., 1., .5)
        assert cy_tools.trunc_normal_lpdf(x, 1., .5, 3., 4.) == approx(expect, rel=1e-12)
        expect = stats.truncnorm.logpdf(x, -15, -5, -10.1, 2.5)
        assert cy_tools.trunc_normal_lpdf(x, -10.1, 2.5, -15, -5) == approx(expect, rel=1e-12)
        expect = stats.truncnorm.logpdf(x, 5e2, 2e3, 1e3, 1e2)
        assert cy_tools.trunc_normal_lpdf(x, 1e3, 1e2, 5e2, 2e3) == approx(expect, rel=1e-12)
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
        assert cy_tools.trunc_exponential_lpdf(x, 2.5, 0., 100.) == approx(cy_tools.exponential_lpdf(x, 2.5), 1e-12)
        assert cy_tools.trunc_exponential_lpdf(x, 2.5, 0., np.inf) == approx(cy_tools.exponential_lpdf(x, 2.5), 1e-12)
        x += 3.5
        expect = cy_tools.exponential_lpdf(x - 3., 2.5)
        assert cy_tools.trunc_exponential_lpdf(x, 2.5, 3., np.inf) == approx(expect, 1e-12)
        expect = np.log(2.5) - x*2.5 - cy_tools.logsumexp(-2.5 * 3, -2.5 * 15, 1., -1.)
        assert cy_tools.trunc_exponential_lpdf(x, 2.5, 3.0, 15.) == approx(expect, abs=1e-12)
    assert cy_tools.exponential_lpdf(-1e-5, 3.) == -np.inf
    for x in stats.uniform.rvs(0., 10., 100):
        assert stats.expon.logpdf(x, scale=1. / 20.) == approx(cy_tools.exponential_lpdf(x, 20.), rel=1e-12)
    assert cy_tools.trunc_power_lpdf(4.0, 0., 3.1, 6.6) == approx(-np.log(3.5))
    for k in stats.uniform.rvs(.1, 3, 100):
        expect = cy_tools.trunc_power_lpdf(2.0, k, 1., 5.)
        assert cy_tools.trunc_power_lpdf(4.0, k, 1., 5.) - np.log(2**k) == approx(expect, rel=1e-12)
        expect = cy_tools.trunc_power_lpdf(1.5, -k, 1., 5.)
        assert cy_tools.trunc_power_lpdf(3.0, -k, 1., 5.) + np.log(2**k) == approx(expect, rel=1e-12)
    expect = cy_tools.trunc_power_lpdf(1.5, -1, 1., 5.)
    assert cy_tools.trunc_power_lpdf(3.0, -1, 1., 5.) - np.log(0.5) == approx(expect, rel=1e-12)
    # Astrophyical priors
    expect = cy_tools.chabrier_lpdf(-1e-5, 0., -1.10237, 0.69, 5.295945)
    assert cy_tools.chabrier_lpdf(1e-5, 0., -1.10237, 0.69, 5.295945) == approx(expect, rel=1e-1)
    expect = cy_tools.chabrier_lpdf(-1e-5, 0.045757, -0.48148, 0.34, 5.295945)
    assert cy_tools.chabrier_lpdf(1e-5, 0.045757, -0.48148, 0.34, 5.295945) == approx(expect, rel=1e-1)
    assert cy_tools.chabrier_lpdf(-.5, 0., -1.10237, 0.69, 5.295945) == approx(-0.9040384867245783, rel=1e-12)
    assert cy_tools.chabrier_lpdf(-.1, 0., -1.10237, 0.69, 5.295945) == approx(-1.57815736931227, rel=1e-12)


def test_ppf():
    for p in stats.uniform.rvs(0., 1., 100):
        assert cy_tools.normal_ppf(p, 1.3e4, 5.2e3) == approx(stats.norm.ppf(p, loc=1.3e4, scale=5.2e3), rel=1e-12)
        assert cy_tools.normal_ppf(p, -1.2e-3, 5.2e-3) == approx(
            stats.norm.ppf(p, loc=-1.2e-3, scale=5.2e-3), rel=1e-12)
        assert cy_tools.beta_ppf(p, 25.3, 12.2) == approx(stats.beta.ppf(p, 25.3, 12.2))
        assert cy_tools.beta_ppf(p, 230.3, 112.2) == approx(stats.beta.ppf(p, 230.3, 112.2))
        assert cy_tools.gamma_ppf(p, 25.3, 12.2) == approx(stats.gamma.ppf(p, 25.3, scale=1. / 12.2))
        assert cy_tools.gamma_ppf(p, 230.3, 112.2) == approx(stats.gamma.ppf(p, 230.3, scale=1. / 112.2))
        assert cy_tools.exponential_ppf(p, 0.15) == approx(stats.expon.ppf(p, scale=1. / .15))
        assert cy_tools.exponential_ppf(p, 13.25) == approx(stats.expon.ppf(p, scale=1. / 13.25))
        assert cy_tools.trunc_power_ppf(p, 0., 5, 10) == approx(5 + 5*p, rel=1e-12)
        assert cy_tools.trunc_power_ppf(p, 1., 1, 2) == approx(np.sqrt(1 + 3*p))
        expect = stats.truncnorm.ppf(p, 0., 1., loc=-1.2e-3, scale=5.2e-3)
        assert cy_tools.trunc_normal_ppf(p, -1.2e-3, 5.2e-3, -1.2e-3, -1.2e-3 + 5.2e-3) == approx(expect, rel=1e-12)
        expect = stats.truncnorm.ppf(p, -1.3e4 / 5.2e3, np.inf, loc=1.3e4, scale=5.2e3)
        assert cy_tools.trunc_normal_ppf(p, 1.3e4, 5.2e3, 0, np.inf) == approx(expect, rel=1e-12)
        expect = stats.norm.ppf(p, 3.5, .53)
        assert cy_tools.trunc_normal_ppf(p, 3.5, .53, -np.inf, np.inf) == approx(expect, rel=1e-12)
        expect = cy_tools.exponential_ppf(p, .37)
        assert cy_tools.trunc_exponential_ppf(p, 0.37, 0., 1e9) == approx(expect, rel=1e-12)
        x = cy_tools.exponential_ppf(p, 0.15)
        assert cy_tools.exponential_cdf(x, 0.15) == approx(p, rel=1e-12)
        expect = stats.truncexpon.ppf(p, .15 * 5., scale=1. / .15)
        assert cy_tools.trunc_exponential_ppf(p, 0.15, 0, 5.) == approx(expect)
        expect = stats.truncexpon.ppf(p, 13.25 * 6.5, scale=1. / 13.25)
        assert cy_tools.trunc_exponential_ppf(p, 13.25, 0, 6.5) == approx(expect)


def test_binorm():
    # If distributions are the same, should output the same as normal dist
    for x in 5 * np.random.randn(5):
        expect = cy_tools.normal_lpdf(x, -1.5, 2.0)
        assert cy_tools.binorm_lpdf(x, .25, -1.5, -1.5, 2.0, 2.0) == approx(expect, 1e-9)
        expect = cy_tools.normal_cdf(x, 31.5, 0.153)
        assert cy_tools.binorm_cdf(x, .75, 31.5, 35.1, .153, .153) == approx(expect, 1e-9)
    for p in np.random.rand(5):
        expect = cy_tools.normal_ppf(p, -12.3, 2.31)
        assert cy_tools.binorm_ppf(p, .75, -12.3, -12.3, 2.31, 2.31) == approx(expect, 1e-8)
    # Check that cdf and pdf are inverses of each other
    for w1, mu1, mu2, s1, s2 in 3 * np.random.randn(50, 5):
        w1 = 1 / (1 + np.exp(-w1))
        s1 = np.exp(s1 / 5)
        s2 = np.exp(s2 / 5)
        p = np.random.rand()
        x = cy_tools.binorm_ppf(p, w1, mu1, mu2, s1, s2)
        assert cy_tools.binorm_cdf(x, w1, mu1, mu2, s1, s2) == approx(p, 1e-6)


@pytest.mark.flaky(reruns=3)
def test_mvnormal():
    # Poorly sampled uniform distribution to get a random covariance
    x = np.random.rand(4, 10)
    x[0] += .8 * x[1]
    x[2] += -.3 * x[3]
    mu = 5 * np.random.randn(4)
    cov = np.cov(x)
    cov_chol = np.linalg.cholesky(cov)
    output = np.zeros([1000, 4])
    for i in range(1000):
        input = np.random.randn(4)
        cy_tools.multinormal_zppf(cov_chol, input, output[i], mu)
    assert cov == approx(np.cov(output.T), rel=.1, abs=.01)
    assert mu == approx(np.mean(output, axis=0), rel=.1, abs=.01)


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


def test_gridding5d():
    x = np.linspace(0, 10, 100)
    y = np.logspace(-1, 1, 75)
    z = np.linspace(-3, 13.5, 32)
    u = np.logspace(-1, 1.5, 21)
    v = np.sqrt(np.linspace(0, 100, 33))
    values = (
        np.sin(x[:, None, None, None, None]) + 1.2 * np.cos(.2 * y[None, :, None, None, None]) +
        (z[None, None, :, None, None] / 3.)**2 / u[None, None, None, :, None] +
        np.sin(0.3 + v[None, None, None, None, :]))
    f = cy_tools.GridInterpolator([x, y, z, u, v], values)
    # Check that we match RegularGridInterpolator
    g = RegularGridInterpolator([x, y, z, u, v], values)
    for xt in (0.1 + 9.9 * np.random.rand(50, 5)):
        assert f._interp5d(xt[0], xt[1], xt[2], xt[3], xt[4]) == approx(g(xt)[0], rel=1e-12)
    assert f([4.32, 5.63, -2.5, 13., 7.]) == approx(g([4.32, 5.63, -2.5, 13., 7.])[0], rel=1e-12)
    # Check bounds handling
    assert np.isnan(f._interp5d(-5, -5, -5, -5, -5))
    assert np.isnan(f._interp5d(5, 1., 6., 50., 12.))
    assert np.isfinite(f._interp5d(10., 0.1, -3, .1, 0.))
    assert np.isfinite(f._interp5d(0., 10., 13.5, 10**1.5, 10.))
