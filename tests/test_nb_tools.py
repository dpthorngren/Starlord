from starlord import nb_tools as nt
from scipy import interpolate
import numpy as np
import pytest


def test_locate_point_uniform():
    # Test underlying python function for coverage and agreement; valid but may confuse linters
    locate_point = nt._locate_point_.py_func  # pyright: ignore
    process_axis = nt._process_axis_.py_func  # pyright: ignore
    # Uniform input axis
    n = 33
    x = np.linspace(5, 15, n)
    grid = process_axis(x, n)
    assert grid == pytest.approx(nt._process_axis_(x, n))
    for xt in 5. + 10. * np.random.rand(100):
        i, weight = nt._locate_point_(xt, grid, n)
        assert i == np.where(xt >= x)[0][-1]
        assert i < n - 1
        assert weight >= 0.
        assert weight < 1. or (weight == 1 and i == n - 2)
        i2, weight2 = locate_point(xt, grid, n)
        assert i2 == i2
        assert weight == pytest.approx(weight2)
    assert nt._locate_point_(3.2, grid, n)[0] == -1
    assert locate_point(3.2, grid, n)[0] == -1
    assert nt._locate_point_(16., grid, n)[0] == -1
    assert locate_point(16., grid, n)[0] == -1
    assert nt._locate_point_(np.nan, grid, n)[0] == -1
    assert locate_point(np.nan, grid, n)[0] == -1
    assert nt._locate_point_(np.inf, grid, n)[0] == -1
    assert locate_point(np.inf, grid, n)[0] == -1
    assert nt._locate_point_(15., grid, n) == (n - 2, 1.)
    assert locate_point(15., grid, n) == (n - 2, 1.)


def test_locate_point_nonuniform():
    # Test underlying python function for coverage and agreement; valid but may confuse linters
    locate_point = nt._locate_point_.py_func  # pyright: ignore
    process_axis = nt._process_axis_.py_func  # pyright: ignore
    # Non-uniform input axis
    n = 57
    x = np.logspace(-1, 2, n)
    grid = process_axis(x, n)
    assert grid == pytest.approx(nt._process_axis_(x, n))
    for xt in .1 + 99.9 * np.random.rand(100):
        i, weight = nt._locate_point_(xt, grid, n)
        assert i == np.where(xt >= x)[0][-1]
        assert i < n - 1
        assert weight >= 0.
        assert weight < 1. or (weight == 1 and i == n - 2)
        i2, weight2 = locate_point(xt, grid, n)
        assert i2 == i2
        assert weight == pytest.approx(weight2)
    assert nt._locate_point_(-2., grid, n)[0] == -1
    assert locate_point(-2., grid, n)[0] == -1
    assert nt._locate_point_(105., grid, n)[0] == -1
    assert locate_point(105., grid, n)[0] == -1
    assert nt._locate_point_(np.nan, grid, n)[0] == -1
    assert locate_point(np.nan, grid, n)[0] == -1
    assert nt._locate_point_(np.inf, grid, n)[0] == -1
    assert locate_point(np.inf, grid, n)[0] == -1
    assert nt._locate_point_(100., grid, n) == (n - 2, 1.)
    assert locate_point(100., grid, n) == (n - 2, 1.)


def test_1d_gridding():
    x = np.linspace(-5, 5, 100)
    y = np.sin(x) + 3.2 * np.cos(x) + .12*x
    grid = nt.pack_interpolator([x], y)
    assert grid[0] == nt._magicNumber
    assert grid[1] == 1.
    ref = interpolate.RegularGridInterpolator([x], y)
    for xt in (-5. + 10. * np.random.rand(100)):
        assert nt.interp1d(grid, xt) == pytest.approx(ref([xt])[0])
    assert nt.interp1d(grid, -5.) == pytest.approx(ref([-5.])[0])
    assert nt.interp1d(grid, 5.) == pytest.approx(ref([5.])[0])
    for xt in (6. + 1000. * np.random.rand(100)):
        assert np.isnan(nt.interp1d(grid, xt))
    for xt in (-6. - 1000. * np.random.rand(100)):
        assert np.isnan(nt.interp1d(grid, xt))
    assert np.isnan(nt.interp1d(grid, np.inf))
    assert np.isnan(nt.interp1d(grid, np.nan))
    # Test underlying python function for coverage; valid but may confuse linters
    interp1d = nt.interp1d.py_func  # pyright: ignore
    assert interp1d(grid, 3.5) == pytest.approx(ref([3.5])[0])
    assert np.isnan(interp1d(grid, -15.))
    assert np.isnan(interp1d(grid[::-1], 2.))


def test_2d_gridding():
    x = np.linspace(-5, 5, 73)
    y = np.logspace(-1, 1, 51)
    z = np.sin(x[:, None] + y[None, :]) + 3.2 * np.cos(y[None, :]) + .12 * x[:, None]
    grid = nt.pack_interpolator([x, y], z)
    assert grid[0] == nt._magicNumber
    assert grid[1] == 2.
    ref = interpolate.RegularGridInterpolator([x, y], z)
    xtest = np.random.rand(100, 2) * np.array([10., 9.9]) + np.array([-5., .1])
    for xt in xtest:
        assert nt.interp2d(grid, xt[0], xt[1]) == pytest.approx(ref(xt)[0])
    assert np.isnan(nt.interp2d(grid, -6, .2))
    assert np.isnan(nt.interp2d(grid, 3., .09999))
    assert np.isfinite(nt.interp2d(grid, -5., .5))
    assert np.isfinite(nt.interp2d(grid, -2., .1))
    assert np.isfinite(nt.interp2d(grid, -5., .1))
    assert np.isfinite(nt.interp2d(grid, 5., 1.))
    assert np.isfinite(nt.interp2d(grid, 3., 1.))
    assert np.isnan(nt.interp2d(grid, np.inf, 2.5))
    assert np.isnan(nt.interp2d(grid, -3.2, np.nan))
    # Test underlying python function for coverage; valid but may confuse linters
    interp2d = nt.interp2d.py_func  # pyright: ignore
    assert interp2d(grid, 3.5, .5) == pytest.approx(ref([3.5, .5])[0])
    assert np.isnan(interp2d(grid, -15, .2))
    assert np.isnan(interp2d(grid[::-1], 2., .5))


def test_3d_gridding():
    x = np.linspace(-5, 5, 73)
    y = np.logspace(-1, 1, 51)
    z = np.linspace(2, 3.5, 13)
    u = np.sin(x[:, None, None] + y[None, :, None]) + 3.2 * np.cos(z[None, None, :]) + .12 * x[:, None, None]
    grid = nt.pack_interpolator([x, y, z], u)
    assert grid[0] == nt._magicNumber
    assert grid[1] == 3.
    ref = interpolate.RegularGridInterpolator([x, y, z], u)
    xtest = np.random.rand(100, 3) * np.array([10., 9.9, 1.5]) + np.array([-5., .1, 2.])
    for xt in xtest:
        assert nt.interp3d(grid, *xt) == pytest.approx(ref(xt)[0])
    assert np.isnan(nt.interp3d(grid, -6, .2, 1.5))
    assert np.isnan(nt.interp3d(grid, 3., .19999, 3.6))
    assert np.isfinite(nt.interp3d(grid, -5., .1, 2.))
    assert np.isfinite(nt.interp3d(grid, -3., .1, 3.5))
    assert np.isfinite(nt.interp3d(grid, -5., 1., 2.))
    assert np.isnan(nt.interp3d(grid, np.inf, 2.5, 3.4))
    assert np.isnan(nt.interp3d(grid, -3.2, 2.5123, np.nan))
    # Test underlying python function for coverage; valid but may confuse linters
    interp3d = nt.interp3d.py_func  # pyright: ignore
    assert interp3d(grid, 3.5, .5, 2.2) == pytest.approx(ref([3.5, .5, 2.2])[0])
    assert np.isnan(interp3d(grid, -4, 15., 3.1))
    assert np.isnan(interp3d(grid, -4, .2, 3.51))
    assert np.isnan(interp3d(grid[::-1], 2., .5, 3.2))


def test_4d_gridding():
    x = np.linspace(-5, 5, 73)
    y = np.logspace(-1, 1, 51)
    z = np.linspace(2, 3.5, 13)
    u = np.array([-5, -3, 0., 1.5, 15.])
    v = np.sin(u[None, None, None, :] + .530 * y[None, :, None, None])
    v = v + 3.2 * np.cos(z[None, None, :, None]) + .12 * x[:, None, None, None]
    grid = nt.pack_interpolator([x, y, z, u], v)
    assert grid[0] == nt._magicNumber
    assert grid[1] == 4.
    ref = interpolate.RegularGridInterpolator([x, y, z, u], v)
    xtest = np.random.rand(100, 4) * np.array([10., 9.9, 1.5, 20.]) + np.array([-5., .1, 2., -5])
    for xt in xtest:
        assert nt.interp4d(grid, *xt) == pytest.approx(ref(xt)[0])
    assert np.isnan(nt.interp4d(grid, -6, .2, 2.5, 16.1))
    assert np.isnan(nt.interp4d(grid, 3., .19999, 3.3, -5.001))
    assert np.isfinite(nt.interp4d(grid, -5., .1, 2., -5))
    assert np.isfinite(nt.interp4d(grid, -3., .15, 2.5, 15.))
    assert np.isfinite(nt.interp4d(grid, 5., 1., 3.5, 15))
    assert np.isnan(nt.interp4d(grid, -4., 2.5, 3.4, np.inf))
    assert np.isnan(nt.interp4d(grid, -3.2, 2.5123, 3.4, np.nan))
    # Test underlying python function for coverage; valid but may confuse linters
    interp4d = nt.interp4d.py_func  # pyright: ignore
    assert interp4d(grid, 3.5, .5, 2.2, 0.1) == pytest.approx(ref([3.5, .5, 2.2, 0.1])[0])
    assert np.isnan(interp4d(grid, -4, 2., 2.9, 15.1))
    assert np.isnan(interp4d(grid, -4, .2, .2, -5.1))
    assert np.isnan(interp4d(grid[::-1], 2., .5, 3.2, 3.5))
