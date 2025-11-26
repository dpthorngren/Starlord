from pathlib import Path

import numpy as np
import pytest

import starlord
from starlord._config import config


@pytest.fixture(scope="session")
def dummy_grids(tmpdir_factory: pytest.TempdirFactory):
    x = np.linspace(-5, 5, 75)[:, None]
    y = np.logspace(-1, 1., 25)[None, :]
    v1 = np.sin(x) + y
    v2 = 25. + np.cos(2.2 * x) / np.sin(y)
    fn = tmpdir_factory.mktemp("grids")
    np.savez_compressed(
        fn / "dummy.npz",
        grid_spec="x, y -> v1, v2; g1, g2",
        x=x.flat,
        y=y.flat,
        v1=v1,
        v2=v2,
        g1="2.5*(5+{x}) + {v1}",
        g2="0.5+math.log10({g1})",
    )
    # Add a grid that recursively depends on the first
    a = np.linspace(-1, 15, 75)[:, None]
    b = np.linspace(-3, 3, 35)[None, :]
    c = a**2 / np.cos(b)
    np.savez_compressed(
        fn / "rdummy.npz",
        grid_spec="a, b -> c; d",
        a=a.flat,
        b=b.flat,
        c=c,
        d="math.exp({c})",
        param_defaults="a: dummy.g1",
    )
    # Add some non-grids to test GridGenerator filtering.
    nonGrid = fn / "filter_test.txt"
    nonGrid.write_text("Filler to make sure the GridGenerator ignores this file.", "utf-8")
    np.savez_compressed(fn / "nongrid.npz", x=x.flat, y=y.flat)
    return Path(fn)


def test_param_defaults(dummy_grids):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    assert "dummy" in starlord.GridGenerator._grids.keys()
    assert "rdummy" in starlord.GridGenerator._grids.keys()
    # Checking that the grid loads and parses the string correctly
    grid = starlord.GridGenerator.get_grid("rdummy")
    assert grid.name == "rdummy"
    assert grid.spec == "a, b -> c; d"
    assert grid.param_defaults == {"a": "dummy.g1", "b": "p.b"}


def test_grid_parsing(dummy_grids):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    assert "dummy" in starlord.GridGenerator._grids.keys()
    assert "nongrid" not in starlord.GridGenerator._grids.keys()
    assert "filter_test" not in starlord.GridGenerator._grids.keys()
    assert "filter_test.txt" not in starlord.GridGenerator._grids.keys()
    with pytest.raises(ValueError):
        starlord.GridGenerator.register_grid(dummy_grids / "nongrid.npz")
    with pytest.raises(FileNotFoundError):
        starlord.GridGenerator.register_grid(dummy_grids / "nonexistent.npz")
    with pytest.raises(KeyError):
        starlord.GridGenerator.get_grid("foo")
    grid = starlord.GridGenerator.get_grid("dummy")
    assert grid.name == "dummy"
    assert grid.spec == "x, y -> v1, v2; g1, g2"
    assert grid.inputs == ["x", "y"]
    assert grid.outputs == ["v1", "v2"]
    assert grid.derived == ["g1", "g2"]
    assert grid.param_defaults == {"x": "p.x", "y": "p.y"}
    assert str(grid) == "Grid_dummy(x, y -> v1, v2; g1, g2)"


def test_grid_building(dummy_grids):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    assert "dummy" in starlord.GridGenerator._grids.keys()
    grid = starlord.GridGenerator.get_grid("dummy")
    with pytest.raises(AssertionError):
        grid.build_grid("foo")
    f = grid.build_grid("v1")
    assert f._interp2d(1., 2.5) == pytest.approx(np.sin(1.) + 2.5, .01)
    g = grid.build_grid("v2")
    assert g._interp2d(3., 2.3) == pytest.approx(25. + np.cos(2.2 * 3.) / np.sin(2.3), .01)
