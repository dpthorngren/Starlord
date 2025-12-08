from collections import OrderedDict
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
    config.grid_dir = Path(tmpdir_factory.mktemp("grids"))
    starlord.GridGenerator.create_grid(
        "dummy",
        inputs=OrderedDict(x=x.flatten(), y=y.flatten()),
        outputs=dict(v1=v1, v2=v2),
        derived=dict(g1="2.5*(5+{x}) + {v1}", g2="0.5+math.log10({g1})"))
    # Add a grid that recursively depends on the first
    a = np.linspace(-1, 15, 75)[:, None]
    b = np.linspace(-3, 3, 35)[None, :]
    c = a**2 / np.cos(b)
    starlord.GridGenerator.create_grid(
        "rdummy",
        inputs=OrderedDict(a=a.flatten(), b=b.flatten()),
        outputs=dict(c=c),
        derived=dict(d="math.exp({c})"),
        default_inputs=dict(a="dummy.g1"))
    # Add some non-grids to test GridGenerator filtering.
    nonGrid = config.grid_dir / "filter_test.txt"
    nonGrid.write_text("Filler to make sure the GridGenerator ignores this file.", "utf-8")
    np.savez_compressed(config.grid_dir / "nongrid.npz", x=x.flat, y=y.flat)
    return config.grid_dir


def test_default_inputs(dummy_grids):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    assert "dummy" in starlord.GridGenerator._grids.keys()
    assert "rdummy" in starlord.GridGenerator._grids.keys()
    # Checking that the grid loads and parses the string correctly
    grid = starlord.GridGenerator.get_grid("rdummy")
    assert grid.name == "rdummy"
    assert grid.spec == "a, b -> c; d"
    assert grid._default_inputs == {"a": "dummy.g1", "b": "p.b"}


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
    assert grid.derived == dict(g1="2.5*(5+{x}) + {v1}", g2="0.5+math.log10({g1})")
    assert grid._default_inputs == {"x": "p.x", "y": "p.y"}
    assert grid.shape == (75, 25)
    assert grid.bounds[0, 0] == -5.
    assert grid.bounds[0, 1] == 5.
    assert grid.bounds[1, 0] == 0.1
    assert grid.bounds[1, 1] == 10.
    assert str(grid) == "Grid_dummy(x, y -> v1, v2; g1, g2)"


def test_grid_building(dummy_grids):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    assert "dummy" in starlord.GridGenerator._grids.keys()
    grid = starlord.GridGenerator.get_grid("dummy")
    with pytest.raises(AssertionError):
        grid.build_grid("foo")
    f = grid.build_grid("v1")
    assert f.shape == (75, 25)
    assert f.bounds[0, 0] == -5.
    assert f.bounds[0, 1] == 5.
    assert f.bounds[1, 0] == 0.1
    assert f.bounds[1, 1] == 10.
    assert f._interp2d(1., 2.5) == pytest.approx(np.sin(1.) + 2.5, .01)
    g = grid.build_grid("v2")
    assert g._interp2d(3., 2.3) == pytest.approx(25. + np.cos(2.2 * 3.) / np.sin(2.3), .01)
