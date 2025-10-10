from pathlib import Path

import numpy as np
import pytest

import starlord
from starlord._config import config


@pytest.fixture(scope="session")
def dummy_grid(tmpdir_factory: pytest.TempdirFactory):
    x = np.linspace(-5, 5, 75)[:, None]
    y = np.logspace(-1, 1., 25)[None, :]
    v1 = np.sin(x) + y
    v2 = 25. + np.cos(2.2 * x) / np.sin(y)
    g1 = "2.5*{x} + {v1}"
    g2 = "5+math.log10(v1)"
    grid_spec = "x, y -> v1, v2; g1, g2"
    fn = tmpdir_factory.mktemp("grids")
    np.savez_compressed(fn / "dummy.npz", grid_spec=grid_spec, x=x.flat, y=y.flat, v1=v1, v2=v2, g1=g1, g2=g2)
    return fn


def test_grid(dummy_grid):
    config.grid_dir = Path(dummy_grid)
    starlord.GridGenerator.reload_grids()
    assert "dummy" in starlord.GridGenerator._grids.keys()
    grid = starlord.GridGenerator.get_grid("dummy")
    assert grid.name == "dummy"
    assert grid.spec == "x, y -> v1, v2; g1, g2"
    assert grid.inputs == ["x", "y"]
    assert grid.outputs == ["v1", "v2"]
    assert grid.derived == ["g1", "g2"]
