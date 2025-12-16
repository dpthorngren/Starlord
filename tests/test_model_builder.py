from pathlib import Path

# flake8: noqa
from test_grids import dummy_grids

import starlord
from starlord._config import config


def test_model_builder_variables():
    fitter = starlord.ModelBuilder()
    assert fitter is not None
    fitter.assign("x", "2.643")
    fitter.expression("logL += p.foo*p.foo / 10.")
    assert fitter._gen.params == ("p.foo",)
    assert fitter._gen.locals == ("l.x",)
    fitter.assign("b.something", "3.5*(p.foo - c.bar)")


def test_recursive_grids(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    fitter = starlord.ModelBuilder()
    fitter.constraint("rdummy.d", "normal", [10., 1.])
    # Running to resolve the grids
    print(fitter.summary())
    assert fitter._gen.params == ('p.b', 'p.x', 'p.y')
    assert fitter._gen.locals == ('l.dummy_g1', 'l.dummy_v1', 'l.rdummy_c', 'l.rdummy_d')
    assert fitter._gen.constants == ('c.grid_dummy_v1', 'c.grid_rdummy_c')


def test_param_overrides(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    fitter = starlord.ModelBuilder()
    fitter.constraint("dummy.v1", "normal", [3., 1.])
    fitter.override_input("dummy", "y", "5.0 + c.fixed_y")
    fitter.prior("p.x", "normal", [2., 5.])
    # Running to resolve the grids
    print(fitter.summary())
    assert fitter._gen.params == ('p.x',)
    assert fitter._gen.locals == ('l.dummy_v1',)
    assert fitter._gen.constants == ('c.fixed_y', 'c.grid_dummy_v1')
    code = fitter.generate()
    assert "5.0 + c_fixed_y" in code
    assert "params[1]" not in code
