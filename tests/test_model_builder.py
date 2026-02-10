from pathlib import Path

# flake8: noqa
from test_grids import dummy_grids

import starlord
from starlord._config import config


def test_deferred_handling(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    src = "{l.foo = 10**d.dummy.v1 + d.dummy.x + p.dont_catch + c.ditto"
    var, source = starlord.ModelBuilder._extract_deferred(src)
    assert var == ["dummy__v1", "dummy__x"]


def test_model_builder_variables():
    fitter = starlord.ModelBuilder()
    assert fitter is not None
    fitter.assign("x", "2.643")
    fitter.expression("logL += p.foo*p.foo / 10.")
    fitter.summary()
    assert fitter._code_generator is not None
    assert fitter._code_generator.params == ("p.foo",)
    assert fitter._code_generator.locals == ("l.x",)
    fitter.assign("b.something", "3.5*(p.foo - c.bar)")


def test_recursive_grids(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    fitter = starlord.ModelBuilder()
    fitter.constraint("d.rdummy.d", "normal", [10., 1.])
    # Running to resolve the grids
    print(fitter.summary())
    assert fitter._code_generator is not None
    assert fitter._code_generator.params == ('p.b', 'p.x', 'p.y')
    assert fitter._code_generator.locals == ('l.dummy__g1', 'l.dummy__v1', 'l.rdummy__c', 'l.rdummy__d')
    assert fitter._code_generator.constants == ('c.grid__dummy__v1', 'c.grid__rdummy__c')


def test_param_overrides(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    fitter = starlord.ModelBuilder()
    fitter.constraint("d.dummy.v1", "normal", [3., 1.])
    fitter.override_input("dummy", "y", "5.0 + c.fixed_y")
    fitter.prior("p.x", "normal", [2., 5.])
    # Running to resolve the grids
    print(fitter.summary())
    assert fitter._code_generator is not None
    assert fitter._code_generator.params == ('p.x',)
    assert fitter._code_generator.locals == ('l.dummy__v1',)
    assert fitter._code_generator.constants == ('c.fixed_y', 'c.grid__dummy__v1')
    code = fitter.generate_code()
    assert "5.0 + self.c_fixed_y" in code
    assert "params[1]" not in code
