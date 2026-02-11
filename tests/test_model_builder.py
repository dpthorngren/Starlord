from pathlib import Path

# flake8: noqa
from test_grids import dummy_grids

import starlord
from starlord._config import config
from starlord.model_builder import DeferredResolver


def test_deferred_handling(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    src = "{l.foo = d.ten**d.dummy.v1 + d.dummy.x + p.dont_catch + c.ditto"
    var, _ = DeferredResolver.extract_deferred(src)
    assert var == ["ten", "dummy__v1", "dummy__x"]


def test_model_builder_variables():
    fitter = starlord.ModelBuilder()
    assert fitter is not None
    fitter.assign("x", "2.643")
    fitter.expression("logL += p.foo*p.foo / 10.")
    fitter.summary()
    assert fitter.code_generator is not None
    assert fitter.code_generator.params == ("p.foo",)
    assert fitter.code_generator.locals == ("l.x",)
    fitter.assign("something", "3.5*(p.foo - c.bar)")


def test_recursive_grids(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    fitter = starlord.ModelBuilder()
    fitter.constraint("d.rdummy.d", "normal", [10., 1.])
    # Running to resolve the grids
    print(fitter.summary())
    assert fitter.code_generator is not None
    assert fitter.code_generator.params == ('p.b', 'p.x', 'p.y')
    assert fitter.code_generator.locals == ('l.dummy__g1', 'l.dummy__v1', 'l.rdummy__c', 'l.rdummy__d')
    assert fitter.code_generator.constants == ('c.grid__dummy__v1', 'c.grid__rdummy__c')


def test_param_overrides(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    fitter = starlord.ModelBuilder()
    fitter.constraint("d.dummy.v1", "normal", [3., 1.])
    fitter.override_mapping("dummy.y", "d.five + c.fixed_y")
    fitter.override_mapping("five", "5.0")
    fitter.prior("p.x", "normal", [2., 5.])
    # Check deferred variable mappings
    mappings = fitter._resolve_deferred()
    print(mappings)
    assert list(mappings.keys()) == ['dummy__x', 'five', 'dummy__y', 'dummy__v1']
    assert mappings["five"] == "5.0"
    assert mappings["dummy__y"] == "5.0 + c.fixed_y"
    assert mappings["dummy__v1"] == "l.dummy__v1"
    # Running to resolve the grids
    print(fitter.summary())
    assert fitter.code_generator is not None
    assert fitter.code_generator.params == ('p.x',)
    assert fitter.code_generator.locals == ('l.dummy__v1',)
    assert fitter.code_generator.constants == ('c.fixed_y', 'c.grid__dummy__v1')
    code = fitter.generate_code()
    assert "5.0 + self.c__fixed_y" in code
    assert "params[1]" not in code
