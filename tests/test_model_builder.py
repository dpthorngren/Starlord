import re
from pathlib import Path

from pytest import raises
# flake8: noqa
from test_grids import dummy_grids

import starlord
from starlord._config import config
from starlord.model_builder import DeferredResolver


def test_builder_regex():
    match = starlord.ModelBuilder.overridable_regex.fullmatch("foo.bar--3")
    assert match is None
    match = starlord.ModelBuilder.overridable_regex.fullmatch("foo__bar--3")
    assert match is not None
    assert match.groups() == ("foo", "bar", "3")
    match = starlord.ModelBuilder.overridable_regex.fullmatch("long_var_name")
    assert match is not None
    assert match.groups() == (None, "long_var_name", None)
    match = starlord.ModelBuilder.varname_regex.fullmatch("p.foo")
    assert match is not None
    match = starlord.ModelBuilder.varname_regex.fullmatch("c.stuff_things")
    assert match is not None
    match = starlord.ModelBuilder.varname_regex.fullmatch("c__stuff_things")
    assert match is None
    assert starlord.ModelBuilder.is_valid_param("p.asdf_dh--3")
    assert starlord.ModelBuilder.is_valid_param("l.asdf__dh__3")
    assert starlord.ModelBuilder.is_valid_param("c.grid__made_up__var--blend")


def test_deferred_regex():
    # Input processing regex
    for s in ["cd.asdf", "c.invalid--234", "np.sin(p.foo--3)"]:
        assert re.match(DeferredResolver.find_input_deferred, s) is None
    s = "np.sin(d.foo.bar--3 + d.things + p.ignore) / c.stuff"
    matches = re.findall(DeferredResolver.find_input_deferred, s)
    assert len(matches) == 2
    assert matches[0] == ("foo", "bar", "3")
    assert matches[1] == ("", "things", "")
    match = DeferredResolver.find_input_deferred.search("-3*+d.grid.things_1-5")
    assert match is not None
    assert match.groups() == ("grid", "things_1", None)
    match = DeferredResolver.find_input_deferred.search("3.5/np.sin(d.grid.an_aggregate35--blend)--3")
    assert match is not None
    assert match.groups() == ("grid", "an_aggregate35", "blend")

    # Key processing regex
    match = DeferredResolver.find_keys_deferred.match("{grid__foo}")
    assert match is not None
    assert match.groups() == ("grid", "foo", None)
    match = DeferredResolver.find_keys_deferred.match("{grid.foo}")
    assert match is None
    match = DeferredResolver.find_keys_deferred.match("{name--43}")
    assert match is not None
    assert match.groups() == (None, "name", "43")
    matches = DeferredResolver.find_keys_deferred.findall("{gridname__col--i} + np.sin({var--3})")
    assert len(matches) == 2
    assert matches[0] == ("gridname", "col", "i")
    assert matches[1] == ("", "var", "3")

    # Variable index extraction regex
    match = DeferredResolver.find_indexed_vars.match("p.foo--3")
    assert match is not None
    assert match.groups() == ("p", "foo", "3")
    match = DeferredResolver.find_indexed_vars.match("l.things--i")
    assert match is not None
    assert match.groups() == ("l", "things", "i")
    matches = DeferredResolver.find_indexed_vars.findall("c.something--1 - c.stuff--mean")
    assert len(matches) == 2
    assert matches[0] == ("c", "something", "1")
    assert matches[1] == ("c", "stuff", "mean")


def test_deferred_handling(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    src = "l.foo = d.ten**d.dummy.v1 + d.dummy.x + p.dont_catch + c.ditto"
    var, _ = DeferredResolver.extract_deferred(src)
    assert var == ["ten", "dummy__v1", "dummy__x"]
    src = "d.fifty + d.dummy.v1--i-d.rdummy.d--1"
    var, out = DeferredResolver.extract_deferred(src, index="3")
    assert out == "{fifty} + {dummy__v1--3}-{rdummy__d--1}"
    assert var == ["fifty", "dummy__v1--3", "rdummy__d--1"]


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


def test_deferred_resolver(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    user_map = {'foo': 'd.rdummy.d', "dummy.x": "p.x_modified", "dummy.y": "p.y_modified"}
    resolver = DeferredResolver(user_map, verbose=True)
    resolver.resolve_all(set(["foo"]))
    assert resolver.def_map['dummy__x'] == 'p.x_modified'
    assert resolver.def_map['dummy__x'] == 'p.x_modified'
    assert resolver.def_map['dummy__y'] == 'p.y_modified'
    assert resolver.def_map['dummy__v1'] == 'l.dummy__v1'
    assert resolver.def_map['dummy__g1'] == 'l.dummy__g1'
    assert resolver.def_map['rdummy__a'] == 'l.dummy__g1'
    assert resolver.def_map['rdummy__b'] == 'p.b'
    assert resolver.def_map['rdummy__c'] == 'l.rdummy__c'
    assert resolver.def_map['rdummy__d'] == 'l.rdummy__d'
    assert resolver.def_map['foo'] == 'l.rdummy__d'
    assert len(resolver.new_components) == 4


def test_deferred_multiple(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    vars, source = DeferredResolver.extract_deferred("d.rdummy.d--1")
    assert vars == ["rdummy__d--1"]
    assert source == "{rdummy__d--1}"
    # Multiple-grid resolution
    user_map = {'foo': '10*d.rdummy.d--1-3', 'bar': '3**d.rdummy.d--2-2'}
    resolver = DeferredResolver(user_map, verbose=True)
    resolver.resolve_all(set(["foo", "bar"]))
    assert "i" not in resolver.def_map.values()
    assert resolver.def_map["dummy__v1--1"] == "l.dummy__v1__1"
    assert resolver.def_map["dummy__g1--1"] == "l.dummy__g1__1"
    assert resolver.def_map["rdummy__d--1"] == "l.rdummy__d__1"
    assert resolver.def_map["rdummy__c--1"] == "l.rdummy__c__1"
    # X should be mapped by index, y should be a common variable
    assert resolver.def_map["dummy__x--1"] == "p.x__1"
    assert resolver.def_map["dummy__x--2"] == "p.x__2"
    for k in resolver.def_map.keys():
        if k.startswith("dummy__y"):
            assert resolver.def_map[k] == "p.y"


def test_deferred_composites(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    multi = dict(rdummy=2)
    resolver = DeferredResolver({}, multi, verbose=True)
    resolver.resolve_all(set(['d.rdummy.d--mean', 'd.rdummy.d--sum', 'd.rdummy.d--blend']))
    assert "p.y__1" not in resolver.def_map.values()
    assert "p.x__1" in resolver.def_map.values()
    assert resolver.def_map["rdummy__d--blend"] == "l.rdummy__d__blend"
    assert resolver.def_map["rdummy__d--mean"] == "l.rdummy__d__mean"
    assert resolver.def_map["dummy__x--1"] == "p.x__1"
    assert resolver.def_map["dummy__x--2"] == "p.x__2"
    assert resolver.def_map["dummy__y--1"] == "p.y"
    assert resolver.def_map["dummy__y--2"] == "p.y"
    for c in resolver.new_components:
        print(c)
    expected = ("rdummy", "1", "d", "math.exp(l.rdummy__c__1)")
    assert resolver.new_components.count(expected) == 1
    expected = ("rdummy", "mean", "d", "(l.rdummy__d__1 + l.rdummy__d__2) / 2")
    assert expected in resolver.new_components
    expected = ("rdummy", "sum", "d", "l.rdummy__d__1 + l.rdummy__d__2")
    assert expected in resolver.new_components
    expected = ("rdummy", "blend", "d", "-2.5*math.log10(10**(-l.rdummy__d__1/2.5) + 10**(-l.rdummy__d__2/2.5))")
    assert expected in resolver.new_components


def test_multiple_model(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    fitter = starlord.ModelBuilder(True, True)
    fitter.constraint("d.dummy.v1--mean", "normal", [.75, .1])
    fitter.constraint("d.dummy.x--sum", "normal", [0, .5])
    fitter.prior("p.x--1", "uniform", [-5, 0])
    fitter.prior("p.x--2", "uniform", [0, 5])
    fitter.prior("p.y", "uniform", [0.1, 10.])
    with raises(AssertionError):
        fitter.summary()
    fitter.multiplicity['dummy'] = 2
    print(fitter.summary())
    assert fitter.code_generator.params == ('p.x__1', 'p.x__2', 'p.y')
    assert fitter.code_generator.constants == ('c.grid__dummy__v1',)
    expected = ('l.dummy__v1__1', 'l.dummy__v1__2', 'l.dummy__v1__mean', 'l.dummy__x__sum')
    assert fitter.code_generator.locals == expected


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
    assert fitter.code_generator is not None
    assert fitter.code_generator.params == ('p.x',)
    assert fitter.code_generator.locals == ('l.dummy__v1',)
    assert fitter.code_generator.constants == ('c.fixed_y', 'c.grid__dummy__v1')
    code = fitter.generate_code()
    assert "5.0 + self.c__fixed_y" in code
    assert "params[1]" not in code
