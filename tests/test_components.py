from pytest import approx, raises

from starlord import code_components


def test_symbols():
    s = code_components.Symb("p.something")
    assert s.label == "p"
    assert s.name == "something"
    assert s.var == "p_something"
    with raises(ValueError):
        code_components.Symb("asdf")
    with raises(ValueError):
        code_components.Symb("p_stuff")
    s = code_components.Symb("b.foo")
    assert s.label == "b"
    assert s.name == "foo"
    assert s.var == "b_foo"
