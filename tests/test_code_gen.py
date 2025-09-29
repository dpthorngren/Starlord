from starlord import CodeGenerator
import re
import pytest

def test_expressions():
    g = CodeGenerator()
    g.expression("l.foo = np.sin(p.stuff)")
    assert len(g._like_components) == 1
    comp = g._like_components[0]
    # Check variable processing
    assert comp.requires == {"p_stuff"}
    assert comp.provides == {"l_foo"}
    assert "".count(comp.code) == 0
    assert comp.code.count("np.") == 1
    assert comp.code.count("l.foo") == 0
    assert comp.code.count("l_foo") == 1
    assert comp.code.count("p.stuff") == 0
    assert comp.code.count("p_stuff") == 1
    # Check variable aggregation
    assert g.variables == {"l_foo", "p_stuff"}
    assert g.params == ["p_stuff"]
    assert g.locals == ["l_foo"]
    assert g.constants == []
    assert g.arrays == []
