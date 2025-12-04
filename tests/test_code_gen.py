import math
import os

from pytest import approx

from starlord import CodeGenerator
from starlord._config import _load_config, config


def test_expressions():
    g = CodeGenerator()
    g.expression("l.foo = np.sin(p.stuff)")
    assert len(g._like_components) == 1
    comp = g._like_components[0]
    # Check variable processing
    assert comp.requires == {"p.stuff"}
    assert comp.provides == {"l.foo"}
    assert comp.code.count("np.") == 1
    assert comp.code.count("l_foo") == 0
    assert comp.code.count("l.foo") == 1
    assert comp.code.count("p_stuff") == 0
    assert comp.code.count("p.stuff") == 1
    # Check variable aggregation
    assert g.variables == {"l.foo", "p.stuff"}
    assert g.params == ("p.stuff",)
    assert g.locals == ("l.foo",)
    assert g.constants == ()
    # Check summary function
    s = g.summary().splitlines()
    assert s[1].startswith("Params:")
    assert "stuff" in s[1]
    assert s[2].startswith("Locals:")
    assert "foo" in s[2]
    # No prior was specified
    assert "Prior" in s[-1]


def test_compilation():
    code = "from libc cimport math\n\n"
    code += "cpdef double testFunction(double x):\n"
    code += "    return 3.5 * math.sin(x/2.)\n"
    hash = CodeGenerator._compile_to_module(code)
    mod = CodeGenerator._load_module(hash)
    assert mod.testFunction(12.) == approx(3.5 * math.sin(12. / 2.))


def test_config():
    _load_config()
    assert config.system in ["Windows", "Linux", "Darwin"]
    assert os.path.exists(config.cache_dir)
    assert os.path.exists(config.grid_dir)
