import numpy as np
from pytest import approx, raises
from scipy.stats import lognorm

from starlord import ModelBuilder, code_components, cy_tools


def test_symbols():
    s = code_components.Symb("p.something")
    assert s.label == "p"
    assert s.name == "something"
    assert s.var == "p__something"
    with raises(ValueError):
        code_components.Symb("asdf")
    with raises(ValueError):
        code_components.Symb("p_stuff")
    s = code_components.Symb("c.foo")
    assert s.label == "c"
    assert s.name == "foo"
    assert s.var == "c__foo"


def test_dist_prefixes():
    p = code_components.Prior.create('p.foo', 'log_normal', [1.0, 0.1])
    assert p.display() == "Log_Normal(p.foo | 1.0, 0.1)"
    assert p.generate_pdf() == \
        "logP += normal_lpdf(math.log10({p__foo}), 1.0, 0.1) + -math.log(math.log(10)) - math.log({p__foo})"
    assert p.generate_ppf() == "{p__foo} = 10**(normal_ppf({p__foo}, 1.0, 0.1))"
    p = code_components.Prior.create('p.foo', 'logit_beta', [31.0, 15.1])
    assert p.display() == "Logit_Beta(p.foo | 31.0, 15.1)"
    assert p.generate_pdf() == "logP += beta_lpdf(logit({p__foo}), 31.0, 15.1) + logddx_logit({p__foo})"
    assert p.generate_ppf() == "{p__foo} = expit(beta_ppf({p__foo}, 31.0, 15.1))"

    # Test prefix code directly, starting with natural log and exp
    builder = ModelBuilder()
    builder.constraint("p.x", "expn_uniform", [1.0, 5.0])
    builder.prior("p.x", "ln_normal", [-0.3, 0.01])
    model = builder.code_generator.compile().Model()
    for x in 5 * np.random.rand(1000, 1):
        assert model.log_prior(x) == approx(lognorm.logpdf(x[0], scale=np.exp(-.3), s=.01), rel=1e-12)
        jac = -np.log(x[0])
        assert model.log_prior(x) == approx(cy_tools.normal_lpdf(np.log(x[0]), -.3, .01) + jac, rel=1e-12)
        jac = x[0]
        assert model.log_like(x) == approx(cy_tools.uniform_lpdf(np.exp(x[0]), 1, 5.) + jac, rel=1e-12)

    # Now base 10 logs
    builder = ModelBuilder()
    builder.constraint("p.x", "exp10_normal", [-0.3, 0.01])
    builder.prior("p.x", "log_uniform", [-1.0, 5.0])
    model = builder.code_generator.compile().Model()
    for x in 5 * np.random.rand(1000, 1):
        jac = -np.log(x[0]) - np.log(np.log(10))
        assert model.log_prior(x) == approx(cy_tools.uniform_lpdf(np.log10(x[0]), -1, 5.) + jac, rel=1e-12)
        jac = np.log(np.log(10)) + np.log(10)*x[0]
        assert model.log_like(x) == approx(cy_tools.normal_lpdf(10**x[0], -.3, .01) + jac, rel=1e-12)

    # Now expit and logit
    builder = ModelBuilder()
    builder.constraint("p.x", "expit_beta", [21.3, 32.6])
    builder.prior("p.x", "logit_normal", [-1, 1])
    model = builder.code_generator.compile().Model()
    for x in np.random.rand(1000, 1):
        jac = cy_tools.logddx_logit(x[0])
        assert model.log_prior(x) == approx(cy_tools.normal_lpdf(cy_tools.logit(x[0]), -1, 1) + jac, rel=1e-12)
        jac = cy_tools.logddx_expit(x[0])
        assert model.log_like(x) == approx(cy_tools.beta_lpdf(cy_tools.expit(x[0]), 21.3, 32.6) + jac, rel=1e-12)
