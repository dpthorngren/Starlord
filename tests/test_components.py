from pytest import raises

from starlord import code_components


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
    assert p.generate_pdf() == "logP += normal_lpdf(math.log10({p__foo}), 1.0, 0.1) + -math.log(10)-({p__foo})"
    assert p.generate_ppf() == "{p__foo} = 10**(normal_ppf({p__foo}, 1.0, 0.1))"
    p = code_components.Prior.create('p.foo', 'logit_beta', [31.0, 15.1])
    assert p.display() == "Logit_Beta(p.foo | 31.0, 15.1)"
    assert p.generate_pdf() == "logP += beta_lpdf(logit({p__foo}), 31.0, 15.1) + logddx_logit({p__foo})"
    assert p.generate_ppf() == "{p__foo} = expit(beta_ppf({p__foo}, 31.0, 15.1))"
