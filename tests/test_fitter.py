import starlord

def test_starfitter_variables():
    # Not exactly a rigorous test yet
    fitter = starlord.StarFitter()
    assert fitter is not None
    fitter.assign("x", "2.643")
    fitter.expression("logL += p.foo*p.foo / 10.")
    assert fitter._gen.params == ("p.foo",)
    assert fitter._gen.locals == ("l.x",)
    fitter.assign("b.something", "3.5*(p.foo - c.bar)")
