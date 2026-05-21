import numpy as np
import pytest
from pytest import approx

import starlord


@pytest.mark.flaky(reruns=3)
def test_builtin_run():
    builder = starlord.ModelBuilder()
    builder.assign("l.sina", "math.sin(p.a)")
    builder.constraint("l.sina", "normal", [0.5, 0.1])
    builder.assign("l.ratio", "p.a / p.b")
    builder.constraint("l.ratio", "normal", ["c.ratio", 0.3])
    builder.prior("p.a", "uniform", [0., 3.14159 / 2.])
    builder.prior("p.b", "uniform", [0.001, 10])

    # Build the sampler and check the resulting metadata
    sampler = builder.build_sampler("builtin", {'ratio': 3.5}, n_walkers=10)
    assert sampler.model.param_names == ['a', 'b']
    assert sampler.model.param_names == sampler.param_names
    assert sampler.model.var_names == ['ratio', 'sina']

    # Run and check outputs against approximate answer
    sampler.run(n_samples=4000, burn_in=100, thin=10)
    expect = np.arcsin(.5)
    assert sampler.stats.mean[:2] == approx([expect, expect / 3.5], rel=0.15)
    assert np.all(np.corrcoef(sampler.post[:, :2].T) > 0.7)
    # Priors are uniform, should have the same value everywhere
    assert sampler.stats.std[-1] == approx(0., abs=1e-9)
