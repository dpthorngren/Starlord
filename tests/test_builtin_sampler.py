import numpy as np
import pytest
from pytest import approx

import starlord
from starlord.samplers import SamplerBuiltin


@pytest.mark.flaky(reruns=3)
def test_initial_state_generator():
    builder = starlord.ModelBuilder()
    builder.assign("v.asquared", "p.a*p.a")
    builder.constraint("v.asquared", "normal", [2.5, 0.1])
    builder.assign("v.ratio", "p.a * p.b")
    builder.constraint("v.ratio", "normal", [2.0, 0.3])
    builder.prior("p.a", "uniform", [0.0, 6.0])
    builder.prior("p.b", "uniform", [0.0, 10])

    sampler = builder.build_sampler("builtin", n_walkers=10)
    assert type(sampler) is SamplerBuiltin
    state = sampler.model.generate_initial_state(100, 200)
    assert np.all(np.isfinite(state))
    assert np.all(state[:, 0] > 0.0)
    assert np.all(state[:, 0] < 6.0)
    assert np.all(state[:, 1] > .001)
    assert np.all(state[:, 1] < 10.)
    assert np.mean(state, axis=0) == approx([np.sqrt(2.5), 2. / np.sqrt(2.5)], abs=0.25)


@pytest.mark.flaky(reruns=3)
def test_builtin_run():
    builder = starlord.ModelBuilder()
    builder.assign("v.sina", "math.sin(p.a)")
    builder.constraint("v.sina", "normal", [0.5, 0.1])
    builder.assign("v.ratio", "p.a / p.b")
    builder.constraint("v.ratio", "normal", ["c.ratio", 0.3])
    builder.prior("p.a", "uniform", [0., 3.14159 / 2.])
    builder.prior("p.b", "uniform", [0.001, 10])

    # Build the sampler and check the resulting metadata
    sampler = builder.build_sampler("builtin", {'ratio': 3.5}, n_walkers=10)
    assert type(sampler) is SamplerBuiltin
    assert sampler.model.param_names == ['a', 'b']
    assert sampler.model.param_names == sampler.param_names
    assert sampler.model.var_names == ['ratio', 'sina']

    # Run and check outputs against approximate answer
    # Ensure this works for pure metropolis, pure stretch steps, and a mixture
    for metropolis_frac in [0.0, 0.2, 1.0]:
        sampler.run(n_samples=4000, burn_in=100, thin=10, metropolis_frac=metropolis_frac)
        expect = np.arcsin(.5)
        assert sampler.stats.mean[:2] == approx([expect, expect / 3.5], rel=0.15)
        assert np.all(np.corrcoef(sampler.post[:, :2].T) > 0.7)
        # Priors are uniform, should have the same value everywhere
        assert sampler.stats.std[-1] == approx(0., abs=1e-9)
        # Check that acceptances make sense
        if metropolis_frac == 0:
            assert np.isnan(sampler.sampler.get_acceptance()[0])
            assert sampler.sampler.get_acceptance()[1] > 0.
        else:
            assert sampler.sampler.get_acceptance()[0] > 0.
            if metropolis_frac < 1:
                assert sampler.sampler.get_acceptance()[1] > 0.
            else:
                assert np.isnan(sampler.sampler.get_acceptance()[1])
