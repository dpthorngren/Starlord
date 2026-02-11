import re
from pathlib import Path

import numpy as np
import pytest
# flake8: noqa
from test_grids import dummy_grids

import starlord
from starlord._config import config


@pytest.mark.flaky(reruns=3)
def test_grid_retrieval(dummy_grids: Path):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()
    builder = starlord.ModelBuilder(True)
    builder.assign("foo", "d.dummy.v1 + c.offset")
    builder.constraint("l.foo", "normal", [0.5, 0.1])
    builder.constraint("d.dummy.g2", "normal", [0.5, 2.3])
    builder.constraint("d.dummy.v2", "normal", [1.5, 1.3])
    builder.prior("x", "uniform", [-5., 5.])
    builder.prior("y", "uniform", [0.1, 10.0])
    print(builder.summary())

    # Check parameters
    code = builder.generate_code()
    print(code)
    assert len(re.findall(r"l__dummy__v1 = ", code)) == 1
    assert builder.code_generator.locals == ('l.dummy__g1', 'l.dummy__g2', 'l.dummy__v1', 'l.dummy__v2', 'l.foo')
    assert builder.code_generator.constants == ('c.grid__dummy__v1', 'c.grid__dummy__v2', 'c.offset')
    sampler = builder.build_sampler("emcee", {'offset': 1.5})

    # Check the forward model works as expected (see test_grids.dummy_grids)
    out = sampler.model.forward_model(np.array([1.5, 4.5]))
    print(out)
    v1 = np.sin(1.5) + 4.5
    assert out['dummy__v1'] == pytest.approx(v1, rel=.01)
    v2 = 25. + np.cos(2.2*1.5) / np.sin(4.5)
    assert out['dummy__v2'] == pytest.approx(v2, rel=.01)
    g1 = 2.5*(5+1.5) + v1
    assert out['dummy__g1'] == pytest.approx(g1, rel=.01)
    g2 = 0.5 + np.log10(g1)
    assert out['dummy__g2'] == pytest.approx(g2, rel=.01)
    foo = v1 + 1.5
    assert out['foo'] == pytest.approx(foo, rel=.01)

    # Check that the results are reasonable
    sampler.run()
    stats = sampler.stats()
    assert np.all(np.isfinite(stats.cov))
    for s in [stats.mean, stats.p16, stats.p50, stats.p84]:
        assert -5. <= s[0] <= 5.
        assert 0.1 <= s[1] <= 10.
    assert 0. < stats.std[0] <= 10.
    assert 0 <= stats.std[1] <= 10.

    # Check output writing is working
    outfile = dummy_grids / "test_grid_retrieval_samples.npz"
    sampler.save_results(str(outfile))
    saved_data = np.load(outfile)
    assert "samples" in saved_data.files
    assert np.all(saved_data['samples'] == sampler.results)


@pytest.mark.flaky(reruns=3)
def test_retrieval(capsys: pytest.CaptureFixture):
    builder = starlord.ModelBuilder(True, False)
    builder.assign("blah", "p.foo")
    builder.constraint("l.blah", "beta", [15., 25])
    builder.prior("foo", "uniform", [0., 1.])
    builder.expression("l.stuff = p.bar + c.offset")
    builder.constraint("l.stuff", "normal", [5., 2])
    builder.prior("bar", "normal", [0., 10.])

    # Test that the summaries were reasonable
    print(builder.summary())
    captured = capsys.readouterr()
    assert "Variables" in captured.out

    # Check parameters
    paramSummary = re.search(r"^Params:\s+(.*)$", captured.out, flags=re.M)
    assert paramSummary is not None
    params = list(map(str.strip, paramSummary.group(1).split(",")))
    assert params == ['p.bar', 'p.foo']
    localSummary = re.search(r"^Locals:\s+(.*)$", captured.out, flags=re.M)
    assert localSummary is not None
    locals = list(map(str.strip, localSummary.group(1).split(",")))
    assert locals == ['l.blah', 'l.stuff']

    # Check that the summary prints properly (largely formats result.stats())
    sampler = builder.build_sampler("dynesty", {'offset': 1.5})
    sampler.run()
    summary = sampler.summary().splitlines()
    assert len(summary) == 3
    assert summary[0].startswith("     Name")
    assert summary[1].startswith("   0")
    assert summary[2].startswith("   1")

    # Check against known mean, std
    stats = sampler.stats()
    # Normal distribution
    s = 1. / (1. / 2.**2 + 1. / 10.**2)
    assert stats.mean[0] == pytest.approx(-1.5 + s * (5. / 2.**2 + 0. / 10.**2), rel=.05)
    assert stats.std[0]**2 == pytest.approx(s, rel=.1)
    # Beta distribution
    assert stats.mean[1] == pytest.approx(15. / (15+25.), rel=.05)
    assert stats.std[1]**2 == pytest.approx(15. * 25. / ((15 + 25)**2 * (15.+25.+1.)), rel=.1)
