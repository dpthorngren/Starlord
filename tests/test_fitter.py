import re
from pathlib import Path

import pytest
# flake8: noqa
from test_grids import dummy_grid

import starlord


def test_starfitter_variables():
    fitter = starlord.StarFitter()
    assert fitter is not None
    fitter.assign("x", "2.643")
    fitter.expression("logL += p.foo*p.foo / 10.")
    assert fitter._gen.params == ("p.foo",)
    assert fitter._gen.locals == ("l.x",)
    fitter.assign("b.something", "3.5*(p.foo - c.bar)")


@pytest.mark.flaky(reruns=1)
def test_grid_retrieval(dummy_grid: Path, capsys: pytest.CaptureFixture):
    fitter = starlord.StarFitter(True)
    # fitter.assign("foo", "dummy.v1 + c.offset")
    fitter.expression("l.foo = dummy.v1 + c.offset")
    fitter.constraint("l.foo", "normal", [0.5, 0.1])
    fitter.prior("x", "uniform", [-5., 5.])
    fitter.prior("y", "uniform", [0.1, 10.0])
    # Test that the summaries were reasonable
    fitter.summary()
    fitter.summary(True)
    captured = capsys.readouterr()
    assert "\n=== Variables ===\n" in captured.out
    # Check parameters
    paramSummary = re.search(r"^Params:\s+(.*)$", captured.out, flags=re.M)
    assert paramSummary is not None
    params = list(map(str.strip, paramSummary.group(1).split(",")))
    assert params == ['x', 'y']
    localSummary = re.search(r"^Locals:\s+(.*)$", captured.out, flags=re.M)
    assert localSummary is not None
    locals = list(map(str.strip, localSummary.group(1).split(",")))
    assert locals == ['dummy_v1', 'foo']
    # Check that the summary prints properly (largely formats result.stats())
    # results = fitter.run_sampler({}, {'offset': 1.5})
    # summary = results.summary().splitlines()
    # assert len(summary) == 3
    # assert summary[0].startswith(" Dim")
    # assert summary[1].startswith("   0")
    # assert summary[2].startswith("   1")
    # Check against known mean, std
    # stats = results.stats()
    # # Normal distribution
    # s = 1. / (1. / 2.**2 + 1. / 10.**2)
    # assert stats[0, 0] == pytest.approx(-1.5 + s * (5. / 2.**2 + 0. / 10.**2), rel=.05)
    # assert stats[0, 1]**2 == pytest.approx(s, rel=.1)
    # # Beta distribution
    # assert stats[1, 0] == pytest.approx(15. / (15+25.), rel=.05)
    # assert stats[1, 1]**2 == pytest.approx(15. * 25. / ((15 + 25)**2 * (15.+25.+1.)), rel=.1)


@pytest.mark.flaky(reruns=1)
def test_retrieval(capsys: pytest.CaptureFixture):
    fitter = starlord.StarFitter(True)
    fitter.assign("blah", "p.foo")
    fitter.constraint("l.blah", "beta", [15., 25])
    fitter.prior("foo", "uniform", [0., 1.])
    fitter.expression("l.stuff = p.bar + c.offset")
    fitter.constraint("l.stuff", "normal", [5., 2])
    fitter.prior("bar", "normal", [0., 10.])
    # Test that the summaries were reasonable
    fitter.summary()
    fitter.summary(True)
    captured = capsys.readouterr()
    assert "\n=== Variables ===\n" in captured.out
    # Check parameters
    paramSummary = re.search(r"^Params:\s+(.*)$", captured.out, flags=re.M)
    assert paramSummary is not None
    params = list(map(str.strip, paramSummary.group(1).split(",")))
    assert params == ['bar', 'foo']
    localSummary = re.search(r"^Locals:\s+(.*)$", captured.out, flags=re.M)
    assert localSummary is not None
    locals = list(map(str.strip, localSummary.group(1).split(",")))
    assert locals == ['blah', 'stuff']
    # Check that the summary prints properly (largely formats result.stats())
    results = fitter.run_sampler({}, {'offset': 1.5})
    summary = results.summary().splitlines()
    assert len(summary) == 3
    assert summary[0].startswith(" Dim")
    assert summary[1].startswith("   0")
    assert summary[2].startswith("   1")
    # Check against known mean, std
    stats = results.stats()
    # Normal distribution
    s = 1. / (1. / 2.**2 + 1. / 10.**2)
    assert stats[0, 0] == pytest.approx(-1.5 + s * (5. / 2.**2 + 0. / 10.**2), rel=.05)
    assert stats[0, 1]**2 == pytest.approx(s, rel=.1)
    # Beta distribution
    assert stats[1, 0] == pytest.approx(15. / (15+25.), rel=.05)
    assert stats[1, 1]**2 == pytest.approx(15. * 25. / ((15 + 25)**2 * (15.+25.+1.)), rel=.1)
