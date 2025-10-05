import starlord
import re

from starlord import cli
import sys

import pytest


def test_dryrun(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(sys, 'argv', ['starlord', 'tests/low_level.toml', '--dry-run', '--code'])
    cli.main()
    captured = capsys.readouterr()
    # Basic code outputs
    assert "from starlord.cy_tools cimport *\n" in captured.out
    assert "\ncpdef double[:] prior_transform(double[:] params):\n" in captured.out
    # Key terms in the output present?
    assert "\n    logL += normal_lpdf(l_A, 0.5, 0.25)\n" in captured.out
    assert "\n    params[0] = normal_ppf(params[0], -5.0, 5.0)" in captured.out
    assert "\n    l_A = math.exp(params[0])" in captured.out
    # Summary was printed?
    assert "\n=== Variables ===\n" in captured.out
    # Check that the params match expectations
    paramSummary = re.search(r"^Params:\s+(.*)$", captured.out, flags=re.M)
    assert paramSummary is not None
    params = list(map(str.strip, paramSummary.group(1).split(",")))
    assert params == ['a', 'b']
    localSummary = re.search(r"^Locals:\s+(.*)$", captured.out, flags=re.M)
    assert localSummary is not None
    locals = list(map(str.strip, localSummary.group(1).split(",")))
    assert locals == ['A', 'B']

