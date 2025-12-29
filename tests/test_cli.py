import re
import sys

import pytest
# flake8: noqa
from test_grids import dummy_grids

from starlord import GridGenerator, cli
from starlord._config import config


def test_grid_listing(dummy_grids, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(sys, 'argv', ['starlord', '--list-grids'])
    config.grid_dir = dummy_grids
    GridGenerator.reload_grids()
    cli.main()
    captured = capsys.readouterr()
    assert "dummy" in captured.out
    assert GridGenerator.get_grid("dummy").spec in captured.out


def test_grid_summary(dummy_grids, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(sys, 'argv', ['starlord', '--list-grids', 'dummy', '-p'])
    config.grid_dir = dummy_grids
    GridGenerator.reload_grids()
    cli.main()
    captured = capsys.readouterr()
    assert captured.out.startswith("Grid dummy")
    grid = GridGenerator.get_grid("dummy")
    for i, grid_input in enumerate(grid.inputs):
        assert f"  {i} {grid_input}" in captured.out
    for out in grid.outputs:
        assert out in captured.out


def test_dryrun(dummy_grids, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(
        sys, 'argv', ['starlord', 'tests/low_level.toml', '--dry-run', '-c', '-v', '-p', '-s', 'c.offset=-1.5'])
    config.grid_dir = dummy_grids
    GridGenerator.reload_grids()
    cli.main()
    captured = capsys.readouterr()
    assert "Warning, section erroneous in input file " in captured.out
    # Basic code outputs
    assert "from starlord.cy_tools cimport *\n" in captured.out
    assert "\n    cpdef double[:] prior_transform(self, double[:] params):\n" in captured.out
    # Key terms in the output present?
    assert "\n        logL += normal_lpdf(self.l_A, 0.5, 0.25)\n" in captured.out
    assert "\n        params[0] = normal_ppf(params[0], -5.0, 5.0)" in captured.out
    assert "\n        self.l_A = math.exp(params[0])" in captured.out
    # Summary was printed?
    assert "Variables" in captured.out
    # Check that the params match expectations
    paramSummary = re.search(r"^Params:\s+(.*)$", captured.out, flags=re.M)
    assert paramSummary is not None
    params = list(map(str.strip, paramSummary.group(1).split(",")))
    assert params == ['a', 'b', 'y']
    localSummary = re.search(r"^Locals:\s+(.*)$", captured.out, flags=re.M)
    assert localSummary is not None
    locals = list(map(str.strip, localSummary.group(1).split(",")))
    assert locals == ['A', 'B', 'dummy_v1', 'temp']
    constSummary = re.search(r"^Constants:\s+(.*)$", captured.out, flags=re.M)
    assert constSummary is not None
    consts = list(map(str.strip, constSummary.group(1).split(",")))
    assert consts == ['B_mean', 'grid_dummy_v1', 'offset']
    # Listed constants prefer CLI to model file?
    assert "Constant Values" in captured.out
    assert "c.offset = -1.5" in captured.out
    assert "c.B_mean = 2.5" in captured.out
