import sys

import pytest
# flake8: noqa
from test_grids import dummy_grids

import starlord
from starlord import cli
from starlord._config import config


def test_hashing(tmpdir):
    test_file = tmpdir.join("test_file.txt")
    test_file.write("This is a checksum test file.")
    expect = "55955d0e48724a1981ba25cde18375e6"
    assert starlord.io._hash_file(test_file, "md5") == expect


def test_classification(dummy_grids):
    # Note: the "posterior" case is handled in test_posterior_handling
    assert starlord.io.classify_file(dummy_grids / "dummy.npz") == "grid"
    assert starlord.io.classify_file(dummy_grids / "filter_test.txt") == "unknown"


def test_posterior_handling(tmpdir, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    builder = starlord.ModelBuilder()
    builder.assign("v.ratio", "p.foo/p.bar")
    builder.constraint("v.ratio", "normal", [1.0, 0.1])
    builder.prior("p.foo", "uniform", [0., 5.])
    builder.prior("p.bar", "uniform", [1., 2.])
    sampler = builder.build_sampler("builtin")
    sampler.run()
    outfile = str(tmpdir.join("output_test.npz"))
    sampler.save_results(outfile)
    assert starlord.io.classify_file(outfile) == "posterior"

    try:
        import corner
        import matplotlib
        outcorner = tmpdir.join("output.png")
        sampler.save_corner(str(outcorner))
        assert outcorner.exists()
    except ImportError:
        pass

    monkeypatch.setattr(sys, 'argv', ['starlord', outfile, '-p'])
    cli.main()
    captured = capsys.readouterr()
    assert captured.out.startswith("Posterior file with contents:")
    assert "Results Summary:" in captured.out
    assert "   0 bar" in captured.out
    assert "   1 foo" in captured.out
    assert "   2 log_like" in captured.out
    assert "   3 log_prior" in captured.out


def test_downloader(dummy_grids, mocker, capsys: pytest.CaptureFixture):
    config.grid_dir = dummy_grids
    starlord.GridGenerator.reload_grids()

    # Don't actually bother Zenodo servers with the test, use this class as a substitute
    class MockRequestGet:

        def __init__(self, url, **args):
            self.url = url
            self.status_code = 200
            self.headers = dict()
            self.raise_for_status = lambda *args: None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return

        def json(self):
            '''Pretends to be Zenodo metadata.'''
            foo = dict(key='foo', size=35.2, checksum='md5:asdf', links={'self': 'sdfg'})
            return {'files': [foo]}

        def iter_content(*args):
            '''Pretend (badly) to be a grid to be downloaded.'''
            return [b"Definitely a real grid, totally.", b" Second chunk."]

    mocker.patch("requests.get", MockRequestGet)

    # Local file will be overwritten on download, then deleted because checksum doesn't match
    localfile = dummy_grids / "foo"
    with open(localfile, 'w') as f:
        f.write("asdf")

    starlord.io.download_grids()
    starlord.io.download_grids("foo")
    captured = capsys.readouterr()

    assert captured.out.startswith("Grids available for download:")
    assert "foo" in captured.out
    assert "mist2" not in captured.out
    assert "Downloading foo" in captured.out
    # Hash was made up, so it should reject the "downloaded" file and delete it
    assert "downloaded file did not yield expected hash" in captured.out
    assert not localfile.exists()
