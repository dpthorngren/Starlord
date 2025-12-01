import argparse
import pathlib
import sys

from . import __version__
from .grid_gen import GridGenerator
from .star_fitter import StarFitter

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def main():
    parser = argparse.ArgumentParser(
        "starlord", description="Fit stellar observations with starlord from the command line.")
    parser.add_argument(
        "input", type=pathlib.Path, nargs="?", default=None, help="A toml file to load run settings from (optional)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--code", action="store_true", help="Print code upon generation.")
    parser.add_argument("--version", action="version", version=f"starlord {__version__}")
    parser.add_argument("-l", "--list-grids", action="store_true")
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    if args.list_grids:
        if args.input is not None:
            grid_name = str(args.input)
            assert grid_name in GridGenerator.grids(), f"Grid {grid_name} not found."
            g = GridGenerator.get_grid(grid_name)
            print("Grid:", g.name)
            print("Inputs:", ", ".join(g.inputs))
            print("Outputs:")
            print(*["    " + i for i in g.outputs], sep="\n")
            if len(g.derived) > 0:
                print("Derived:")
                print(*["    " + d for d in g.derived.keys()], sep="\n")
            return
        print("Available grids:")
        for g in GridGenerator.grids().values():
            # Print short grid info, no need for "Grid_" prefix.
            print("   ", str(g)[5:])
        return

    # === Load Settings ===
    # Default initial settings (keep minimal)
    settings = {'output': {'terminal': True, 'file': ""}, "sampling": {}}

    if args.input is not None:
        if args.verbose:
            print(args.input)
        with open(args.input, 'rb') as f:
            settings.update(tomllib.load(f))
    # TODO: Handle syntax errors in the toml file

    # Report ignored sections
    for section in settings.keys():
        if section not in ['model', 'sampling', 'output']:
            print(f"Warning, section {section} in input file {args.input} is not used.")

    # TODO: Update settings with command line arguments

    if args.verbose:
        print("Args:", args)
        print("Settings: ", settings)

    # === Setup the fitter ===
    assert "model" in settings.keys(), "No model information was specified."
    fitter = StarFitter(args.verbose)
    fitter.set_from_dict(settings['model'])
    if args.code:
        # TODO: Set prior type based on sampler type
        print(fitter.generate())
    if args.dry_run:
        # TODO: Check constants
        fitter.summary(args.verbose)
        return

    # === Run Sampler ==
    consts = settings['sampling'].get('const', {})
    if args.verbose:
        print("Constants:", consts)
    results = fitter.run_sampler({}, constants=consts)

    # === Write Outputs ===
    out: dict = {"terminal": False, "file": ""}
    out.update(settings['output'])
    if out['terminal']:
        print(results.summary())
    if out['file'] != "":
        print("TODO: write results to ", out['file'])
        # fitter.write_results()
