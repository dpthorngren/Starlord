import argparse
import pathlib
import sys

from ._version import __version__
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
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # === Load Settings ===
    # Default initial settings (keep minimal)
    settings = {'output': {'terminal': True, 'file': ""}, "run": {}}
    if args.verbose:
        print("Args:", args)
        print("Settings: ", settings)

    if args.input is not None:
        if args.verbose:
            print(args.input)
        with open(args.input, 'rb') as f:
            settings.update(tomllib.load(f))
    # Handle syntax errors in the toml file

    # Report ignored sections

    # Update settings with command line arguments

    # === Setup the fitter ===
    if "model" in settings.keys():
        fitter = StarFitter(args.verbose)
        fitter.set_from_dict(settings['model'])
    else:
        print("No model information was specified.")
        return
    if args.code:
        print(fitter.generate_log_like())
    if args.dry_run:
        fitter.summary(args.verbose)
        return

    # === Run Sampler ==
    # Use settings.sampling for config, contents depends on sampler type
    results = fitter.run_sampler(settings['sampling'])

    # === Write Outputs ===
    out: dict = {"terminal": False, "file": ""}
    out.update(settings['output'])
    if out['terminal']:
        print("TODO: print results to terminal.")
        # results.summary()
    if out['file'] != "":
        print("TODO: write results to ", out['file'])
        # fitter.write_results()
