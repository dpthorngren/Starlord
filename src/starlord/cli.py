import argparse
import pathlib
import re
import sys

from . import __version__
from ._config import config
from .grid_gen import GridGenerator
from .model_builder import ModelBuilder

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def main():
    parser = argparse.ArgumentParser(
        "starlord", description="Fit stellar observations with starlord from the command line.")
    parser.add_argument(
        "input", type=pathlib.Path, nargs="?", default=None, help="A toml file to load run settings from (optional)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print a model summary and exit without running the sampler.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print extra debugging information.")
    parser.add_argument("-p", "--plain-text", action="store_true", help="Do not use ANSI codes for terminal output.")
    parser.add_argument(
        "-s", "--set-const", action="append", default=[], help="Set a model constant, e.g. '-s a=3'; repeatable.")
    parser.add_argument("-c", "--code", action="store_true", help="Print code upon generation.")
    parser.add_argument("--version", action="version", version=f"starlord {__version__}")
    parser.add_argument("-l", "--list-grids", action="store_true", help="List all grids available to Starlord.")
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    txt = config.text_format_off if args.plain_text else config.text_format

    if args.verbose:
        print(f"    {txt.underline}CLI Arguments{txt.end}")
        print(args, end='\n\n')

    if args.list_grids:
        if args.input is not None:
            grid_name = str(args.input)
            assert grid_name in GridGenerator.grids(), f"Grid {grid_name} not found."
            g = GridGenerator.get_grid(grid_name)
            g.summary(True, fancy_text=~args.plain_text)
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
        with open(args.input, 'rb') as f:
            settings.update(tomllib.load(f))
    # TODO: Handle syntax errors in the toml file

    # Report ignored sections
    for section in settings.keys():
        if section not in ['model', 'sampling', 'output']:
            print(f"Warning, section {section} in input file {args.input} is not used.")

    # TODO: Update settings with command line arguments

    if args.verbose:
        print(f"    {txt.underline}Settings{txt.end}")
        for key, value in settings.items():
            print(key, value, sep=": ")
        print("")

    # === Setup the fitter ===
    assert "model" in settings.keys(), "No model information was specified."
    fitter = ModelBuilder(args.verbose, not args.plain_text)
    fitter.set_from_dict(settings['model'])
    if args.code:
        code = fitter.generate()
        if not args.plain_text:
            code = re.sub(r"(?<!\w)(l_[a-zA-z]\w*)", f"{txt.bold}{txt.green}\\g<1>{txt.end}", code, flags=re.M)
            code = re.sub(r"(?<!\w)(c_[a-zA-z]\w*)", f"{txt.bold}{txt.blue}\\g<1>{txt.end}", code, flags=re.M)
            code = re.sub(r"(?<!\w)(params\[\d+\])", f"{txt.bold}{txt.yellow}\\g<1>{txt.end}", code, flags=re.M)
        print(code)
    if args.dry_run:
        print(fitter.summary())

    # Assemble the constants dictionary
    # TODO: Check constants for validity
    consts = settings['sampling'].get('const', {})
    for const_str in args.set_const:
        key, value = const_str.split("=")
        if key.startswith("c."):
            key = key[2:]
        consts[key] = float(value)
    if args.dry_run and consts:
        # Note: Constants also printed during non-dry-run by StarFitter.run_sampler().
        print(f"\n    {txt.underline}Constant Values{txt.end}")
        for k, v in consts.items():
            print(f"{txt.blue}{txt.bold}c.{k}{txt.end} = {txt.blue}{v:.4n}{txt.end}")
    if args.dry_run:
        return

    # === Run Sampler ==
    # TODO: Get sampler settings
    results = fitter.run_sampler({}, constants=consts)

    # === Write Outputs ===
    out: dict = {"terminal": False, "file": ""}
    out.update(settings['output'])
    if out['terminal']:
        print(results.summary())
    if out['file'] != "":
        print("TODO: write results to ", out['file'])
        # fitter.write_results()
