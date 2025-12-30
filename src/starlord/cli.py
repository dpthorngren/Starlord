import argparse
import pathlib
import re
import sys

import numpy as np

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
        "-g", "--grids", action="store_true", help="List available grids, or summarize a specific one, then exit.")
    parser.add_argument("--version", action="version", version=f"starlord {__version__}")
    model_group = parser.add_argument_group("model options", "Modify the model, overriding input file settings.")
    model_group.add_argument(
        "-s", "--set-const", action="append", default=[], help="Set a model constant, e.g. '-s a=3'; repeatable.")
    output_group = parser.add_argument_group("output options")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Print extra debugging information.")
    output_group.add_argument(
        "-p", "--plain-text", action="store_true", help="Do not use ANSI codes for terminal output.")
    output_group.add_argument(
        "-d", "--dry-run", action="store_true", help="Exit just before running the sampler (useful with -a)")
    output_group.add_argument("-c", "--code", action="store_true", help="Print code upon generation.")
    output_group.add_argument("-o", "--output", help="Set output file, overriding input file setting.")
    output_group.add_argument(
        "-a", "--analyze", "--analyse", action="store_true", help="Print analysis info for the model.")
    output_group.add_argument(
        "-t",
        "--test-case",
        help="Tests the forward model and likelihood at the given parameters (comma-separated, no spaces)")
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    txt = config.text_format_off if args.plain_text else config.text_format

    if args.verbose:
        print(f"    {txt.underline}CLI Arguments{txt.end}")
        print(args, end='\n\n')

    if args.grids:
        if args.input is not None:
            grid_name = str(args.input)
            assert grid_name in GridGenerator.grids(), f"Grid {grid_name} not found."
            g = GridGenerator.get_grid(grid_name)
            g.summary(True, fancy_text=not args.plain_text)
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

    # Update settings with command line arguments (TODO: More CLI options)
    if args.output:
        settings['output']['file'] = args.output
    consts = settings['sampling'].get('const', {})
    for const_str in args.set_const:
        key, value = const_str.split("=")
        if key.startswith("c."):
            key = key[2:]
        consts[key] = float(value)
    settings['sampling']['const'] = consts

    if args.verbose:
        print(f"    {txt.underline}Settings{txt.end}")
        for key, value in settings.items():
            print(key, value, sep=": ")
        print("")

    # === Setup the Model ===
    assert "model" in settings.keys(), "No model information was specified."
    builder = ModelBuilder(args.verbose, not args.plain_text)
    builder.set_from_dict(settings['model'])
    if args.code:
        code = builder.generate()
        if not args.plain_text:
            code = re.sub(r"(?<!\w)(l_[a-zA-z]\w*)", f"{txt.bold}{txt.green}\\g<1>{txt.end}", code, flags=re.M)
            code = re.sub(r"(?<!\w)(c_[a-zA-z]\w*)", f"{txt.bold}{txt.blue}\\g<1>{txt.end}", code, flags=re.M)
            code = re.sub(r"(?<!\w)(params(\[\d+\])?)", f"{txt.bold}{txt.yellow}\\g<1>{txt.end}", code, flags=re.M)
        print(code)
    if args.analyze:
        print(builder.summary())
        builder.validate_constants(consts, True)

    # === Run Sampler ==
    sampler_type = settings['sampling'].get('sampler', "emcee")
    sampler_args = settings['sampling'].get(sampler_type + "_init", {})
    sampler = builder.build_sampler(sampler_type, constants=consts, **sampler_args)
    if args.test_case:
        test_case = np.array([float(x) for x in args.test_case.split(",")])
        assert len(test_case) == len(sampler.param_names)
        out = sampler.model.forward_model(test_case)
        padding = max(len(i) for i in set(sampler.param_names) | set(out.keys()))
        for name, value in zip(sampler.param_names, test_case):
            print(f"p.{name:<{padding}}  {value:.6}")
        for name, value in out.items():
            print(f"l.{name:<{padding}}  {value:.6}")
        print("log_like".ljust(padding), f"   {sampler.model.log_like(test_case):.6}")
        print("log_prior".ljust(padding), f"   {sampler.model.log_prior(test_case):.6}")
    if args.dry_run:
        return
    sampler_args = settings['sampling'].get(sampler_type + "_run", {})
    sampler.run(**sampler_args)

    # === Write Outputs ===
    out: dict = {"terminal": False, "file": ""}
    out.update(settings['output'])
    if out['terminal']:
        print(sampler.summary())
    if out['file'] != "":
        sampler.save_results(out['file'])
