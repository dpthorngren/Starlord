from starlord._version import __version__
from starlord.star_fitter import StarFitter
import argparse


def main():
    parser = argparse.ArgumentParser(
        "starlord", description="Fit stellar observations with starlord from the command line.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", action="version", version=f"starlord {__version__}")
    args = parser.parse_args()
    if args.dry_run:
        print("This command doesn't do anything yet, but if it did this would be a dry-run.")
    else:
        print("Doing everything in the power of this command (nothing).")
        StarFitter()
