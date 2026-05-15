import sys
from pathlib import Path

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def classify_file(filename: str | Path) -> str:
    extension = Path(filename).suffix
    if extension == ".toml":
        return "model"
    elif extension == ".npz":
        post_contents = [
            'params', 'outputs', 'consts', 'output_names', 'param_names', 'const_names', 'code', 'code_hash', 'grids',
            'grid_vars'
        ]
        grid_contents = ['_grid_spec', '_input_mappings', '_derived', '_bounds', '_shape']
        target = np.load(filename)
        if all(i in target.files for i in post_contents):
            return "posterior"
        elif all(i in target.files for i in grid_contents):
            return "grid"
    return "unknown"


def read_model_toml(filename: str | Path) -> dict:
    # TODO: Handle syntax errors in the toml file
    with open(filename, 'rb') as f:
        results = tomllib.load(f)
    return results
