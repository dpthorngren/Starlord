import sys
from pathlib import Path

import numpy as np

from .samplers import ResultStats

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def classify_file(filename: str | Path) -> str:
    extension = Path(filename).suffix
    if extension == ".toml":
        return "model"
    elif extension == ".npz":
        # Minimum requirements to be considered a Starlord file:
        post_contents = ['params', 'param_names', 'outputs', 'output_names']
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


def load_posterior(filename, metadata_only=False, include_outputs=True):
    file = np.load(filename)
    expected_keys = ['params', 'outputs', 'output_names', 'param_names']
    assert all([k in file.files for k in expected_keys]), f"File {filename} does not appear to be a Starlord output."
    result = dict(
        # Required keys
        output_names=[str(i) for i in file['output_names']],
        param_names=[str(i) for i in file['param_names']],
        # Optional keys
        constants=file.get('consts', np.array([])),
        const_names=[str(i) for i in file.get('output_names', [])],
        code=str(file.get('code', "")),
        code_hash=str(file.get('code_hash', "")),
        grids=[str(i) for i in file.get('grids', [])],
        grid_vars=[str(i) for i in file.get('grid_vars', [])],
        stats=ResultStats.create_from_array(file['stats']) if 'stats' in file.files else None,
        time=str(file.get('time', "")),
        starlord_version=str(file.get('starlord_version', "")),
        python_version=str(file.get('python_version', "")),
    )
    if not metadata_only:
        posterior = file['params']
        if include_outputs:
            posterior = np.hstack([posterior, file['outputs']])
        if 'weights' in file.files:
            result['output_names'] = [str(i) for i in file['output_names']] + ["weights"]
            posterior = np.hstack([posterior, file['weights'][:, None]])
        result['posterior'] = posterior  # type:ignore
    return result


def load_to_frame(filename, simplify_names=True, include_outputs=True):
    '''Loads an npz file saved by Starlord into a Pandas Data Frame.

    This requires that Pandas is installed, but this is not a required dependency so
    that is not guaranteed by a standard install.

    Args:
        filename: The npz file to load in as a string.
        simplify_names: Whether to remove grid names at the front of variable names and
            combine underscores if the resulting resulting name is unambiguous (e.g.
            "mist__logG__1" becomes "logG_1".
        include_outputs: If true, includes generated outputs; otherwise only the actual
            model parameters are loaded.

    Returns:
        A Pandas DataFrame with the output samples organized into rows and the parameters
            and output variables as the columns.  If nested sampling was used, the weights
            are included as an additional column.

    Raises:
        AssertionError: if expected entries in the npz file are missing, implying that the file
            was not saved by Starlord.
    '''
    import pandas as pd

    data = load_posterior(filename)
    posterior = data['posterior']
    names = data['param_names']
    if include_outputs and simplify_names:
        for i in data['output_names']:
            isplit = str(i).split("__")
            if (len(isplit) > 1) and (isplit[0] in data['grids']):
                simplified = "_".join(isplit[1:])
            else:
                simplified = "_".join(isplit)
            if simplified in names:
                names.append(i)
            else:
                names.append(simplified)
    else:
        names += [str(i) for i in data['output_names']]
    return pd.DataFrame(posterior, columns=names)  # type:ignore
