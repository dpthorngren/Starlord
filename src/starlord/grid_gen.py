from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import numpy as np

from ._config import config
from .cy_tools import GridInterpolator


class GridGenerator:
    '''Manages grids and generates grid interpolators.

    You can use :meth:`create_grid` to make a new grid. Starlord uses the class
    methods :meth:`reload_grids` :meth:`register_grid`, :meth:`grids`, and
    :meth:`get_grid` to manage the grids available to it. These are all optional
    for the user -- you can initialize a GridGenerator directly on a file path.
    Once you have a GridGenerator ready, you can use :meth:`build_grid` to make
    an interpolator in the desired output.
    '''

    _initialized = False
    _grids = {}

    @classmethod
    def create_grid(
            cls,
            grid_name: str,
            inputs: OrderedDict[str, np.ndarray],
            outputs: dict[str, np.ndarray],
            derived: dict[str, str] = {},
            input_mappings: dict[str, str] = {}) -> None:
        '''Create a new grid and write it to the Starlord grid directory.

        The input, output, and derived names must be unique, valid as Python variable names, and not start with "_".
        A fair few validity checks are made to ensure that the grid is valid before the grid is written.  Once
        you make a grid with this function, you can use it in your Starlord models or build interpolators
        using :func:`get_grid` and :func:`build_grid`.

        Args:
            grid_name: A name for your grid, overwrites any existing grid of the same name. If the name does not include
                a directory, the file will be saved in the Starlord grid storage.
            inputs: The grid inputs as an OrderedDict of 1-d, strictly-increasing arrays of floats in the same
                order as the output axes.
            outputs: The output variables for the grid, a dict of float arrays with a shape corresponding to the
                inputs provided.
            derived: Values that may be computed from the grid (the dict keys) and the code required to compute
                them (the values).  Variables used must be inputs, outputs, or derived keys and enclosed by curly
                braces.
            input_mappings: The code to be used for the inputs, by axis (keys must match input keys) if not overridden
                by the model.  If not specified, this defaults to being a model parameter "p.[input_name]".

        Raises:
            AssertionError: If any of the validity checks fail -- see the error message for further explanation.
        '''
        # General validity checks
        assert type(grid_name) is str
        assert type(inputs) is OrderedDict, "Inputs must be type collections.OrderedDict; the order matters."
        assert type(outputs) is dict
        assert type(derived) is dict
        assert type(input_mappings) is dict
        assert not outputs.keys() & inputs.keys(), "Outputs and inputs have overlapping names."
        # Sort outputs alphabetically by key
        outputs = OrderedDict(sorted(outputs.items(), key=lambda i: i[0].lower()))
        derived = OrderedDict(sorted(derived.items(), key=lambda i: i[0].lower()))
        input_mappings = OrderedDict(sorted(input_mappings.items(), key=lambda i: i[0].lower()))

        # Check input validity and extract shape
        shape = []
        for name, input in inputs.items():
            assert re.fullmatch(r'[a-zA-Z1-9]\w*', name), f'Input name "{name}" is not valid.'
            assert input.ndim == 1, f'Input "{name}" is not 1d as required.'
            shape.append(len(input))
            assert np.all(np.diff(input) > 0), f'Input {name} was not strictly increasing as required.'
        shape = tuple(shape)

        # Check output validity
        for name, output in outputs.items():
            assert re.fullmatch(r'[a-zA-Z1-9]\w*', name), f'Output name "{name}" is not valid.'
            assert output.shape == shape, f'Output shape of "{name}" was {output.shape}; expected {shape}.'
            assert np.any(np.isfinite(output)), f'Output "{name}" is entirely bad values (inf, nan, etc).'
        assert not derived.keys() & inputs.keys(), "Derived and inputs have overlapping names."
        assert not derived.keys() & outputs.keys(), "Derived and outputs have overlapping names."
        for name, output in derived.items():
            assert re.fullmatch(r'[a-zA-Z1-9]\w*', name), f'Derived value name "{name}" is not valid.'
            assert type(output) is str
            # TODO: Validate derived parameter formulas
        for name, output in input_mappings.items():
            assert name in inputs.keys(), f'Input default "{name}" doesn\'t match any actual inputs.'
            assert type(output) is str

        # Construct metadata and create the grid
        grid_spec = ", ".join(inputs.keys())
        grid_spec += " -> "
        grid_spec += ", ".join(outputs.keys())
        if derived:
            grid_spec += "; "
            grid_spec += ", ".join(derived.keys())
        bounds = np.column_stack([
            [np.min(i) for i in inputs.values()],
            [np.max(i) for i in inputs.values()],
        ])
        inout_arrays = dict(inputs)
        inout_arrays.update(outputs)
        filepath = str(config.grid_dir / grid_name) if "/" not in grid_name else grid_name
        np.savez_compressed(
            filepath,
            _grid_spec=grid_spec,
            _input_mappings=json.dumps(input_mappings),
            _derived=json.dumps(derived),
            _bounds=bounds,
            _shape=shape,
            **inout_arrays,
        )
        GridGenerator.reload_grids()

    @classmethod
    def register_grid(cls, filename: str) -> None:
        '''Add a grid by filename to the GridGenerator tracked list for e.g. :func:`get_grid`.

        The file does not need to be in the Starlord grid directory.

        Args:
            filename: The npz file to load the grid from

        Raises:
            AssertionError: if the grid is not a proper StarlordGrid from :func:`create_grid`
        '''
        grid = np.load(filename)
        if "_grid_spec" not in grid.files:
            raise ValueError(f"Not a valid grid file: {filename}")
        gridname = Path(filename).stem
        assert gridname not in cls._grids.keys(), "Grid already registered"
        cls._grids[gridname] = GridGenerator(filename)

    @classmethod
    def reload_grids(cls) -> None:
        '''Clear the grids and load them again from the grid directory.

        Note that this removes any grids added with :func:`register_grid` which are not in
        that directory.
        '''
        cls._grids = {}
        for filename in config.grid_dir.glob("*.npz"):
            try:
                cls.register_grid(filename)
            except ValueError:
                pass  # Non-grid file, ignore it

    @classmethod
    def grids(cls) -> dict[str, GridGenerator]:
        '''Gets a dict of the grids known to Starlord.'''
        if not cls._initialized:
            cls.reload_grids()
        return cls._grids.copy()

    @classmethod
    def get_grid(cls, grid_name: str) -> GridGenerator:
        '''Gets a specific grid from the dict of known grids.

        Raises:
            KeyError: if grid_name is not registered with Starlord.
        '''
        if not cls._initialized:
            cls.reload_grids()
        return cls._grids[grid_name]

    def __init__(self, filename: str | Path):
        self.file_path = Path(filename)
        self.name = self.file_path.stem
        self.data = np.load(str(filename))
        assert "_grid_spec" in self.data.files, f"{filename} is not a Starlord grid file."
        self.spec: str = str(self.data['_grid_spec'])
        spec = self.spec.split('->')
        self.bounds = self.data['_bounds']
        self.shape = tuple(self.data['_shape'])
        self.inputs: list[str] = [i.strip() for i in spec[0].split(",")]
        self.ndim = len(self.inputs)
        spec = spec[1].split(";")
        self.outputs: list[str] = [i.strip() for i in spec[0].split(",")]
        if '_derived' in self.data.files:
            self.derived: dict[str, str] = json.loads(str(self.data['_derived']))
        else:
            self.derived = {}
        self.provides = self.outputs + list(self.derived.keys())
        for k in self.inputs + self.outputs:
            assert k in self.data.files, f"Bad grid: {k} in _grid_spec but was not found."
        self._input_mappings = {p: f"p.{p}" for p in self.inputs}
        if '_input_mappings' in self.data.files:
            self._input_mappings.update(json.loads(str(self.data['_input_mappings'])))

    def __repr__(self) -> str:
        out = f"Grid_{self.name}("
        out += ", ".join(self.inputs)
        out += " -> " + ", ".join(self.outputs[:4])
        if len(self.outputs) > 4:
            out += f", +{len(self.outputs)-4}"
        if len(self.derived) > 0:
            out += "; " + ", ".join(list(self.derived.keys())[:4])
        if len(self.derived) > 4:
            out += f", +{len(self.derived)-4}"
        out += ")"
        return out

    def _get_input_map(self, overrides={}):
        '''Returns a dict converting grid input names into variables to use
        in generated code.  In order of decreasing priority these are
        overrides[input_name], the grid default for that input, or
        "p.{input_name}" if neither exists.
        '''
        overrides = {k: v for k, v in overrides.items() if k in self.inputs}
        input_map = self._input_mappings.copy()
        input_map.update(overrides)
        return input_map

    def summary(self, full: bool = False, fancy_text: bool = True) -> None:
        '''Prints basic information about the grid.

        Args:
            full: if False, only print the first few outputs and derived outputs,
                otherwise print them all.
            fancy_text: whether to style the output with colors and bolding.
        '''
        txt = config.text_format if fancy_text else config.text_format_off
        print(f"{txt.bold}{txt.underline}Grid {self.name}{txt.end}")
        print("   ", "Input".ljust(10), "Min".rjust(10), "Max".rjust(10), end=" ")
        print("Length".rjust(10), "    Default Mapping")
        for i, name in enumerate(self.inputs):
            print(
                f"{i:>3d} {txt.bold}{name:<10s}{txt.end}",
                f"{self.bounds[i, 0]:>10.4n}",
                f"{self.bounds[i, 1]:>10.4n}",
                f"{self.shape[i]:>10n}    ",
                f"{self._input_mappings[name]}",
            )
        print(f"{txt.underline}Outputs{txt.end}")
        if len(self.outputs) < 12 or full:
            print(*[f"    {i}" for i in self.outputs], sep="\n")
        else:
            print(*[f"    {i}" for i in self.outputs[:12]], sep="\n")
            print(f"    [+{len(self.outputs)-12} more]")
        print(f"{txt.underline}Derived{txt.end}")
        if len(self.derived) < 12 or full:
            print(*[f"    {i}" for i in self.derived], sep="\n")
        else:
            print(*[f"    {i}" for i in self.derived.keys()][:12], sep="\n")
            print(f"    [+{len(self.derived)-12} more]")

    def build_grid(
            self, column: str, axis_tf: dict[str, Callable] = {}, value_tf: Callable = lambda x: x) -> GridInterpolator:
        '''Build the grid into an interpolator of the requested column.

        Args:
            column (str): The output column to interpolate.
            axis_tf: A dictionary mapping input column names to functions to
                be applied to them before the interpolator is constructed.  Note
                that the transformed axis must still be in strictly-increasing order.
            value_tf: A function that will be applied to the output column.

        Returns:
            A GridInterpolator of the requested grid and output.

        Raises:
            AssertionError: if the column is not a grid output, the grid itself
                is malformed, or if an axis transform un-sorted the axis.
        '''
        assert column in self.provides
        assert all([k in self.inputs for k in axis_tf.keys()])
        if column in self.derived:
            # TODO: Handle derived columns in Python
            raise NotImplementedError
        axes = [axis_tf.get(k, lambda x: x)(self.data[k]) for k in self.inputs]
        assert all([np.all(np.diff(ax) > 0) for ax in axes])
        values = value_tf(self.data[column])
        return GridInterpolator(axes, values)
