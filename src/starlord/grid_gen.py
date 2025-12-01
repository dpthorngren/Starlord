from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ._config import config
from .cy_tools import GridInterpolator


class GridGenerator:
    _initialized = False
    _grids = {}

    @classmethod
    def create_grid(
            cls,
            grid_name: str,
            inputs: OrderedDict[str, np.ndarray],
            outputs: dict[str, np.ndarray],
            derived: dict[str, str] = {},
            default_inputs: dict[str, str] = {}):
        # General validity checks
        assert type(grid_name) is str
        assert type(inputs) is OrderedDict, "Inputs must be type collections.OrderedDict; the order matters."
        assert type(outputs) is dict
        assert type(derived) is dict
        assert type(default_inputs) is dict
        assert not outputs.keys() & inputs.keys(), "Outputs and inputs have overlapping names."
        shape = []
        for name, input in inputs.items():
            assert re.fullmatch(r'[a-zA-Z1-9]\w*', name), f'Input name "{name}" is not valid.'
            assert input.ndim == 1, f'Input "{name}" is not 1d as required.'
            shape.append(len(input))
            assert np.all(np.diff(input) > 0), f'Input {name} was not strictly increasing as required.'
        shape = tuple(shape)
        for name, output in outputs.items():
            assert re.fullmatch(r'[a-zA-Z1-9]\w*', name), f'Output name "{name}" is not valid.'
            assert output.shape == shape, f'Output shape of "{name}" was {output.shape}; expected {shape}.'
            assert np.any(np.isfinite(output)), f'Output "{name}" is entirely bad values (inf, nan, etc).'
        assert type(derived) is dict
        assert not derived.keys() & inputs.keys(), "Derived and inputs have overlapping names."
        assert not derived.keys() & outputs.keys(), "Derived and outputs have overlapping names."
        for name, output in derived.items():
            assert re.fullmatch(r'[a-zA-Z1-9]\w*', name), f'Derived value name "{name}" is not valid.'
            assert type(output) is str
            # TODO: Validate derived parameter formulas
        assert type(default_inputs) is dict
        for name, output in default_inputs.items():
            assert name in inputs.keys(), f'Input default "{name}" doesn\'t match any actual inputs.'
            assert type(output) is str

        # Construct metadata
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
            _default_inputs=json.dumps(default_inputs),
            _derived=json.dumps(derived),
            _bounds=bounds,
            **inout_arrays,
        )

    @classmethod
    def register_grid(cls, filename: str) -> None:
        grid = np.load(filename)
        if "_grid_spec" not in grid.files:
            raise ValueError(f"Not a valid grid file: {filename}")
        gridname = Path(filename).stem
        assert gridname not in cls._grids.keys(), "Grid already registered"
        cls._grids[gridname] = GridGenerator(filename)

    @classmethod
    def reload_grids(cls) -> None:
        cls._grids = {}
        for filename in config.grid_dir.glob("*.npz"):
            try:
                cls.register_grid(filename)
            except ValueError:
                pass  # Non-grid file, ignore it

    @classmethod
    def grids(cls) -> dict[str, GridGenerator]:
        if not cls._initialized:
            cls.reload_grids()
        return cls._grids

    @classmethod
    def get_grid(cls, grid_name: str) -> GridGenerator:
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
        self._default_inputs = {p: f"p.{p}" for p in self.inputs}
        if '_default_inputs' in self.data.files:
            self._default_inputs.update(json.loads(str(self.data['_default_inputs'])))

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

    def get_input_map(self, overrides={}):
        '''Returns a dict converting grid input names into variables to use
        in generated code.  In order of decreasing priority these are
        overrides[input_name], the grid default for that input, or
        "p.{input_name}" if neither exists.
        '''
        overrides = {k: v for k, v in overrides.items() if k in self.inputs}
        input_map = self._default_inputs.copy()
        input_map.update(overrides)
        return input_map

    def summary(self, full=False) -> None:
        print(f"=== Grid {self.name} ===")
        print("   ", "Input".ljust(10), "Min".rjust(10), "Max".rjust(10))
        for i, name in enumerate(self.inputs):
            print(
                f"{i:>3d} {name:<10s}",
                f"{self.bounds[i,0]:>10.4n}",
                f"{self.bounds[i,1]:>10.4n}",
            )
        print("== Outputs ==")
        if len(self.outputs) < 12 or full:
            print(*[f"    {i}" for i in self.outputs], sep="\n")
        else:
            print(*[f"    {i}" for i in self.outputs[:12]], sep="\n")
            print(f"    [+{len(self.outputs)-12} more]")
        print("== Derived ==")
        if len(self.derived) < 12 or full:
            print(*[f"    {i}" for i in self.derived], sep="\n")
        else:
            print(*[f"    {i}" for i in self.derived.keys()][:12], sep="\n")
            print(f"    [+{len(self.derived)-12} more]")

    def build_grid(self, column: str) -> GridInterpolator:
        assert column in self.provides
        if column in self.derived:
            # TODO: Handle derived columns in Python
            raise NotImplementedError
        axes = [self.data[i] for i in self.inputs]
        values = self.data[column]
        return GridInterpolator(axes, values)
