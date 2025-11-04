from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ._config import config
from .cy_tools import GridInterpolator


class GridGenerator:
    _initialized = False
    _grids = {}

    @classmethod
    def register_grid(cls, filename: str) -> None:
        grid = np.load(filename)
        if "grid_spec" not in grid.files:
            raise ValueError(f"Not a valid grid file: {filename}")
        gridname = Path(filename).stem
        assert gridname not in cls._grids.keys()
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
        assert "grid_spec" in self.data.files
        self.spec: str = str(self.data['grid_spec'])
        spec = self.spec.split('->')
        self.inputs: list[str] = [i.strip() for i in spec[0].split(",")]
        spec = spec[1].split(";")
        self.outputs: list[str] = [i.strip() for i in spec[0].split(",")]
        self.derived: list[str] = []
        if len(spec) > 1:
            self.derived = [i.strip() for i in spec[1].split(",")]
        self.provides = self.outputs + self.derived
        for k in self.inputs + self.outputs:
            assert k in self.data.files

    def __repr__(self) -> str:
        out = f"Grid_{self.name}("
        out += ", ".join(self.inputs)
        out += " -> " + ", ".join(self.outputs)
        if len(self.derived) > 0:
            out += "; " + ", ".join(self.derived)
        out += ")"
        return out

    def build_grid(self, columns: list[str] | str) -> GridInterpolator:
        if type(columns) is str:
            columns = [columns]
        assert len(columns) > 0
        # Sort columns into outputs and derived values
        derived: list[str] = [c for c in columns if c in self.derived]
        outputs: list[str] = [c for c in columns if c not in self.derived]
        assert all([c in self.outputs for c in outputs])
        if len(outputs) > 1:
            raise NotImplementedError("TODO: grids with multiple return values.")
        axes = [self.data[i] for i in self.inputs]
        value = self.data[columns[0]]
        # Generate function to compute derived values, if there are any
        derived_map = {k: str(self.data[k]) for k in self.derived}
        get_derived = None
        if len(derived) > 0:
            mapping = {name: f"inputs[{i}]" for i, name in enumerate(self.inputs)}
            mapping.update({name: f"outputs[{i}]" for i, name in enumerate(self.outputs)})
            mapping.update({name: f"result[{i}]" for i, name in enumerate(self.derived)})
            func = ["def get_derived(inputs, outputs):"]
            func += ["    inputs = np.atleast_1d(inputs)"]
            func += ["    outputs = np.atleast_1d(outputs)"]
            func += [f"    result = np.zeros({len(derived)})"]
            # TODO: Handle derived value inter-dependencies and other grid dependencies.
            for i, d in enumerate(derived):
                func += [f"    result[{i}] = " + derived_map[d].format_map(mapping)]
            func += ["    return result.squeeze()"]
            locals = {}
            exec("\n".join(func), {'math': math, 'np': np}, locals)
            get_derived = locals['get_derived']
        return GridInterpolator(
            axes,
            value,
            inputs=self.inputs,
            outputs=outputs,
            derived=derived_map,
            get_derived=get_derived,
        )
