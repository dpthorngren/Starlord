from __future__ import annotations

from pathlib import Path

import numpy as np

from ._config import config
from .cy_tools import GridInterpolator


class GridGenerator:
    _initialized = False
    _grids = {}

    @classmethod
    def register_grid(cls, filename: str):
        grid = np.load(filename)
        if "grid_spec" not in grid.files:
            raise ValueError(f"Not a valid grid file: {filename}")
        gridname = Path(filename).stem
        assert gridname not in cls._grids.keys()
        cls._grids[gridname] = GridGenerator(filename)

    @classmethod
    def reload_grids(cls):
        cls._grids = {}
        for filename in config.grid_dir.glob("*.npz"):
            try:
                cls.register_grid(filename)
            except ValueError:
                pass  # Non-grid file, ignore it

    @classmethod
    def get_grid(cls, grid_name: str):
        if not cls._initialized:
            cls.reload_grids()
        return cls._grids[grid_name]

    def __init__(self, filename):
        self.file_path = Path(filename)
        self.name = self.file_path.stem
        self.data = np.load(filename)
        assert "grid_spec" in self.data.files
        self.spec: str = str(self.data['grid_spec'])
        spec = self.spec.split('->')
        self.inputs: list[str] = [i.strip() for i in spec[0].split(",")]
        spec = spec[1].split(";")
        self.outputs: list[str] = [i.strip() for i in spec[0].split(",")]
        self.derived: list[str] = [i.strip() for i in spec[1].split(",")]
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

    def build_grid(self, columns: list[str] | str):
        if type(columns) is str:
            columns = [columns]
        assert len(columns) > 0
        for col in columns:
            assert col in self.outputs
        if len(columns) > 1:
            raise NotImplementedError("TODO: grids with multiple return values.")
        axes = [self.data[i] for i in self.inputs]
        value = self.data[columns[0]]
        return GridInterpolator(axes, value)
