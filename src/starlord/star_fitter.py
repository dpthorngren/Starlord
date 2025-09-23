from __future__ import annotations

from .code_components import Symb
from .code_gen import CodeGenerator


class StarFitter():
    '''Fits parameters of a stellar grid to observed data'''
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._gen = CodeGenerator()
        self._grids = {}
        self._avail_grids = {"mist": None}  # TODO: Real grid loading

    def set_from_dict(self, model: dict) -> None:
        if self.verbose:
            print("Loading from model dict:", model)
        if "expr" in model.keys():
            print("TODO: raw expression list passing")
        if "var" in model.keys():
            for key, value in model['var'].items():
                if type(value) is str:
                    self.assignment(key, value)
                elif type(value) is list:
                    assert type(value[0]) is str
                    # TODO: Check if value[0] is a distribution (which would be an error)
                    self.assignment(key, value.pop(0))
                    if len(value) > 0:
                        self._constraint("var." + key, value)
            print(f"TODO: Variable assignment")
        if "prior" in model.keys():
            print(f"TODO: prior assignment")
        for grid in self._avail_grids.keys():
            if grid in model.keys():
                self._grids.setdefault(grid, set())
                for key, value in model[grid].items():
                    self._constraint(grid + "." + key, value)

    def assignment(self, var: str, expr: str) -> None:
        print("TODO: variable assignment")
        return

    def _constraint(self, var: str, spec: list) -> None:
        assert type(spec) == list
        assert len(spec) >= 2
        dist: str = "normal"
        if type(spec[0]) is str:
            dist = spec.pop(0)
        self.constraint(var, dist, spec)

    def constraint(self, var: str, dist: str, params: list[float]) -> None:
        '''Adds a constraint to the model, either "var" or "grid.var".'''
        if self.verbose:
            print(f"Adding constraint {dist}({var} | {params})")
        label, name = var.split(".")  # TODO: better exception
        if label == "var":
            print("Not a grid, simple assignment")
        else:
            assert label in self._avail_grids.keys()
            print("TODO: Provide grid variable.")
            # Or recurse if it's a computable
            # assert key in self._avail_grids[label].keys() # Need grid object
            # Register that we're using label.key
            # self._grids[label] += key
            # Assign variable label_key = interp{N}D(a_label({params})
            # self.assignment()

    def _ensure_symbol(self, symb: str) -> Symb:
        return Symb(symb)

    def run_sampler(self, options: dict) -> dict:
        if self.verbose:
            print(self._gen.summary())
            print("TODO: run sampler with options: ", options)
        return {}
