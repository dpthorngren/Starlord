from __future__ import annotations

import re

from .code_gen import CodeGenerator
from .sampler import SamplerNested


class StarFitter():
    '''Fits parameters of a stellar grid to observed data'''

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._gen = CodeGenerator(verbose)
        self._grids = {}
        self._avail_grids = {"mist": None}  # TODO: Real grid loading

    def set_from_dict(self, model: dict) -> None:
        if self.verbose:
            print("Loading from model dict:", model)
        if "expr" in model.keys():
            for name, code in model['expr'].items():
                if self.verbose:
                    print(name, code[:50])
                self.expression(code)
        if "var" in model.keys():
            for key, value in model['var'].items():
                if self.verbose:
                    print(key, value)
                if type(value) is str:
                    self.assign(key, value)
                elif type(value) is list:
                    assert type(value[0]) is str
                    assert value[0] not in self._avail_grids.keys()
                    self.assign(key, value.pop(0))
                    if len(value) > 0:
                        self._unpack_distribution("l." + key, value)
        if "prior" in model.keys():
            for key, value in model['prior'].items():
                if self.verbose:
                    print(key, value)
                self._unpack_distribution("p." + key, value, True)
        for grid in self._avail_grids.keys():
            if grid in model.keys():
                for key, value in model[grid].items():
                    if self.verbose:
                        print(grid, key, value)
                    self._unpack_distribution(grid + "." + key, value)

    def _register_grid_key(self, grid: str, key: str):
        assert grid in self._avail_grids.keys()
        # TODO: Check if key is in the grid
        self._grids.setdefault(grid, set())
        self._grids[grid].add(key)

    def expression(self, expr: str) -> None:
        if self.verbose:
            print(f"    SF: Expression('{expr[:50]}...')")
        # Identify grids, register required columns
        match = re.findall(r"(?<=[\W])(\w+)\.([A-Za-z_]\w*)", expr)
        if match is not None:
            for label, name in set(match):
                if label in 'pcbla':
                    continue
                elif label in self._avail_grids.keys():
                    self._register_grid_key(label, name)
                    expr = expr.replace(f"{label}.{name}", f"l.{label}_{name}")
                # TODO: Check against library names to avoid compilation errors
        if self.verbose:
            print("    ---> ", expr)
        self._gen.expression(expr)

    def assign(self, var: str, expr: str) -> None:
        if self.verbose:
            print(f"    SF: Assignment({var}, '{expr[:50]}...')")
        self._gen.assign(var, expr)

    def constraint(self, var: str, dist: str, params: list[str]) -> None:
        '''Adds a constraint to the model, either "var" or "grid.var".'''
        if self.verbose:
            print(f"    SF: Constraint({dist}({var} | {params})", end="")
        label, name = var.split(".")  # TODO: better exception
        if label == "l":
            if self.verbose:
                print(" (Simple Variable)")
            self._gen.constraint("l." + name, dist, params)
            return
        assert label in self._avail_grids.keys(), label
        if self.verbose:
            print(" (Grid Variable)")
        self._register_grid_key(label, name)
        self._gen.constraint(f"{label}_{name}", dist, params)

    def prior(self, var: str, dist: str, params: list[str]):
        if self.verbose:
            print(f"    SF: Prior {var} ~ {dist}({params})")
        self._gen.constraint(var, dist, params, True)

    def _unpack_distribution(self, var: str, spec: list, is_prior: bool = False) -> None:
        '''Checks if spec specifies a distribution, otherwise defaults to normal.  Passes
        the results on to prior(...) if prior=True else constraint(...)'''
        assert type(spec) == list
        assert len(spec) >= 2
        dist: str = "normal"
        if type(spec[0]) is str:
            dist = spec.pop(0)
        if is_prior:
            self.prior(var, dist, spec)
        else:
            self.constraint(var, dist, spec)

    def summary(self, print_code: bool = False, prior_type="ppf") -> None:
        print("Grids:", self._grids)
        print(self._gen.summary(print_code, prior_type))

    def generate_log_like(self) -> str:
        return self._gen.generate_log_like()

    def run_sampler(self, options: dict):
        # TODO: Move some of this over back into CodeGenerator
        hash = CodeGenerator._compile_to_module(self._gen.generate())
        mod = CodeGenerator._load_module(hash)
        samp = SamplerNested(mod.log_like, mod.prior_transform, len(self._gen.params), {})
        samp.run({})
        return samp
