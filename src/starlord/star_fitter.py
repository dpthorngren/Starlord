from __future__ import annotations

import re

from .code_gen import CodeGenerator
from .grid_gen import GridGenerator
from .sampler import SamplerNested


class StarFitter():
    '''Fits parameters of a stellar grid to observed data'''

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._gen = CodeGenerator(verbose)
        self.grids = {}
        self.used_grids = {}
        self.all_grids = GridGenerator.grids()
        self._generate_prior_transform = self._gen.generate_prior_transform

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
                if type(value) in [str, float, int]:
                    self.assign(key, str(value))
                elif type(value) is list:
                    assert type(value[0]) is str
                    assert value[0] not in self.all_grids.keys()
                    self.assign(key, value.pop(0))
                    if len(value) > 0:
                        self._unpack_distribution("l." + key, value)
        if "prior" in model.keys():
            for key, value in model['prior'].items():
                if self.verbose:
                    print(key, value)
                self._unpack_distribution("p." + key, value, True)
        for grid in self.all_grids.keys():
            if grid in model.keys():
                for key, value in model[grid].items():
                    assert len(value) in [2, 3]
                    if self.verbose:
                        print(grid, key, value)
                    self._register_grid_key(grid, key)
                    self._unpack_distribution(f"l.{grid}_{key}", value)

    def _register_grid_key(self, grid: str, key: str):
        assert grid in self.all_grids.keys(), f"Grid {grid} not recognized."
        assert key in self.all_grids[grid].provides, f"{key} not in grid {grid}."
        self.used_grids.setdefault(grid, set())
        self.used_grids[grid].add(key)

    def generate(self):
        self._resolve_grids()
        return self._gen.generate()

    def _generate_log_like(self):
        self._resolve_grids()
        return self._gen.generate_log_like()

    def _resolve_grids(self) -> None:
        # TODO: Handle grids already in the generator
        self.grids = {}
        for name, keys in self.used_grids.items():
            # TODO Support multiple keys
            key = list(keys)[0]
            grid = self.all_grids[name]
            self.grids[name] = grid.build_grid(key)
            n = len(grid.inputs)
            params = ", ".join([f"p.{p}" for p in grid.inputs])
            grid_var = f"c.grid_{name}"
            self.assign(f"l.{name}_{key}", f"{grid_var}._interp{n}d({params})")
            self._gen.constant_types[grid_var[2:]] = "GridInterpolator"

    def expression(self, expr: str) -> None:
        if self.verbose:
            print(f"    SF: Expression('{expr[:50]}...')")
        # Switch any tabs out for spaces
        expr = expr.replace("\t", "    ")
        # Identify grids, register required columns
        match = re.findall(r"(?<=[\W])(\w+)\.([A-Za-z_]\w*)", expr)
        if match is not None:
            for label, name in set(match):
                if label in 'pcbl':
                    continue
                elif label in self.all_grids.keys():
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

    def constraint(self, var: str, dist: str, params: list[str | float]) -> None:
        '''Adds a constraint to the model, either "l.var" or "grid.var".'''
        if self.verbose:
            print(f"    SF: Constraint({dist}({var} | {params})", end="")
        label, name = var.split(".")  # TODO: better exception
        if label in self.all_grids.keys():
            self._register_grid_key(label, name)
            if self.verbose:
                print(" (Grid Variable)")
        else:
            assert label in "lp"
            if self.verbose:
                print(" (Normal Variable)")
        self._gen.constraint(f"{label}.{name}", dist, params)

    def prior(self, var: str, dist: str, params: list[str | float]):
        if self.verbose:
            print(f"    SF: Prior {var} ~ {dist}({params})")
        if not var.startswith("p."):
            assert "." not in var
            var = "p." + var
        self._gen.constraint(var, dist, params, True)

    def _unpack_distribution(self, var: str, spec: list, is_prior: bool = False) -> None:
        '''Checks if spec specifies a distribution, otherwise defaults to normal.  Passes
        the results on to prior(...) if prior=True else constraint(...)'''
        assert type(spec) is list
        assert len(spec) >= 2
        dist: str = "normal"
        if type(spec[0]) is str:
            dist = spec.pop(0)
        if is_prior:
            self.prior(var, dist, spec)
        else:
            self.constraint(var, dist, spec)

    def summary(self, print_code: bool = False, prior_type="ppf") -> None:
        print("Grids:", self.used_grids)
        print(self._gen.summary(print_code, prior_type))

    def run_sampler(self, options: dict):
        # TODO: Move some of this over back into CodeGenerator
        hash = CodeGenerator._compile_to_module(self.generate())
        mod = CodeGenerator._load_module(hash)
        samp = SamplerNested(mod.log_like, mod.prior_transform, len(self._gen.params), {})
        samp.run({})
        return samp
