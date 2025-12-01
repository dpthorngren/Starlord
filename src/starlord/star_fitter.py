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
        self._grids = {}
        self._used_grids = {}
        self._input_overrides = {}

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
                    assert value[0] not in GridGenerator.grids().keys()
                    self.assign(key, value.pop(0))
                    if len(value) > 0:
                        self._unpack_distribution("l." + key, value)
        if "prior" in model.keys():
            for key, value in model['prior'].items():
                if self.verbose:
                    print(key, value)
                self._unpack_distribution("p." + key, value, True)
        for grid in GridGenerator.grids().keys():
            if grid in model.keys():
                for key, value in model[grid].items():
                    assert len(value) in [2, 3]
                    if self.verbose:
                        print(grid, key, value)
                    self._register_grid_key(grid, key)
                    self._unpack_distribution(f"l.{grid}_{key}", value)
        if "override" in model.keys():
            for key, override in model['override'].items():
                if self.verbose:
                    print(key, override)
                for input_name, value in override.items():
                    self.override_input(key, input_name, value)

    def override_input(self, grid_name: str, input_name: str, value: str):
        grid = GridGenerator.get_grid(grid_name)
        assert input_name in grid.inputs, f"Cannot override nonexistent input {input_name}"
        self._input_overrides.setdefault(grid_name, {})
        self._input_overrides[grid_name][input_name] = value

    def expression(self, expr: str) -> None:
        if self.verbose:
            exprStr = expr[50:] + "..." if len(expr) > 50 else expr
            print(f"    SF: Expression('{exprStr}')")
        # Switch any tabs out for spaces and process any grids
        expr = expr.replace("\t", "    ")
        expr = self._extract_grids(expr)
        if self.verbose:
            print("    ---> ", expr)
        self._gen.expression(expr)

    def assign(self, var: str, expr: str) -> None:
        if self.verbose:
            print(f"    SF: Assignment({var}, '{expr[:50]}...')")
        expr = self._extract_grids(expr)
        self._gen.assign(var, expr)

    def constraint(self, var: str, dist: str, params: list[str | float]) -> None:
        '''Adds a constraint to the model, either "l.var" or "grid.var".'''
        if self.verbose:
            print(f"    SF: Constraint({dist}({var} | {params})")
        var = self._extract_grids(var)
        assert var.count(".") == 1, 'Variables must be of the form "label.name".'
        label, name = var.split(".")
        assert label in "pbl", "Variable label must be a grid name, p, b, or l."
        self._gen.constraint(f"{label}.{name}", dist, params)

    def prior(self, var: str, dist: str, params: list[str | float]):
        if self.verbose:
            print(f"    SF: Prior {var} ~ {dist}({params})")
        if not var.startswith("p."):
            assert "." not in var
            var = "p." + var
        self._gen.constraint(var, dist, params, True)

    def summary(self, print_code: bool = False, prior_type="ppf") -> None:
        self._resolve_grids()
        print("Grids:", self._used_grids)
        print(self._gen.summary(print_code, prior_type))

    def generate(self):
        self._resolve_grids()
        return self._gen.generate()

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

    def _extract_grids(self, source: str) -> str:
        '''Extracts grid names from the source string and replaces them with local variables.
        Registers the grid variables to be interpolated on grid resolution.'''
        # Identifies variables of the form "foo.bar", including grids, variables, and library functions.
        match = re.findall(r"([a-z_]\w*)\.([A-Za-z_]\w*)", source)
        if match is not None:
            for label, name in set(match):
                if label in GridGenerator.grids().keys():
                    self._register_grid_key(label, name)
                    source = source.replace(f"{label}.{name}", f"l.{label}_{name}")
        return source

    def _register_grid_key(self, grid: str, key: str):
        '''Adds a grid to the list and key to the target outputs.  Redundant calling is fine.'''
        assert grid in GridGenerator.grids().keys(), f"Grid {grid} not recognized."
        assert key in GridGenerator.grids()[grid].provides, f"{key} not in grid {grid}."
        self._used_grids.setdefault(grid, set())
        self._used_grids[grid].add(key)

    def _resolve_grids(self) -> None:
        '''Add grid interpolator components to the generator object (deleting existing ones)
        and build the required grid objects, storing them in self.grids.'''
        # Remove any previously autogenerated components
        self._gen.remove_generated()
        self._grids.clear()
        self._gen._mark_autogen = True
        input_maps = {}

        try:
            # First pass handles derived grid outputs and default parameters that refer to other grids
            while True:
                for name, columns in self._used_grids.items():
                    grid = GridGenerator.get_grid(name)

                    # Resolve grid inputs that depend on other grids
                    if name not in input_maps.keys():
                        in_map = grid.get_input_map(self._input_overrides.get(name, {}))
                        input_maps[name] = {k: self._extract_grids(v) for k, v in in_map.items()}
                        break

                    # Identify desired grid outputs that are derived but not already resolved
                    name_map = {f"derived_{name}_{c}": c for c in columns if c in grid.derived}
                    derived = set(name_map.keys()) - set(self._grids.keys())
                    if len(derived) != 0:
                        der = derived.pop()
                        # Sub variables into the code needed to calculate the derived grid outputs
                        mapping = {k: f"p.{k}" for k in grid.inputs}
                        mapping.update({k: f"{name}.{k}" for k in grid.provides})
                        code = str(grid.derived[name_map[der]]).format_map(mapping)
                        # Add the code to _grids for tracking and send the assigment code to GridGenerator
                        self._grids[der] = code
                        self.assign("l." + der[8:], code)
                        # Begin again in case it recursively requires additional grids / vars
                        break
                else:
                    break

            # Second pass builds the grids and add interpolators to the code generator
            for name, keys in self._used_grids.items():
                grid = GridGenerator.get_grid(name)
                input_map = input_maps[name]
                for key in keys:
                    if key in grid.derived:
                        continue
                    grid_var = f"grid_{name}_{key}"
                    self._grids[grid_var] = grid.build_grid(key)
                    param_string = ", ".join([input_map[i] for i in grid.inputs])
                    self.assign(f"l.{name}_{key}", f"c.{grid_var}._interp{grid.ndim}d({param_string})")
                    self._gen.constant_types[grid_var] = "GridInterpolator"
        except Exception as e:
            # Must disable marking components as autogenerated whether or not there was an exception.
            self._gen._mark_autogen = False
            raise e
        self._gen._mark_autogen = False

    def run_sampler(self, options: dict, constants: dict = {}):
        self._resolve_grids()
        mod = self._gen.compile()
        constants.update(self._grids)
        params = [p[2:] for p in self._gen.params]
        consts = [constants[str(c.name)] for c in self._gen.constants]
        samp = SamplerNested(mod.log_like, mod.prior_transform, len(self._gen.params), {}, consts, params)
        samp.run(options)
        return samp
