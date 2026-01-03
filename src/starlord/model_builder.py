from __future__ import annotations

import re
from typing import Tuple

from ._config import config
from .code_gen import CodeGenerator
from .grid_gen import GridGenerator
from .samplers import SamplerEnsemble, SamplerNested


class ModelBuilder():
    '''Builds and fits a Bayesian model to the given specification.

    Variables are defined implicitly -- if you use a variable, the ModelBuilder
    will handle declaring them based on their category, which is determined by
    a prefix (e.g. ``p.foo`` is a parameter named foo).  The categories of
    variable are:

    :Parameters: ``p.[name]``, these are model parameters to be sampled from.
    :Constants: ``c.[name]``, these are set when the sampler is run and don't
       change.
    :Local Variables: ``l.[name]`` these are calculated for each log likelihood call
       but not recorded
    :Grid Variables: ``[grid_name].[output_name]``, these indicate the grid
       should be interpolated to get the value, which will often result in more
       parameters being implicitly defined.

    Typically, you initialize the builder (there are no significant options at
    init) and use :meth:`constraint`, :meth:`assign`, and sometimes
    :meth:`expression` to define the model's likelihood.  Then you can look at
    how the model is set up with :meth:`summary`; using grid variables often
    automatically defines new parameters for their inputs.  If you don't like
    the default inputs, you can override them with `override_input`.  Finally,
    you must define priors with :meth:`prior` before you can get a sampler for
    the model with :meth:`build_sampler`.
    '''

    def __init__(self, verbose: bool = False, fancy_text: bool = True):
        '''
        Args:
            verbose: If True, print extra debugging info
            fancy_text: If True, color and style terminal output text
        '''
        self._verbose: bool = verbose
        self._fancy_text: bool = fancy_text
        self._gen = CodeGenerator(verbose)
        self._used_grids: dict[str, set[str]] = {}
        self._input_overrides: dict[str, dict[str, str]] = {}

    def set_from_dict(self, model: dict) -> None:
        '''Load model description from a dict following the TOML input spec.

        Args:
            model: The model dict to be loaded, it should only have the keys
                'expr', 'var', 'prior', 'override', or the name of a grid.

        Example:
            Loading the model from a TOML file to be used within the Python API::

                model = tomllib.load("mymodel.toml")['model']
                builder = ModelBuilder().set_from_dict(model)
        '''
        txt = config.text_format if self._fancy_text else config.text_format_off
        if self._verbose:
            print(f"    {txt.underline}Model Processing{txt.end}")
        if "expr" in model.keys():
            for name, code in model['expr'].items():
                if self._verbose:
                    print(f"expr.{name} = '{code}'")
                self.expression(code)
        if "var" in model.keys():
            for key, value in model['var'].items():
                if self._verbose:
                    print(f"var.{key} = {value}")
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
                if self._verbose:
                    print(f"prior.{key} = {value}")
                self._unpack_distribution("p." + key, value, True)
        for grid in GridGenerator.grids().keys():
            if grid in model.keys():
                for key, value in model[grid].items():
                    assert len(value) in [2, 3]
                    if self._verbose:
                        print(f"{grid}.{key} = {value}")
                    self._register_grid_key(grid, key)
                    self._unpack_distribution(f"l.{grid}_{key}", value)
        if "override" in model.keys():
            for key, override in model['override'].items():
                if self._verbose:
                    print(f"override.{key} = {override}")
                for input_name, value in override.items():
                    self.override_input(key, input_name, value)

    def override_input(self, grid_name: str, input_name: str, value: str):
        '''Sets the value or symbol to use for the given grid input.

        This can be used to fix grid axes to a particular value, or make them depend on some
        additional grid output or calculation.  Grid inputs are set by default according to
        the their entry in the `input_mappings` grid metadata.  If there is no entry then they
        default to being a parameter named "p.{input_name}".

        Args:
            grid_name: The grid to set the input of
            input_name: Which input to set
            value: What to set the input to

        Examples:
            Suppose you are fitting a stellar model and wish to lock the metallicity to solar.
            If you're using the mist grid, you could do this with::

                builder.override_input("mist", "feh", "0")

            In the same circumstance, if you wanted to set logG to 2% higher than
            what the evolution tracks output (as a sensitivity test, perhaps), you could use::

                builder.override_input("mist", "logG", "1.02*mistTracks.logG")

            Note that this uses another grid.  Starlord will detect and handle this without
            issue.  In fact, the default input refers to mistTracks.logG already.
        '''
        if self._verbose:
            print(f"  ModelBuilder.override_input('{grid_name}', '{input_name}', '{value}')")
        grid = GridGenerator.get_grid(grid_name)
        assert input_name in grid.inputs, f"Cannot override nonexistent input {input_name}"
        self._input_overrides.setdefault(grid_name, {})
        self._input_overrides[grid_name][input_name] = value

    def expression(self, expr: str) -> None:
        '''Directly insert an expression into the generated code.

        Starlord will identify any variables assigned or used within to ensure the code
        is sorted properly by dependency (see ModelBuilder docstring).  Most of the time
        you can use :func:`assign` or :func:`constraint` instead, but this gives you the flexibility
        to add more complicated log-likelihood calculations.  For now this is the only way
        to implement a for loop.

        Args:
            expr: The expression to be inserted into the code, as a str.
        '''
        if self._verbose:
            expr_str = expr[50:] + "..." if len(expr) > 50 else expr
            print(f"  ModelBuilder.expression('{expr_str}')")
        # Switch any tabs out for spaces and process any grids
        expr = expr.replace("\t", "    ")
        expr = self._extract_grids(expr)
        self._gen.expression(expr)

    def assign(self, var: str, expr: str) -> None:
        '''Adds a likelihood component that sets a local variable to the given expression.

        Args:
            var: The variable to be assigned (e.g. `l.varname`)
            expr: The value or expression to set the variable to (e.g. `math.log10(p.mass)`)

        Example:
            Suppose a grid you wish to use named `foo` outputs `bar`, but you have measured
            the sqrt(bar).  Rather than propagating uncertainties (an approximation),
            you could instead use::

                builder.assign("l.sqrt_bar", "math.sqrt(foo.bar)")
                builder.constraint("l.sqrt_bar", "normal", ["c.sqrt_bar_mu", "c.sqrt_bar_sigma"])

            Grid names are resolved in expr as usual.  I've written the mean and uncertainty
            as (arbitrarily-named) constants to set later, but you can use literals instead
            if you want to.
        '''
        # If l or b is omitted, l is implied
        var = var if re.match(r"^[bl]\.", var) is not None else f"l.{var}"
        if self._verbose:
            print(f"  ModelBuilder.assignment('{var}', {expr})")
        expr = self._extract_grids(expr)
        self._gen.assign(var, expr)

    def constraint(self, var: str, dist: str, params: list[str | float]) -> None:
        '''Adds a constraint term to the log-likelihood for the given distribution and variable.

        Args:
            var: The variable to which the distribution applies
            dist: The distribution to be used -- should be one of "uniform", "normal", "gamma",
                or "beta".
            params: The parameters of the distribution.

        Example:
            Suppose you are fitting a stellar model and the `2MASS_H` magnitude is 6.5 +/- 0.05.
            If you're using the `MIST` grid, you could add this constraint to the model with::

                builder.constraint("mist.2MASS_H", "normal", [6.5, 0.05])
        '''
        if self._verbose:
            print(f"  ModelBuilder.constraint('{var}', '{dist}', {params})")
        var = self._extract_grids(var)
        assert var.count(".") == 1, 'Variables must be of the form "label.name".'
        label, name = var.split(".")
        assert label in "pbl", "Variable label must be a grid name, p, b, or l."
        self._gen.constraint(f"{label}.{name}", dist, params)

    def prior(self, param: str, dist: str, params: list[str | float]) -> None:
        '''Sets the prior for a model parameter.  All parameters must have a prior.

        Args:
            param: The name of the parameter to set, e.g. `p.some_param`
            dist: The distribution to be used -- should be one of "uniform", "normal", "gamma",
                or "beta".
            params: The parameters of the distribution.
        '''
        if not param.startswith("p."):
            assert "." not in param
            param = "p." + param
        if self._verbose:
            print(f"  ModelBuilder.prior('{param}', '{dist}', {params})")
        self._gen.prior(param, dist, params)

    def summary(self) -> str:
        '''Generates a summary of the model currently defined.

        The model does not need to be in a finalized state to be run, so it may help
        to check this periodically as you build the model.

        Returns:
            The model summary.
        '''
        txt = config.text_format if self._fancy_text else config.text_format_off
        self._resolve_grids()
        result = [f"    {txt.underline}Grids{txt.end}"]
        if self._used_grids:
            for k, v in self._used_grids.items():
                result.append(k + " " + ", ".join(v))
        else:
            result.append("None")
        return "\n".join(result) + "\n\n" + self._gen.summary(self._fancy_text)

    def generate(self) -> str:
        '''Generates the code for the model.

        Returns:
            A string containing the generated Cython code.

        Raises:
            AssertionError: if one of the various consistency checks fails.
        '''
        self._resolve_grids()
        return self._gen.generate()

    def _unpack_distribution(self, var: str, spec: list, is_prior: bool = False) -> None:
        '''Checks if spec specifies a distribution, otherwise defaults to normal.  Passes
        the results on to :func:`prior` if prior=True else :func:`constraint`'''
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
        self._gen._mark_autogen = True
        input_maps = {}
        self._gen.auto_constants.clear()
        defined = set()

        try:
            # First pass handles derived grid outputs and default parameters that refer to other grids
            while True:
                # Sort the grids to make code generation deterministic
                for name in sorted(self._used_grids.keys()):
                    columns = sorted(self._used_grids[name])
                    grid = GridGenerator.get_grid(name)

                    # Resolve grid inputs that depend on other grids
                    if name not in input_maps.keys():
                        in_map = grid._get_input_map(self._input_overrides.get(name, {}))
                        input_maps[name] = {k: self._extract_grids(v) for k, v in in_map.items()}
                        break

                    # Identify desired grid outputs that are derived but not already resolved
                    name_map = {f"derived_{name}_{c}": c for c in columns if c in grid.derived}
                    derived = set(name_map.keys()) - defined
                    if len(derived) != 0:
                        der = derived.pop()
                        # Add the code to _grids for tracking and send the assigment code to GridGenerator
                        code = grid.derived[name_map[der]]
                        defined.add(der)
                        self.assign("l." + der[8:], code)
                        # Begin again in case it recursively requires additional grids / vars
                        break
                else:
                    break

            # Second pass builds the grids and add interpolators to the code generator
            for name in sorted(self._used_grids.keys()):
                grid = GridGenerator.get_grid(name)
                input_map = input_maps[name]
                for key in sorted(self._used_grids[name]):
                    if key in grid.derived:
                        continue
                    grid_var = f"grid_{name}_{key}"
                    defined.add(grid_var)
                    self._gen.auto_constants[grid_var] = f"GridGenerator.get_grid('{name}').build_grid('{key}')"
                    param_string = ", ".join([input_map[i] for i in grid.inputs])
                    self.assign(f"l.{name}_{key}", f"c.{grid_var}._interp{grid.ndim}d({param_string})")
                    self._gen.constant_types[grid_var] = "GridInterpolator"
        except Exception as e:
            # Must disable marking components as autogenerated whether or not there was an exception.
            self._gen._mark_autogen = False
            raise e
        self._gen._mark_autogen = False
        if self._verbose:
            print("")

    def validate_constants(self, constants: dict, print_summary: bool = False) -> Tuple[set[str], set[str]]:
        '''Check that the constants provided match those that were expected.

        Args:
            constants: a dict of the constant names and values (without the 'c.') to test.
            print_summary: if True, print a list of the constants and values, noting extra
                or missing constants.

        Returns:
            A set() of any missing constant names
            A set() of any extra constant names that weren't expected
        '''
        txt = config.text_format if self._fancy_text else config.text_format_off
        expected = {c.name for c in self._gen.constants}
        missing = expected - set(constants.keys())
        missing -= set(self._gen.auto_constants.keys())
        extra = set(constants.keys()) - expected
        if print_summary:
            print(f"\n    {txt.underline}Constant Values{txt.end}")
            if not missing and not constants.items():
                print("[None]")
            for k in missing:
                print(f"{txt.blue}{txt.bold}c.{k}{txt.end} is not set")
            for k, v in constants.items():
                if k in extra:
                    print(f"{txt.blue}{txt.bold}c.{k}{txt.end} is set but not used")
                elif k in expected:
                    # Excludes grid variables, which are managed internally by Starlord
                    print(f"{txt.blue}{txt.bold}c.{k}{txt.end} = {txt.blue}{v:.4n}{txt.end}")
            print("")
        return missing, extra

    def build_sampler(self, sampler_type: str, constants: dict = {}, **args):
        '''Construct an MCMC sampler for the model.

        Args:
            sampler_type: selects the sampler, should be "dynesty" or "emcee"
            constants: a dict of constant names and the values they should take

        Returns:
            A properly-initialized :class:`SamplerNested` if sampler_type is "dynesty"
            or a :class:`SamplerEnsemble` if it is "emcee"

        Raises:
            KeyError: if a required constant was not provided in constants
            ValueError: if the `sampler_type` was not one of "dynesty" or "emcee"
        '''
        self._resolve_grids()
        mod = self._gen.compile()
        missing, _ = self.validate_constants(constants, self._verbose)
        if missing:
            raise KeyError("Missing values for constant(s): " + ", ".join(missing))
        consts = []
        for c in self._gen.constants:
            if c[2:] not in self._gen.auto_constants.keys():
                consts.append(constants[str(c.name)])
        sampler_type = sampler_type.lower().strip()
        if sampler_type == "dynesty":
            return SamplerNested.create_from_module(mod, consts, **args)
        elif sampler_type == "emcee":
            return SamplerEnsemble.create_from_module(mod, consts, **args)
        raise ValueError(f"Sampler type '{sampler_type}' was not recognized.")
