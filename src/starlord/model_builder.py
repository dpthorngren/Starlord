from __future__ import annotations

import re
from functools import partial
from typing import List, Optional, Tuple

from ._config import _TextFormatCodes_, config
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
    :Grid Variables: ``d.[grid_name].[output_name]``, these indicate the grid
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

    @property
    def code_generator(self) -> CodeGenerator:
        if self.__gen__ is None:
            self.__grids__ = {}
            deferred_map = self._resolve_deferred()
            if self.verbose:
                print(f"\n    {self.txt.underline}Code Generation{self.txt.end}")
            self.__gen__ = CodeGenerator(self.verbose)
            for deferred_vars, expr in self._expressions:
                assert all([i in deferred_map.keys() for i in deferred_vars])
                self.__gen__.expression(expr.format_map(deferred_map))
            for deferred_vars, var, expr in self._assignments + self.__assignments_gen__:
                assert all([i in deferred_map.keys() for i in deferred_vars])
                self.__gen__.assign(var.format_map(deferred_map), expr.format_map(deferred_map))
            for deferred_vars, var, dist, params in self._constraints + self.__constraints_gen__:
                assert all([i in deferred_map.keys() for i in deferred_vars])
                self.__gen__.constraint(var.format_map(deferred_map), dist.format_map(deferred_map), params)
            for param, dist, params in self._priors:
                self.__gen__.prior(param, dist, params)
            self.__gen__.auto_constants = self._auto_constants.copy()
            self.__gen__.constant_types = self._constant_types.copy()
            print("")
        return self.__gen__

    @property
    def txt(self) -> _TextFormatCodes_:
        if self.fancy_text:
            return config.text_format
        return config.text_format_off

    def __init__(self, verbose: bool = False, fancy_text: bool = True):
        '''
        Args:
            verbose: If True, print extra debugging info
            fancy_text: If True, color and style terminal output text
        '''
        self.verbose: bool = verbose
        self.fancy_text: bool = fancy_text
        self.__gen__: Optional[CodeGenerator] = None
        self.__grids__: dict[str, list[str]] = {}
        self.user_mappings: dict[str, str] = {}
        # Data for the CodeGenerator setup, formatted as ([deferred vars], arguments...)
        # TODO: Namedtuples instead?
        self._expressions: List[Tuple[List[str], str]] = []
        self._assignments: List[Tuple[List[str], str, str]] = []
        self._constraints: List[Tuple[List[str], str, str, List[str | float]]] = []
        # Generated variables
        self.__auto_generating__ = False
        self.__assignments_gen__: List[Tuple[List[str], str, str]] = []
        self.__constraints_gen__: List[Tuple[List[str], str, str, List[str | float]]] = []
        # Priors do not have deferred_vars, so they're just (var, dist, params)
        self._priors: List[Tuple[str, str, List[str | float]]] = []
        # Lists to be passed directly to CodeGenerator
        self._auto_constants: dict[str, str] = {}
        self._constant_types: dict[str, str] = {}

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
        if self.verbose:
            print(f"    {self.txt.underline}Model Processing{self.txt.end}")
        if "expr" in model.keys():
            for name, code in model['expr'].items():
                if self.verbose:
                    print(f"expr.{name} = '{code}'")
                self.expression(code)
        if "var" in model.keys():
            for key, value in model['var'].items():
                if self.verbose:
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
                if self.verbose:
                    print(f"prior.{key} = {value}")
                self._unpack_distribution("p." + key, value, True)
        for grid in GridGenerator.grids().keys():
            if grid in model.keys():
                for key, value in model[grid].items():
                    assert len(value) in [2, 3]
                    if self.verbose:
                        print(f"d.{grid}.{key} = {value}")
                    self._unpack_distribution(f"d.{grid}.{key}", value)
        if "override" in model.keys():
            for key, override in model['override'].items():
                if self.verbose:
                    print(f"override.{key} = {override}")
                if type(override) is dict:
                    for input_name, value in override.items():
                        self.override_mapping(f"{key}.{input_name}", value)
                else:
                    assert type(override) is str
                    self.override_mapping(key, override)

    def override_mapping(self, key: str, value: str):
        '''Sets the value or symbol to use a deferred variable, often grid variables.

        This can be used to fix grid axes to a particular value, or make them depend on some
        additional grid output or calculation.  Grid inputs are set by default according to
        the their entry in the `input_mappings` grid metadata.  If there is no entry then they
        default to being a parameter named "p.{input_name}".

        Args:
            key: The deferred variable key, e.g. "grid.output_var" or "nongrid_var".
            value: What to set the variable to wherever it appears.

        Examples:
            Suppose you are fitting a stellar model and wish to lock the metallicity to solar.
            If you're using the mist grid, you could do this with::

                builder.override_input("mist.feh", "0")

            In the same circumstance, if you wanted to set logG to 2% higher than
            what the evolution tracks output (as a sensitivity test, perhaps), you could use::

                builder.override_input("mist.logG", "1.02*d.mistTracks.logG")

            Note that this uses another grid via a deferred variable.  Starlord detectrs this
            via the "d." prefix.  In fact, the default input refers to d.mistTracks.logG already.
        '''
        if self.verbose:
            print(f"  ModelBuilder.override_input('{key}', '{value}')")
        assert re.fullmatch(r"[a-zA-Z]\w*(\.[a-zA-Z]\w*)?", key), key
        if "." in key:
            grid_name, input_name = key.split(".")
            grid = GridGenerator.get_grid(grid_name)
            assert input_name in grid.inputs, f"Cannot override nonexistent input {input_name}"
            key = f"{grid_name}__{input_name}"
        self._gen = None
        _, value = self._extract_deferred(value)
        self.user_mappings[key] = value

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
        if self.verbose:
            expr_str = expr[50:] + "..." if len(expr) > 50 else expr
            print(f"  ModelBuilder.expression('{expr_str}')")
        # Switch any tabs out for spaces and process any grids
        expr = expr.replace("\t", "    ")
        deferred_vars, expr = self._extract_deferred(expr)
        self._gen = None
        self._expressions.append((deferred_vars, expr))

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
        if self.verbose:
            print(f"  ModelBuilder.assignment('{var}', {expr})")
        assert re.fullmatch(r"({\w+}|(l\.)?[a-zA-Z]\w*)", var), var
        # l is implied if it is omitted.
        if not (var.startswith("l.") or var.startswith("{")):
            var = f"l.{var}"
        deferred_vars, expr = self._extract_deferred(expr)
        self._gen = None
        if self.__auto_generating__:
            self.__assignments_gen__.append((deferred_vars, var, expr))
        else:
            self._assignments.append((deferred_vars, var, expr))

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
        if self.verbose:
            print(f"  ModelBuilder.constraint('{var}', '{dist}', {params})")
        deferred_vars, var = self._extract_deferred(var)
        assert re.fullmatch(r"({\w+}|[pcl]\.[a-zA-Z]\w*)", var), f'Bad variable name {var}.'
        self._gen = None
        if self.__auto_generating__:
            self.__constraints_gen__.append((deferred_vars, var, dist, params))
        else:
            self._constraints.append((deferred_vars, var, dist, params))

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
        if self.verbose:
            print(f"  ModelBuilder.prior('{param}', '{dist}', {params})")
        self._gen = None
        self._priors.append((param, dist, params))

    def summary(self) -> str:
        '''Generates a summary of the model currently defined.

        The model does not need to be in a finalized state to be run, so it may help
        to check this periodically as you build the model.

        Returns:
            The model summary.
        '''
        summary_text = self.code_generator.summary(self.fancy_text)
        result = [f"    {self.txt.underline}Grids{self.txt.end}"]
        if self.__grids__:
            for k, v in sorted(self.__grids__.items(), key=lambda g: g[0]):
                result.append(k + " " + ", ".join(sorted(v)))
        else:
            result.append("None")
        return "\n".join(result) + "\n\n" + summary_text

    def generate_code(self) -> str:
        '''Generates the code for the model.

        Returns:
            A string containing the generated Cython code.

        Raises:
            AssertionError: if one of the various consistency checks fails.
        '''
        return self.code_generator.generate()

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

    @staticmethod
    def _extract_deferred(source: str) -> Tuple[List[str], str]:
        '''Extracts grid names from the source string and replaces them with deferred variables.'''
        # Identifies deferred variables of the form "d.foo.bar"
        vars = []
        replace_grids = partial(ModelBuilder._replace_grid_name, accum=vars)
        source = re.sub(r"d\.([a-zA-Z_]\w+)(?:\.([a-zA-Z1-9]\w*))?", replace_grids, source)
        return vars, source

    @staticmethod
    def _replace_grid_name(match, accum):
        label, name = match.groups()
        if name is not None:
            assert label in GridGenerator.grids().keys()
            var = f"{label}__{name}"
            accum.append(var)
            return f"{{{var}}}"
        else:
            accum.append(label)
            return f"{{{label}}}"

    def _resolve_deferred(self) -> dict[str, str]:
        # TODO: Docs
        if self.verbose:
            print(f"\n    {self.txt.underline}Variable Resolution{self.txt.end}")

        # Setup the recursive resolver
        deferred_mappings = dict()
        stack = []

        def recursive_resolve(dvar):
            if type(dvar) is re.Match:
                dvar = dvar.group(1)
            # If symbol in deferred_mappings, just return that
            if dvar in deferred_mappings.keys():
                return deferred_mappings[dvar]

            # Detect circular definitions
            assert dvar not in stack, f"The definition of {dvar} is circular."
            stack.append(dvar)

            # Get the value to sub in for symbol
            if dvar in self.user_mappings:
                # User-specified mapping for the variable takes priority
                value = self.user_mappings[dvar]
                _, value = self._extract_deferred(value)
                value = re.sub(r"{(\w+(?:\.\w+)?)}", recursive_resolve, value)
            else:
                # Only remaining valid option is a grid variable
                grid_name, key = dvar.split('__')
                grid = GridGenerator.get_grid(grid_name)

                if key in grid.inputs:
                    value = grid._get_input_map()[key]
                    _, value = self._extract_deferred(value)
                    value = re.sub(r"{(\w+(?:\.\w+)?)}", recursive_resolve, value)
                elif key in grid.outputs:
                    grid_var = f"grid__{grid_name}__{key}"
                    param_string = ", ".join([f"{{{grid_name}__{i}}}" for i in grid.inputs])
                    code = f"c.{grid_var}._interp{grid.ndim}d({param_string})"
                    code = re.sub(r"{(\w+(?:\.\w+)?)}", recursive_resolve, code)
                    try:
                        self.__auto_generating__ = True
                        self.assign(f"{grid_name}__{key}", code)
                        self._auto_constants[grid_var] = f"GridGenerator.get_grid('{grid_name}').build_grid('{key}')"
                        self._constant_types[grid_var] = "GridInterpolator"
                    finally:
                        self.__auto_generating__ = False
                    value = f"l.{grid_name}__{key}"
                elif key in grid.derived:
                    _, code = self._extract_deferred(grid.derived[key])
                    code = re.sub(r"{(\w+(?:\.\w+)?)}", recursive_resolve, code)
                    try:
                        self.__auto_generating__ = True
                        self.assign(f"l.{grid_name}__{key}", code)
                    finally:
                        self.__auto_generating__ = False
                    value = f"l.{grid_name}__{key}"
                else:
                    raise ValueError(f"Key {key} not in grid {grid_name}.")
                self.__grids__.setdefault(grid_name, []).append(key)

            if self.verbose:
                print("d." + dvar.replace("__", ".").ljust(30), value)

            # Value is now fully resolved, so record and return it.
            deferred_mappings[dvar] = value
            stack.remove(dvar)
            return value

        # Collect base list of deferred variables
        all_deferred: set[str] = set()
        all_deferred = all_deferred.union(*[i[0] for i in self._expressions])
        all_deferred = all_deferred.union(*[i[0] for i in self._assignments])
        all_deferred = all_deferred.union(*[i[0] for i in self._constraints])

        while True:
            unresolved = all_deferred - set(deferred_mappings.keys())
            if len(unresolved) == 0:
                break
            recursive_resolve(unresolved.pop())

        return deferred_mappings

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
        expected = {c.name for c in self.code_generator.constants}
        missing = expected - set(constants.keys())
        missing -= set(self.code_generator.auto_constants.keys())
        extra = set(constants.keys()) - expected
        if print_summary:
            print(f"\n    {self.txt.underline}Constant Values{self.txt.end}")
            if not missing and not constants.items():
                print("[None]")
            for k in missing:
                print(f"{self.txt.blue}{self.txt.bold}c.{k}{self.txt.end} is not set")
            for k, v in constants.items():
                if k in extra:
                    print(f"{self.txt.blue}{self.txt.bold}c.{k}{self.txt.end} is set but not used")
                elif k in expected:
                    # Excludes grid variables, which are managed internally by Starlord
                    print(f"{self.txt.blue}{self.txt.bold}c.{k}{self.txt.end} = {self.txt.blue}{v:.4n}{self.txt.end}")
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
        mod = self.code_generator.compile()
        missing, _ = self.validate_constants(constants, self.verbose)
        if missing:
            raise KeyError("Missing values for constant(s): " + ", ".join(missing))
        consts = []
        for c in self.code_generator.constants:
            if c[2:] not in self.code_generator.auto_constants.keys():
                consts.append(constants[str(c[2:])])
        sampler_type = sampler_type.lower().strip()
        if sampler_type == "dynesty":
            return SamplerNested.create_from_module(mod, consts, **args)
        elif sampler_type == "emcee":
            return SamplerEnsemble.create_from_module(mod, consts, **args)
        raise ValueError(f"Sampler type '{sampler_type}' was not recognized.")
