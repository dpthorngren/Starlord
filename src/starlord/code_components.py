from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial

# The number of parameters for each type of distribution.
_num_params = {
    'normal': 2,
    'uniform': 2,
    'beta': 2,
    'gamma': 2,
    'exponential': 1,
    'trunc_power': 3,
    'trunc_normal': 4,
    'trunc_exponential': 3,
    'chabrier': 4,
    'chabrier_disk': 0,
    'chabrier_globular': 0,
    'chabrier_spheroid': 0,
    'binorm': 5,
    'casagrande_disk': 0,
    'apogee_dr17_afe': 1,
    'galah_dr4_afe': 1,
}

prefixes = {
    'log_': ('math.log10', "10**", "-math.log(10)-"),
    'exp10_': ('10**', 'math.log10', "-math.log(10)-"),
    'ln_': ('math.log', 'math.exp', "-"),
    'expn_': ('math.exp', 'math.log', "+"),
    'expit_': ('expit', 'logit', '-logddx_logit'),
    'logit_': ('logit', 'expit', "logddx_logit"),
}


def process_distribution(var: str | Symb, dist: str,
                         params: list[str | float | Symb]) -> tuple[Symb, str, list[str], set[Symb]]:
    '''Validates a distribution input and converts to the appropriate types.'''
    dist = dist.lower()
    assert dist in _num_params.keys(), f"Unrecognized distribution name '{dist}' for '{var}'."
    nparams = _num_params[dist]
    assert nparams == len(params), \
        f"Wrong number of parameters for distribution '{dist}', (expected {nparams}, got {len(params)})"
    # Prior Aliases
    if dist == 'chabrier_disk':
        params += [0.0, -1.10237, 0.69, 5.295945]
        dist = 'chabrier'
    elif dist == 'chabrier_globular':
        params += [-0.04575749, -0.48148, 0.34, 5.295945]
        dist = 'chabrier'
    elif dist == 'chabrier_spheroid':
        params += [-0.15490195, -0.65757, 0.33, 5.295945]
        dist = 'chabrier'
    elif dist == "casagrande_disk":
        params = [0.8, .016, -.15, 0.15, 0.22]
        dist = 'binorm'
    elif dist == 'apogee_dr17_afe':
        # Get [Fe/H] variable from input params, then generate output params
        x = params[0]
        params = [
            f"0.7693 - 0.571 * math.exp(-({x} + 0.8137)**2 / 0.10369458)",
            f"-.0052 + .286 * smootherstep({x}, 0.7, -2.1993)",
            f"-.0052 + .286 * smootherstep({x}, 0.6189, -0.7542)",
            f"0.024893 + 0.01672 * smootherstep({x}, 0.5, -2.)",
            f"0.024893 + 0.01672 * smootherstep({x}, 0.5, -2.)",
        ]
        dist = 'binorm'
    elif dist == "galah_dr4_afe":
        # Get [Fe/H] variable from input params, then generate output params
        x = params[0]
        params = [
            f"math.fmin(0.0305 - 0.2862*{x} + 0.186*{x}*{x}, 0.3987)",
            f"0.0378 + 0.02667 * smootherstep({x}, 0.3543, -1.0635)",
        ]
        dist = "normal"

    # Process parameters into strings, list requirements
    pars: list[str] = []
    reqs: set[Symb] = set()
    for p in params:
        p = str(p)
        template, vars = _extract_params(p)
        pars.append(template)
        reqs |= vars
    return Symb(var), dist, pars, reqs


def _extract_params(source: str) -> tuple[str, set[Symb]]:
    '''Extracts variables from the given string and replaces them with format brackets.
    Variables can be constants "c.name", parameters "p.name", or local variables "v.name".'''
    vars = set()
    replace_var = partial(_replace_var, vars=vars)
    template = re.sub(r"(?<!\w)([pcv]\.[A-Za-z_]\w*)", replace_var, source, flags=re.M)
    return template, vars


def _replace_var(source: re.Match, vars: set[Symb]) -> str:
    var = Symb(source.group())
    vars.add(var)
    return var.bracketed


class Symb(str):
    '''Represents a single symbol or constant in the code generator.'''

    def __new__(cls, source: str | float | int) -> Symb:
        try:
            value: float = float(source)
            return super().__new__(cls, str(value))
        except ValueError:
            if type(source) is str:
                source = source.strip("{ }").replace("-", "_")
                if re.fullmatch(r"[pcv]\.[A-Za-z_]\w*", source):
                    return super().__new__(cls, source)
            raise ValueError(f'Could not interpret "{source}" as a symbol or literal.') from None

    @property
    def name(self) -> str:
        return self[2:]

    @property
    def label(self) -> str:
        return self[0]

    @property
    def var(self) -> str:
        return f"{self.label}__{self.name}"

    @property
    def is_literal(self) -> bool:
        try:
            float(self)
            return True
        except ValueError:
            return False

    @property
    def bracketed(self) -> str:
        if self.is_literal:
            return str(self)
        return f"{{{self.label}__{self.name}}}"


@dataclass(frozen=True)
class Component:
    '''Represents a section of code for CodeGenerator.'''
    requires: set[Symb]
    provides: set[Symb]
    code: str

    def display(self) -> str:
        mapping = {s.var: str(s) for s in self.requires | self.provides}
        return self.code.format(**mapping) + " [Expr]"

    def generate_code(self) -> str:
        return self.code

    def __lt__(self, other) -> bool:
        return ", ".join(sorted(list(self.provides))) < ", ".join(sorted(list(other.provides)))


@dataclass(frozen=True)
class AssignmentComponent(Component):

    @classmethod
    def create(cls, var: Symb, expr: str, requires: set[Symb]):
        assert var.label == "v"
        return cls(requires, set([var]), expr)

    def display(self) -> str:
        mapping = {s.var: str(s) for s in self.requires | self.provides}
        return f"{list(self.provides)[0]} = {self.code.format(**mapping)}"

    def generate_code(self) -> str:
        code: str = f"{list(self.provides)[0].bracketed} = {self.code}"
        return code


@dataclass(frozen=True)
class DistributionComponent(Component):
    params: list[str]
    var: Symb

    @property
    def params_str(self) -> str:
        return ", ".join([p for p in self.params])

    @classmethod
    def create(cls, var: str | Symb, dist: str, params: list[str | float | Symb]):
        var, dist, pars, requires = process_distribution(var, dist, params)
        requires.add(var)
        return cls(requires, set(), dist, pars, var)

    def display(self) -> str:
        return f"{self.code.title()}({self.var} | {self.params_str})"

    def generate_code(self) -> str:
        return f"logL += {self.code}_lpdf({self.var.bracketed}, {self.params_str})"


@dataclass(frozen=True)
class Prior:
    vars: list[Symb]
    code_ppf: str
    code_pdf: str
    requires: set[Symb]
    params: list[str]
    distribution: str

    @property
    def provides(self) -> set[Symb]:
        return set(self.vars)

    @property
    def vars_str(self) -> str:
        return ", ".join([v.bracketed for v in self.vars])

    @property
    def params_str(self) -> str:
        return ", ".join([p for p in self.params])

    @classmethod
    def create(cls, var: str | Symb, dist: str, params: list[str | float | Symb]):
        # Check for transform prefixes
        for k, (fwd, inv, log_jac) in prefixes.items():
            if dist.startswith(k):
                dist_prefix = k
                dist = dist[len(k):]
                code_ppf = f"{{vars}} = {inv}({dist}_ppf({{vars}}, {{paramStr}}))"
                code_pdf = f"logP += {dist}_lpdf({fwd}({{vars}}), {{paramStr}}) + {log_jac}({{vars}})"
                break
        else:
            dist_prefix = ""
            code_ppf = f"{{vars}} = {dist}_ppf({{vars}}, {{paramStr}})"
            code_pdf = f"logP += {dist}_lpdf({{vars}}, {{paramStr}})"
        var, dist, pars, requires = process_distribution(var, dist, params)
        for req in requires:
            assert req.label in 'pc', f"Bad prior parameter '{req}'.  " + \
                "Prior parameters may only use constants or parameters, not variables."
        return Prior(
            vars=[var],
            code_ppf=code_ppf,
            code_pdf=code_pdf,
            requires=requires,
            params=pars,
            distribution=dist_prefix + dist,
        )

    def __lt__(self, other):
        return ", ".join(sorted(self.vars)) < ", ".join(sorted(other.vars))

    def display(self) -> str:
        mapping = {s.var: str(s) for s in self.requires | self.provides}
        return f"{self.distribution.title()}({', '.join(self.vars)} | {self.params_str.format(**mapping)})"

    def generate_ppf(self) -> str:
        return self.code_ppf.format(vars=self.vars_str, paramStr=self.params_str)

    def generate_pdf(self) -> str:
        return self.code_pdf.format(vars=self.vars_str, paramStr=self.params_str)
