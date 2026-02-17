from __future__ import annotations

import re
from dataclasses import dataclass


class Symb(str):
    '''Represents a single symbol or constant in the code generator.'''

    def __new__(cls, source: str | float | int) -> Symb:
        try:
            value: float = float(source)
            return super().__new__(cls, str(value))
        except ValueError:
            if type(source) is str:
                source = source.strip("{ }").replace("-", "_")
                if re.fullmatch(r"[pcl]\.[A-Za-z_]\w*", source):
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
        mapping = {s.var: str(s) for s in self.requires.union(self.provides)}
        return self.code.format(**mapping) + " [Expr]"

    def generate_code(self) -> str:
        return self.code


@dataclass(frozen=True)
class AssignmentComponent(Component):

    @classmethod
    def create(cls, var: Symb, expr: str, requires: set[Symb]):
        assert var.label in "lb"
        return cls(requires, set([var]), expr)

    def display(self) -> str:
        mapping = {s.var: str(s) for s in self.requires.union(self.provides)}
        return f"{list(self.provides)[0]} = {self.code.format(**mapping)}"

    def generate_code(self) -> str:
        code: str = f"{list(self.provides)[0].bracketed} = {self.code}"
        return code


@dataclass(frozen=True)
class DistributionComponent(Component):
    params: list[str]
    var: Symb

    @classmethod
    def create(cls, var: Symb, dist: str, params: list[Symb]):
        dist = dist.lower()
        assert dist in ["normal", "uniform", "beta", "gamma"]
        requires: set[Symb] = set(p for p in params if not p.is_literal)
        requires = requires | {var}
        pars = [str(p) if p.is_literal else f"{{{p}}}" for p in params]
        return cls(requires, set(), dist, pars, var)

    def display(self) -> str:
        params = ", ".join([p for p in self.params])
        return f"{self.code.title()}({self.var} | {params})"

    def generate_code(self) -> str:
        params = ", ".join([Symb(p).bracketed for p in self.params])
        return f"logL += {self.code}_lpdf({self.var.bracketed}, {params})"


@dataclass(frozen=True)
class Prior:
    vars: list[Symb]
    code_ppf: str
    code_pdf: str
    params: list[Symb]
    distribution: str

    @property
    def requires(self) -> set[Symb]:
        return set()

    @property
    def provides(self) -> set[Symb]:
        return set(self.vars)

    @classmethod
    def create(cls, var: Symb, distribution: str, params: list[Symb]):
        distribution = distribution.lower()
        assert distribution in ["normal", "uniform", "beta", "gamma"]
        return Prior(
            vars=[var],
            code_ppf="{vars} = " + distribution + "_ppf({vars}, {paramStr})",
            code_pdf="logP += " + distribution + "_lpdf({vars}, {paramStr})",
            params=params,
            distribution=distribution,
        )

    def display(self) -> str:
        params = ", ".join([p for p in self.params])
        vars = ", ".join([v for v in self.vars])
        return f"{self.distribution.title()}({vars} | {params})"

    def generate_ppf(self) -> str:
        vars = [v.bracketed for v in self.vars]
        params = [p.bracketed for p in self.params]
        return self.code_ppf.format(vars=", ".join(vars), params=params, paramStr=", ".join(params))

    def generate_pdf(self) -> str:
        vars = [v.bracketed for v in self.vars]
        fmt = dict(vars=", ".join(vars), params=self.params, paramStr=", ".join(self.params))
        return self.code_pdf.format(**fmt)
