from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


class Symb(str):

    def __new__(cls, source: str) -> Symb:
        if re.fullmatch(r"[pcbla]\.[A-Za-z_]\w*", source) is not None:
            return super().__new__(cls, source)
        try:
            value: float = float(source)
        except ValueError:
            raise ValueError(f'Could not interpret "{source}" as a symbol or literal.') from None
        return super().__new__(cls, str(value))

    @property
    def name(self) -> str:
        return self[2:]

    @property
    def label(self) -> str:
        return self[0]

    @property
    def var(self) -> str:
        return self.label + "_" + self.name

    def is_literal(self) -> bool:
        try:
            float(self)
            return True
        except:
            return False


@dataclass(frozen=True)
class Component:
    '''Represents a section of code for CodeGenerator.'''
    requires: set[Symb]
    provides: set[Symb]
    code: str

    def __repr__(self) -> str:
        return f"ExprComponent({', '.join(self.requires)}) -> ({', '.join(self.provides)})"

    def generate_code(self, name_map: Optional[dict] = None, prior_type: Optional[str] = None) -> str:
        if name_map is None:
            return self.code
        else:
            return self.code.format_map(name_map)


@dataclass(frozen=True)
class AssignmentComponent(Component):
    def __repr__(self) -> str:
        return f"{list(self.requires)[0]} = {self.code}"


@dataclass(frozen=True)
class DistributionComponent(Component):
    params: list[str]
    var: Symb

    def __init__(self, var: Symb, dist: str, params: list[Symb]):
        assert dist in ["normal", "uniform", "beta", "gamma"]
        # Must use object.__setattr__ to init because the type is frozen
        object.__setattr__(self, 'provides', set())
        object.__setattr__(self, 'requires', set(p for p in params if not p.is_literal()))
        object.__setattr__(self, 'code', dist)
        object.__setattr__(self, 'params', [str(p) if p.is_literal() else f"{{{p}}}" for p in params])
        object.__setattr__(self, 'var', var)

    def __repr__(self) -> str:
        return f"{self.code}({self.var} | {', '.join(self.params)})"

    def generate_code(self, name_map: Optional[dict] = None, prior_type: Optional[str] = None) -> str:
        if prior_type is None:
            result = f"logL += {self.code}_lpdf({{{self.var}}}, {', '.join(self.params)})"
        elif prior_type == "pdf":
            result = f"logP += {self.code}_lpdf({{{self.var}}}, {', '.join(self.params)})"
        elif prior_type == "ppf":
            result = f"{{{self.var}}} = {self.code}_ppf({{{self.var}}}, {', '.join(self.params)})"
        else:
            raise ValueError(f"Unrecognized prior option {prior_type} -- must be None, 'ppf', or 'pdf'.")
        if name_map is None:
            return result
        else:
            return result.format_map(name_map)


class InterpolateComponent(Component):
    pass
