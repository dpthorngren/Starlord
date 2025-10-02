from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Union


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


@dataclass(frozen=True)
class Component:
    '''Represents a section of code for CodeGenerator.'''
    requires: set[Symb]
    provides: set[Symb]
    code: str

    def __repr__(self) -> str:
        return f"ExprComponent({', '.join(self.requires)}) -> ({', '.join(self.provides)})"

    def generate_code(self, name_map: Union[dict, None] = None) -> str:
        if name_map is None:
            return self.code
        else:
            return self.code.format_map(name_map)


class AssignmentComponent(Component):
    def __repr__(self) -> str:
        return f"{list(self.requires)[0]} = {self.code}"


class DistributionComponent(Component):
    pass


class InterpolateComponent(Component):
    pass
