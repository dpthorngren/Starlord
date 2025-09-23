from __future__ import annotations

import re
from dataclasses import dataclass


class Symb(str):
    # TODO: Add Symb literal
    def __new__(cls, source: str) -> Symb:
        assert re.fullmatch(r"[pcbla][_.][A-Za-z_]\w*", source) is not None
        source = source.replace(".", "_")
        output: Symb = super().__new__(cls, source)
        return output

    @property
    def name(self) -> str:
        return self[2:]

    @property
    def label(self) -> str:
        return self[0]


@dataclass(frozen=True)
class Component:
    '''Represents a section of code for CodeGenerator.'''
    requires: set[Symb]
    provides: set[Symb]
    code: str

    def __repr__(self) -> str:
        return f"ExprComponent({self.requires}) -> ({self.provides})"

    def generate_code(self, name_map: dict) -> str:
        return self.code.format_map(name_map)


class AssignmentComponent(Component):
    pass


class DistributionComponent(Component):
    pass


class InterpolateComponent(Component):
    pass
