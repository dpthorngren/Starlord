from __future__ import annotations

import re

from .code_components import Symb, Component


class CodeGenerator:
    def __init__(self):
        self._like_components = []
        self._prior_components = []

    def summary(self):
        result = ["Likelihood:"]
        for i in self._like_components:
            result += [str(i)]
        result += ["Prior:"]
        for i in self._prior_components:
            result += [str(i)]
        return "\n".join(result)

    def _add_component(self, req: set[Symb], prov: set[Symb], params: list[Symb], template: str, prior: bool) -> None:
        new_comp: Component = Component(req, prov, template)
        if prior:
            self._prior_components.append(new_comp)
        else:
            self._like_components.append(new_comp)

    def normal(self, val: str, mean: str, std: str, prior: bool = False) -> None:
        # TODO: Account for val
        _mean = Symb(mean)
        _std = Symb(std)
        template: str = f"NORMAL({{{_mean}}}, {{{_std}}})"
        self._add_component({_mean, _std}, set(), [_mean, _std], template, prior)

    @staticmethod
    def _extract_params_(source: str) -> tuple[str, set[str]]:
        '''Extracts variables from the given string and replaces them with format brackets.
        Variables can be constants "c.name", blobs "b.name", parameters "p.name", or local variables "l.name".'''
        template: str = re.sub(r"(?<=[\W])([pcbl])\.(([A-Za-z_]\w*))", r"{\1_\2}", source)
        variables: set[str] = set(re.findall(r"(?<=\{)[pcbl]_[A-Za-z_]\w*(?=\})", template))
        return template, variables
