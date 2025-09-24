from __future__ import annotations

import re

from .code_components import Component, Symb


class CodeGenerator:
    '''A class for generated log_likelihood, log_prior, and prior_ppf functions for use in MCMC fitting.'''

    def __init__(self, verbose: bool = False):
        self._like_components = []
        self._prior_components = []
        self.verbose = verbose

    def get_variables(self, filter_label: str = "pcbla", prior: bool = False) -> set[Symb]:
        result: set[Symb] = set()
        target: list[Component] = self._prior_components if prior else self._like_components
        for comp in target:
            for sym in comp.requires.union(comp.provides):
                if sym.label in filter_label:
                    result.add(sym)
        return result

    def summary(self, code: bool = False) -> str:
        result: list[str] = []
        result += ["=== Variables ==="]
        for label in ["Params", "Constants", "Blobs", "Locals", "Arrays"]:
            variables: set[Symb] = self.get_variables(label[0].lower())
            if len(variables) > 0:
                result += [label + ": " + ", ".join(variables)]
        result += ["=== Likelihood ==="]
        result += [i.generate_code() if code else str(i) for i in self._like_components]
        result += ["=== Prior ==="]
        result += [i.generate_code() if code else str(i) for i in self._prior_components]
        return "\n".join(result)

    def expression(self, expr: str) -> None:
        '''Specify a general expression to add to the code.  Assignments and variables used will be
        automatically detected so long as they are formatted properly (see CodeGenerator doc)'''
        provides = set()
        # Finds assignment blocks like "l.foo = " and "l.bar, l.foo = "
        assigns = re.findall(r"^\s*[pcbla]\.[A-Za-z_]\w*\s*(?:,\s*[pcbla]\.[A-Za-z_]\w*)*\s*=(?!=)", expr)
        assigns += re.findall(r"^\s*\(\s*[pcbla]\.[A-Za-z_]\w*\s*(?:,\s*[pcbla]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr)
        # Same as above but covers when vars are enclosed by parentheses like "(l.a, l.b) ="
        assigns += re.findall(r"^\s*\(\s*[pcbla]\.[A-Za-z_]\w*\s*(?:,\s*[pcbla]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr)
        for block in assigns:
            # Handles parens, multiple assignments, extra whitespace, and removes the "="
            block = block[:-1].strip(" ()")
            # Block now looks like "l.foo" or "l.foo, l.bar"
            for var in block.split(","):
                var = var.strip()
                # Verify that the result is a local or blob formatted as "l.foo" or "b.bar"
                assert var[0] in "lb" and var[1] == ".", var
                provides.add(Symb(var))
        code, variables = self._extract_params_(expr)
        requires = {Symb(i) for i in variables} - provides
        self._like_components.append(Component(requires, provides, code))

    def assign(self, var: str, expr: str) -> None:
        # If l or b omitted, add l.
        if self.verbose:
            print("        Gen TODO: variable assignment")

    def constraint(self, var: str, dist: str, params: list[str]):
        var = Symb(var)
        # TODO: Check dist name
        params = [Symb(str(i)) for i in params]
        if self.verbose:
            print("        Gen TODO: Constraint")

    def prior(self, var: str, dist: str, params: list[str]):
        if self.verbose:
            print("        Gen TODO: prior assignment")

    def _add_component(self, req: set[Symb], prov: set[Symb], params: list[Symb], template: str, prior: bool) -> None:
        new_comp: Component = Component(req, prov, template)
        if prior:
            self._prior_components.append(new_comp)
        else:
            self._like_components.append(new_comp)

    # def normal(self, val: str, mean: str, std: str, prior: bool = False) -> None:
    #     # TODO: Account for val
    #     _mean = Symb(mean)
    #     _std = Symb(std)
    #     template: str = f"NORMAL({{{_mean}}}, {{{_std}}})"
    #     self._add_component({_mean, _std}, set(), [_mean, _std], template, prior)

    @staticmethod
    def _extract_params_(source: str) -> tuple[str, set[str]]:
        '''Extracts variables from the given string and replaces them with format brackets.
        Variables can be constants "c.name", blobs "b.name", parameters "p.name", or local variables "l.name".'''
        template: str = re.sub(r"(?<=[\W])([pcbl])\.(([A-Za-z_]\w*))", r"{\1_\2}", source)
        variables: set[str] = set(re.findall(r"(?<=\{)[pcbl]_[A-Za-z_]\w*(?=\})", template))
        return template, variables
