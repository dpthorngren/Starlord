from __future__ import annotations

import re

from .code_components import Component, Symb


class CodeGenerator:
    '''A class for generated log_likelihood, log_prior, and prior_ppf functions for use in MCMC fitting.'''

    def __init__(self, verbose: bool = False):
        self._like_components = []
        self._prior_components = []
        self.verbose = verbose

    def get_variables(self, prior: bool = False) -> dict[str, set[Symb]]:
        result: dict[str, set[Symb]] = {i: set() for i in 'pcbla'}
        target: list[Component] = self._prior_components if prior else self._like_components
        for comp in target:
            for sym in comp.requires.union(comp.provides):
                assert sym.label in 'pcbla'
                result[sym.label].add(sym)
        return result

    def generate_log_like(self) -> str:
        # Get all variables
        variables = self.get_variables()
        mapping: dict[str, str] = {c: c for c in variables['c']}
        mapping.update({a: a for a in variables['a']})
        mapping.update({l: l for l in variables['l']})
        # Set the sorted order of indexed variables
        params: list[str] = sorted(list(variables['p']))
        mapping.update({name: f"params[{i}]" for i, name in enumerate(params)})
        blobs: list[str] = sorted(list(variables['b']))
        mapping.update({name: f"blobs[{i}]" for i, name in enumerate(blobs)})
        # Write the function header 
        result = []
        result.append("@nb.njit")
        result.append("def log_like(params):")
        result.append("    logL = 0.")
        # Check that every variable used is initialized somewhere
        result: list[str] = []
        components = self._like_components.copy()
        initialized = set()
        for v in variables['l'].union(variables['b']):
            for comp in components:
                if v in comp.provides:
                    break
            else:
                raise LookupError(f"Variable {v} is used but never initialized.")
        # Call components according to their initialization requirements
        while len(components) > 0:
            for comp in components:
                reqs = {c for c in comp.requires if c[0] in "bl" and c not in initialized}
                if len(reqs) == 0:
                    result.append("    " + comp.generate_code(mapping))
                    components.remove(comp)
                    initialized = initialized.union(comp.provides)
                    break
            else:
                raise LookupError("Circular dependencies in local / blob variables.")
        result.append("    return logL if np.isfinite(logL) else -np.inf")
        return "\n".join(result)

    def summary(self, code: bool = False) -> str:
        result: list[str] = []
        result += ["=== Variables ==="]
        variables = self.get_variables()
        for label in ["Params", "Constants", "Blobs", "Locals", "Arrays"]:
            key = label[0].lower()
            if len(variables[key]) > 0:
                result += [(label + ": ").ljust(12) + ", ".join(variables[key])]
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
