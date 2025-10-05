from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
from importlib import util
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

from ._config import config
from .code_components import (AssignmentComponent, Component, DistributionComponent, Symb)


class Namespace(SimpleNamespace):
    '''A slightly less simple namespace, allowing for [] and iteration'''

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.items().__iter__()


class CodeGenerator:
    '''A class for generated log_likelihood, log_prior, and prior_ppf functions for use in MCMC fitting.'''

    _dynamic_modules_: dict = {}

    @property
    def variables(self):
        if self._vars_out_of_date:
            self._update_vars()
        return self._variables

    @property
    def params(self):
        if self._vars_out_of_date:
            self._update_vars()
        return tuple(self._params)

    @property
    def constants(self):
        if self._vars_out_of_date:
            self._update_vars()
        return tuple(self._constants)

    @property
    def blobs(self):
        if self._vars_out_of_date:
            self._update_vars()
        return tuple(self._blobs)

    @property
    def locals(self):
        if self._vars_out_of_date:
            self._update_vars()
        return tuple(self._locals)

    @property
    def arrays(self):
        if self._vars_out_of_date:
            self._update_vars()
        return tuple(self._arrays)

    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self._like_components = []
        self._prior_components = []
        # Lazily-updated property backers
        self._vars_out_of_date: bool = True
        self._variables: set[Symb] = set()
        self._params: list[Symb] = []
        self._constants: list[Symb] = []
        self._blobs: list[Symb] = []
        self._locals: list[Symb] = []
        self._arrays: list[Symb] = []

    def _update_vars(self):
        self._variables = set()
        result: dict[str, set[Symb]] = {i: set() for i in 'pcbla'}
        for comp in self._prior_components + self._like_components:
            for sym in comp.requires.union(comp.provides):
                assert sym.label in 'pcbla', f"Bad symbol name {sym}"
                result[sym.label].add(sym)
                self._variables.add(sym)
        self._params = sorted(list(result['p']))
        self._constants = sorted(list(result['c']))
        self._blobs = sorted(list(result['b']))
        self._locals = sorted(list(result['l']))
        self._arrays = sorted(list(result['a']))
        self._vars_out_of_date = False

    def get_mapping(self) -> dict[str, Namespace]:
        # TODO: Add options based on the type of output
        self._update_vars()
        mapping: dict[str, Namespace] = {}
        mapping['c'] = Namespace(**{c.name: c.var for c in self.constants})
        mapping['a'] = Namespace(**{a.name: a.var for a in self.arrays})
        mapping['l'] = Namespace(**{l.name: l.var for l in self.locals})
        mapping['p'] = Namespace(**{n.name: f"params[{i}]" for i, n in enumerate(self.params)})
        mapping['b'] = Namespace(**{n.name: f"blobs[{i}]" for i, n in enumerate(self.blobs)})
        return mapping

    def generate_prior_transform(self, prior_type: str = "ppf") -> str:
        mapping = self.get_mapping()
        result: list[str] = []
        result.append("cpdef double[:] prior_transform(double[:] params):")
        # TODO: Resolve prior dependencies
        for comp in self._prior_components:
            code: str = comp.generate_code(mapping, prior_type)
            result.append("\n".join("    " + l for l in code.splitlines()))
        result.append("    return params\n")
        return "\n".join(result)

    def generate_log_like(self) -> str:
        mapping = self.get_mapping()
        # Write the function header
        result: list[str] = []
        result.append("cpdef double log_like(double[:] params):")
        result.append("    cdef double logL = 0.")
        for _, loc in mapping['l']:
            result.append(f"    cdef {loc}")
        # Check that every local and blob used is initialized somewhere
        components = self._like_components.copy()
        initialized = set()
        for v in self.locals + self.blobs:
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
                    code: str = comp.generate_code(mapping)
                    result.append("\n".join("    " + l for l in code.splitlines()))
                    components.remove(comp)
                    initialized = initialized.union(comp.provides)
                    break
            else:
                raise LookupError("Circular dependencies in local / blob variables.")
        result.append("    return logL if math.isfinite(logL) else -math.INFINITY\n")
        return "\n".join(result)

    def generate(self, use_class: bool = False, prior_type: str = "ppf") -> str:
        # TODO: Other options
        if use_class:
            raise NotImplementedError
        if prior_type != "ppf":
            raise NotImplementedError
        result: list[str] = []
        result.append("from starlord.cy_tools cimport *\n")
        result.append(self.generate_log_like())
        result.append(self.generate_prior_transform())
        return "\n".join(result)

    def summary(self, code: bool = False, prior_type=None) -> str:
        result: list[str] = []
        result += ["=== Variables ==="]
        if self.params:
            result += ["Params:".ljust(12) + ", ".join([p[2:] for p in self.params])]
        if self.constants:
            result += ["Constants:".ljust(12) + ", ".join([c[2:] for c in self.constants])]
        if self.blobs:
            result += ["Blobs:".ljust(12) + ", ".join([b[2:] for b in self.blobs])]
        if self.locals:
            result += ["Locals:".ljust(12) + ", ".join([l[2:] for l in self.locals])]
        if self.arrays:
            result += ["Arrays:".ljust(12) + ", ".join([a[2:] for a in self.arrays])]
        result += ["=== Likelihood ==="]
        result += [i.generate_code() if code else str(i) for i in self._like_components]
        result += ["=== Prior ==="]
        for c in self._prior_components:
            if code:
                result += [c.generate_code(prior_type=prior_type)]
            elif type(c) is DistributionComponent:
                result += [f"p({c.var}) = {c}"]
            else:
                result += [str(c)]
        return "\n".join(result)

    def expression(self, expr: str) -> None:
        '''Specify a general expression to add to the code.  Assignments and variables used will be
        automatically detected so long as they are formatted properly (see CodeGenerator doc)'''
        provides = set()
        # Finds assignment blocks like "l.foo = " and "l.bar, l.foo = "
        assigns = re.findall(r"^\s*[pcbla]\.[A-Za-z_]\w*\s*(?:,\s*[pcbla]\.[A-Za-z_]\w*)*\s*=(?!=)", expr, flags=re.M)
        assigns += re.findall(
            r"^\s*\(\s*[pcbla]\.[A-Za-z_]\w*\s*(?:,\s*[pcbla]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr, flags=re.M)
        # Same as above but covers when vars are enclosed by parentheses like "(l.a, l.b) ="
        assigns += re.findall(
            r"^\s*\(\s*[pcbla]\.[A-Za-z_]\w*\s*(?:,\s*[pcbla]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr, flags=re.M)
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
        requires = variables - provides
        self._like_components.append(Component(requires, provides, code))

    def assign(self, var: str, expr: str) -> None:
        # If l or b is omitted, l is implied
        var = Symb(var if re.match(r"^[bl]\.", var) is not None else f"l.{var}")
        code, variables = self._extract_params_(expr)
        comp = AssignmentComponent(var, code, variables-{var})
        self._like_components.append(comp)

    def constraint(self, var: str, dist: str, params: list[str], is_prior=False):
        var = Symb(var)
        assert len(params) == 2
        pars: list[Symb] = [Symb(i) for i in params]
        comp = DistributionComponent(var, dist, pars)
        if is_prior:
            self._prior_components.append(comp)
        else:
            self._like_components.append(comp)

    @staticmethod
    def _extract_params_(source: str) -> tuple[str, set[Symb]]:
        '''Extracts variables from the given string and replaces them with format brackets.
        Variables can be constants "c.name", blobs "b.name", parameters "p.name", or local variables "l.name".'''
        template: str = re.sub(r"(?<!\w)([pcbla]\.[A-Za-z_]\w*)", r"{\1}", source, flags=re.M)
        all_vars: list[str] = re.findall(r"(?<=\{)[pcbla]\.[A-Za-z_]\w*(?=\})", template, flags=re.M)
        variables: set[Symb] = {Symb(v) for v in all_vars}
        return template, variables

    @staticmethod
    def _compile_to_module(code: str) -> str:
        # Get the code hash for file lookup
        hasher = hashlib.shake_128(code.encode())
        hash = base64.b32encode(hasher.digest(25)).decode("utf-8")
        name = f"sl_gen_{hash}"
        pyxfile = config.cache_dir / (name+".pyx")
        # Clean up old cached files
        # TODO: If temp files exceeds 100, delete anything not accessed within a week
        # 		path.stat.st_atime # Verify that this works before using!
        # 		Don't delete the requested file or a file in use (how to track?)
        # Write the pyx file if needed
        if not pyxfile.exists():
            with pyxfile.open("w") as pxfh:
                pxfh.write(code)
                pxfh.close()
            assert pyxfile.exists(), "Wrote the code to a file, but the file still doesn't exist."
        libfiles = list(config.cache_dir.glob(name + ".*.*"))
        if len(libfiles) == 0:
            os.system(f"cythonize -f -i {pyxfile}")
            cfile = config.cache_dir / (name+".c")
            libfiles = list(config.cache_dir.glob(name + ".*.*"))
            assert len(libfiles) >= 1, "Compiled but failed to produce an object file to import."
            # Remove the (surprisingly large) build c file artifact
            assert cfile.exists()
            cfile.unlink()
            builddir = config.cache_dir / "build"
            # Remove the build directory -- the output was moved to cache_dir automatically
            assert builddir.exists()
            shutil.rmtree(builddir)
        return hash

    @staticmethod
    def _load_module(hash: str):
        if hash in CodeGenerator._dynamic_modules_.keys():
            return CodeGenerator._dynamic_modules_[hash]
        name = f"sl_gen_{hash}"
        libfiles = list(config.cache_dir.glob(name + ".*.*"))
        assert len(libfiles) > 0, f"Could not find module with hash {hash}"
        assert len(libfiles) == 1, f"Unexpected files in the cache directory: {libfiles}"
        libfile = libfiles[0]
        assert libfile.suffix in [
            ".so", ".dll", ".dynlib", ".sl"
        ], f"Compiled module format {libfile.suffix} unrecognized."
        spec: ModuleSpec | None = util.spec_from_file_location(f"{name}", f"{libfile}")
        assert spec is not None, f"Couldn't load the module specs from file {libfile}"
        dynmod = util.module_from_spec(spec)
        assert spec.loader is not None, f"Couldn't load the module from file {libfile}"
        spec.loader.exec_module(dynmod)
        CodeGenerator._dynamic_modules_[hash] = dynmod
        return dynmod
