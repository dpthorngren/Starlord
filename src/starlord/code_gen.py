from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import sys
import time
from functools import partial
from importlib import util
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import NamedTuple, Optional

import cython

from ._config import __version__, _TextFormatCodes_, config
from .code_components import (AssignmentComponent, Component, DistributionComponent, Prior, Symb)

_VarCache = NamedTuple(
    'VarCache', [('p', tuple[Symb]), ('c', tuple[Symb]), ('l', tuple[Symb]), ('map', dict[str, str])])


class CodeGenerator:
    '''A class for generated log_likelihood, log_prior, and prior_ppf functions for use in MCMC fitting.'''

    _dynamic_modules_: dict = {}

    @property
    def txt(self) -> _TextFormatCodes_:
        if self.fancy_text:
            return config.text_format
        return config.text_format_off

    @property
    def variables(self) -> _VarCache:
        if self.__variables__ is None:
            vars = self._collect_vars(self._like_components + self._prior_components)
            params = tuple(sorted(vars[0]))
            constants = tuple(sorted(vars[1]))
            locals = tuple(sorted(vars[2]))
            mapping = {c.var: f"self.{c.var}" for c in constants}
            mapping.update({loc.var: f"self.{loc.var}" for loc in locals})
            mapping.update({p.var: f"params[{i}]" for i, p in enumerate(params)})
            self.__variables__ = _VarCache(params, constants, locals, mapping)  # type: ignore
        return self.__variables__

    @property
    def params(self) -> tuple[Symb]:
        return self.variables.p

    @property
    def constants(self) -> tuple[Symb]:
        return self.variables.c

    @property
    def locals(self) -> tuple[Symb]:
        return self.variables.l

    @property
    def mapping(self) -> dict[str, str]:
        return self.variables.map

    def __init__(self, verbose: bool = False, fancy_text=False):
        self.verbose: bool = verbose
        self.fancy_text = fancy_text
        self._like_components = []
        self._prior_components = []
        self.imports: list[str] = [
            "from starlord.cy_tools cimport *",
            "from starlord import GridGenerator",
        ]
        self.auto_constants = {}
        self.constant_types = {}
        # Lazily-updated property backer
        self.__variables__: Optional[_VarCache] = None

    def generate_prior_ppf(self) -> str:
        result: list[str] = []
        result.append("cpdef double[:] prior_transform(self, double[:] params):")
        prior_params = {list(c.vars)[0] for c in self._prior_components}
        params = set(self.params)
        assert not params - prior_params, f"Priors were not set for param(s) {params-prior_params}."
        assert not prior_params - params, f"Priors were set for unrecognized param(s) {prior_params-params}."
        for comp in self._prior_components:
            code: str = comp.generate_ppf().format(**self.mapping)
            result.append("\n".join("    " + loc for loc in code.splitlines()))
        result.append("    return params\n")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_log_prior(self) -> str:
        result: list[str] = []
        result.append("cpdef double log_prior(self, double[:] params):")
        result.append("    cdef double logP = 0.")
        params = set(self.params)
        prior_params = {list(c.vars)[0] for c in self._prior_components}
        assert not params - prior_params, f"Priors were not set for param(s) {params-prior_params}."
        assert not prior_params - params, f"Priors were set for unrecognized param(s) {prior_params-params}."
        for comp in self._prior_components:
            code: str = comp.generate_pdf().format(**self.mapping)
            result.append("\n".join("    " + i for i in code.splitlines()))
        result.append("    return logP\n")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_forward_model(self) -> str:
        # Write the function header
        result: list[str] = []
        result.append("cdef void _forward_model(self, double[:] params):")
        # Generate the code for each component, sorted to satisfy their interdependencies
        components = [c for c in self._like_components if type(c) is not DistributionComponent]
        components = self._sort_by_dependency(components)
        for comp in components:
            code: str = comp.generate_code().format(**self.mapping)
            result.append("\n".join("    " + loc for loc in code.splitlines()))
        result[-1] = result[-1] + "\n"
        result.append("cpdef dict forward_model(self, double[:] params):")
        result.append("    self._forward_model(params)")
        ret_locals = ", ".join([f"{loc.name}={self.mapping[loc.var]}" for loc in self.locals])
        result.append(f"    return dict({ret_locals})\n")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_log_like(self) -> str:
        # Write the function header
        result: list[str] = []
        result.append("cdef double _log_like(self, double[:] params):")
        result.append("    cdef double logL = 0.")
        for comp in self._like_components:
            if type(comp) is DistributionComponent:
                code: str = comp.generate_code().format(**self.mapping)
                result.append("\n".join("    " + loc for loc in code.splitlines()))
        result.append("    return logL if math.isfinite(logL) else -math.INFINITY\n")
        # Generate the user-facing wrapper function
        result.append("cpdef double log_like(self, double[:] params):")
        result.append("    self._forward_model(params)")
        result.append("    return self._log_like(params)\n")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_log_prob(self) -> str:
        result: list[str] = []
        result.append("cpdef double log_prob(self, double[:] params):")
        result.append("    cdef double logP = self.log_prior(params)")
        result.append("    logP += self.log_like(params)")
        result.append("    return logP\n")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_init(self) -> str:
        result: list[str] = []
        # Organize function arguments and constants to be set
        args = ['self']
        definitions = []
        for c in self.constants:
            ct = self.constant_types.get(c.name, "double")
            cm = self.mapping[c.var]
            if c.name in self.auto_constants:
                definitions.append(f"    if '{c.name}' in args:")
                definitions.append(f"        {cm} = args['{c.name}']")
                definitions.append("    else:")
                definitions.append(f"        {cm} = {self.auto_constants[c.name]}")
            else:
                args.append(f"{ct} {c[5:]}")
                definitions.append(f"    {cm} = {c[5:]}")
        args.append("**args")
        args = ", ".join(args)
        # Write the function
        result.append(f"def __init__({args}):")
        result.append(f"    self.param_names = {[p.name for p in self.params]}")
        result += definitions
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate(self) -> str:
        result: list[str] = []
        result.append("# Generated by Starlord.  Versions:")
        versions = f"# Starlord {__version__}, Cython {cython.__version__}, Python {sys.version}"
        result.append(re.sub("\n", " ", versions))
        result.append("\n".join(self.imports) + "\n")

        # Class and constant declarations
        result.append("cdef class Model:")
        result.append("    cdef public object param_names")
        for c in self.constants:
            ct = self.constant_types.get(c.name, "double")
            cm = self.mapping[c.var][5:]
            result.append(f"    cdef public {ct} {cm}")
        result.append("")

        # Local variable declarations
        for loc in self.locals:
            result.append(f"    cdef public double {self.mapping[loc.var][5:]}")
        result.append("")

        result.append(self.generate_forward_model())
        result.append(self.generate_log_like())
        result.append(self.generate_prior_ppf())
        result.append(self.generate_log_prior())
        result.append(self.generate_log_prob())
        result.append(self.generate_init())
        return "\n".join(result) + "\n"

    def compile(self) -> ModuleType:
        hash = CodeGenerator._compile_to_module(self.generate())
        return CodeGenerator._load_module(hash)

    def summary(self, fancy=False) -> str:
        result: list[str] = []
        result += [f"    {self.txt.underline}Variables{self.txt.end}"]
        if self.params:
            result += ["Params:".ljust(12) + ", ".join([p for p in self.params])]
        if self.constants:
            result += ["Constants:".ljust(12) + ", ".join([c for c in self.constants])]
        if self.locals:
            result += ["Locals:".ljust(12) + ", ".join([loc for loc in self.locals])]
        result += [f"\n    {self.txt.underline}Forward Model{self.txt.end}"]
        likelihood = []
        for comp in self._sort_by_dependency(self._like_components):
            if type(comp) is DistributionComponent:
                likelihood.append(comp.display())
            else:
                result.append(comp.display().format(**self.mapping))
        result += [f"\n    {self.txt.underline}Likelihood{self.txt.end}"]
        result += [str(i) for i in likelihood]
        result += [f"\n    {self.txt.underline}Prior{self.txt.end}"]
        prior_comps = sorted(self._prior_components, key=lambda c: "_".join(sorted(c.vars)))
        result += [c.display() for c in prior_comps]
        result_str = "\n".join(result)
        # Highlight the output, if requested
        if fancy:
            result_str = CodeGenerator.fancy_print(result_str, self.txt)
        return result_str

    def expression(self, expr: str) -> None:
        '''Specify a general expression to add to the code.  Assignments and variables used will be
        automatically detected so long as they are formatted properly (see CodeGenerator doc)'''
        provides = set()
        # Finds assignment blocks like "l.foo = " and "l.bar, l.foo = "
        assigns = re.findall(r"^\s*[pcl]\.[A-Za-z_]\w*\s*(?:,\s*[pcl]\.[A-Za-z_]\w*)*\s*=(?!=)", expr, flags=re.M)
        assigns += re.findall(
            r"^\s*\(\s*[pcl]\.[A-Za-z_]\w*\s*(?:,\s*[pcl]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr, flags=re.M)
        # Same as above but covers when vars are enclosed by parentheses like "(l.a, l.b) ="
        assigns += re.findall(
            r"^\s*\(\s*[pcl]\.[A-Za-z_]\w*\s*(?:,\s*[pca]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr, flags=re.M)
        for block in assigns:
            # Handles parens, multiple assignments, extra whitespace, and removes the "="
            block = block[:-1].strip(" ()")
            # Block now looks like "l.foo" or "l.foo, l.bar"
            for var in block.split(","):
                var = var.strip()
                # Verify that the result is a local var "l.foo"
                assert var[:2] == "l.", var
                provides.add(Symb(var))
        code, variables = self._extract_params(expr)
        requires = variables - provides
        comp = Component(requires, provides, code)
        if self.verbose:
            print(CodeGenerator.fancy_print("\n".join([line for line in str(comp).split("\n")]), self.txt))
        self._like_components.append(comp)
        self._vars_out_of_date = True

    def assign(self, var: str, expr: str) -> None:
        # If l or b is omitted, l is implied
        var = Symb(var if re.match(r"^l\.", var) is not None else f"l.{var}")
        code, variables = self._extract_params(expr)
        comp = AssignmentComponent.create(var, code, variables - {var})
        if self.verbose:
            print(CodeGenerator.fancy_print(comp.display(), self.txt))
        self._like_components.append(comp)
        self._vars_out_of_date = True

    def constraint(self, var: str, dist: str, params: list[str | float]) -> None:
        var = Symb(var)
        assert len(params) == 2
        pars: list[Symb] = [Symb(i) for i in params]
        comp = DistributionComponent.create(var, dist, pars)
        if self.verbose:
            print(CodeGenerator.fancy_print(comp.display(), self.txt))
        self._like_components.append(comp)
        self._vars_out_of_date = True

    def prior(self, var: str, dist: str, params: list[str | float]):
        var = Symb(var)
        assert len(params) == 2
        pars: list[Symb] = [Symb(i) for i in params]
        comp = Prior.create(var, dist, pars)
        if self.verbose:
            print(CodeGenerator.fancy_print(comp.display(), self.txt))
        self._prior_components.append(comp)
        self._vars_out_of_date = True

    @staticmethod
    def fancy_print(source, txt):
        source = re.sub(r"(?<!\w)(d\.[a-zA-Z_]\w+)", f"{txt.bold}{txt.red}\\g<1>{txt.end}", source)
        source = re.sub(r"(?<!\w)(p\.[a-zA-Z_]\w+)", f"{txt.bold}{txt.yellow}\\g<1>{txt.end}", source)
        source = re.sub(r"(?<!\w)(c\.[a-zA-Z_]\w+)", f"{txt.bold}{txt.blue}\\g<1>{txt.end}", source)
        source = re.sub(r"(?<!\w)(l\.[a-zA-Z_]\w+)", f"{txt.bold}{txt.green}\\g<1>{txt.end}", source)
        source = re.sub(r"(?<!\033\[)(?<![\w\\])([+-]?(?:[0-9]*[.])?[0-9]+)", f"{txt.blue}\\g<1>{txt.end}", source)
        return source

    @staticmethod
    def _sort_by_dependency(components: list[Component]) -> list[Component]:
        '''Takes a list of components and returns a new one sorted such that components which provide
        variables are listed before those that require them. Beyond this the sort is stable
        (components which could appear in any order appear in the order found in their input list).'''
        _, _, locals = CodeGenerator._collect_vars(components)
        # Check that every local used is initialized somewhere
        for loc in locals:
            for comp in components:
                if loc in comp.provides:
                    break
            else:
                raise LookupError(f"Variable {loc} is used but never initialized.")
        # Sort components according to their initialization requirements
        result = []
        initialized = set()
        components = components.copy()
        while len(components) > 0:
            for comp in components:
                reqs = {c for c in comp.requires if c[:2] == "l." and c not in initialized}
                if len(reqs) == 0:
                    initialized = initialized | comp.provides
                    result.append(comp)
                    components.remove(comp)
                    break
            else:
                raise LookupError(f"Circular dependencies in components {components}")
        return result

    @staticmethod
    def _collect_vars(target: list[Component]) -> tuple[set[Symb], set[Symb], set[Symb]]:
        params = set()
        consts = set()
        locals = set()
        for comp in target:
            for sym in comp.requires | comp.provides:
                if sym.label == "p":
                    params.add(sym)
                elif sym.label == "c":
                    consts.add(sym)
                elif sym.label == "l":
                    locals.add(sym)
                else:
                    raise ValueError(f"Invalid symbol {sym}.")
        return params, consts, locals

    @staticmethod
    def _extract_params(source: str) -> tuple[str, set[Symb]]:
        '''Extracts variables from the given string and replaces them with format brackets.
        Variables can be constants "c.name", parameters "p.name", or local variables "l.name".'''
        vars = set()
        replace_var = partial(CodeGenerator._replace_var, vars=vars)
        template = re.sub(r"(?<!\w)([pcl]\.[A-Za-z_]\w*)", replace_var, source, flags=re.M)
        return template, vars

    @staticmethod
    def _replace_var(source: re.Match, vars: set[Symb]) -> str:
        var = Symb(source.group())
        vars.add(var)
        return var.bracketed

    @staticmethod
    def _cleanup_old_modules(exclude: list[str] = [], ignore_below: int = 20, stale_time: float = 7.) -> None:
        module_files = list(config.cache_dir.glob("sl_gen_*.so"))
        now = time.time()
        candidates = []
        for file in module_files:
            age = (now - file.stat().st_atime)
            hash = file.name[7:47]
            if hash not in exclude and age > stale_time * 86400:  # Seconds per day
                candidates.append((age, hash))
        candidates.sort()
        for age, hash in candidates[ignore_below:]:
            files = list(config.cache_dir.glob(f"sl_gen_{hash}*"))
            files = [f for f in files if f.suffix in [".pyx", ".so", ".dll", ".dynlib", ".sl"]]
            for f in files:
                # A few last checks out of paranoia, then delete
                assert f.exists() and f.is_file(), "Tried to delete a file that doesn't exist.  What?"
                assert f.parent == config.cache_dir, "Tried to delete a file out of the cache directory."
                f.unlink()

    @staticmethod
    def _compile_to_module(code: str) -> str:
        # Get the code hash for file lookup
        hasher = hashlib.shake_128(code.encode())
        hash = base64.b32encode(hasher.digest(25)).decode("utf-8")
        name = f"sl_gen_{hash}"
        pyxfile = config.cache_dir / (name+".pyx")
        # Write the pyx file if needed
        if not pyxfile.exists():
            with pyxfile.open("w") as pxfh:
                pxfh.write(code)
                pxfh.close()
            assert pyxfile.exists(), "Wrote the code to a file, but the file still doesn't exist."
        libfiles = list(config.cache_dir.glob(name + ".*.*"))
        if len(libfiles) == 0:
            CodeGenerator._cleanup_old_modules([hash])
            assert os.system(f"cythonize -f -i {pyxfile}") == 0, "Compilation failed (see error message)"
            cfile = config.cache_dir / (name+".c")
            libfiles = list(config.cache_dir.glob(name + ".*.*"))
            assert len(libfiles) >= 1, "Compiled but failed to produce an object file to import."
            # Remove the (surprisingly large) build c file artifact
            if cfile.exists():
                cfile.unlink()
            builddir = config.cache_dir / "build"
            # Remove the build directory -- the output was moved to cache_dir automatically
            if builddir.exists():
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
