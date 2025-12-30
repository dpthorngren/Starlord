from __future__ import annotations

import base64
import hashlib
import os
import re
import shutil
import sys
import time
from importlib import util
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import cython

from ._config import __version__, config
from .code_components import (AssignmentComponent, Component, DistributionComponent, Prior, Symb)


class Namespace(SimpleNamespace):
    '''A slightly less simple namespace, allowing for [] and iteration'''

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return self.__dict__.items().__iter__()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()


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

    def __init__(self, verbose: bool = False):
        self.verbose: bool = verbose
        self._like_components = []
        self._prior_components = []
        self._mark_autogen: bool = False
        self.imports: list[str] = [
            "from starlord.cy_tools cimport *",
            "from starlord import GridGenerator",
        ]
        self.auto_constants = {}
        # Lazily-updated property backers
        self._vars_out_of_date: bool = True
        self._variables: set[Symb] = set()
        self._params: list[Symb] = []
        self._constants: list[Symb] = []
        self._blobs: list[Symb] = []
        self._locals: list[Symb] = []
        self.constant_types = {}

    def _update_vars(self):
        self._variables, result = self._collect_vars(self._like_components + self._prior_components)
        self._params = sorted(list(result['p']))
        self._constants = sorted(list(result['c']))
        self._blobs = sorted(list(result['b']))
        self._locals = sorted(list(result['l']))
        self._vars_out_of_date = False

    def get_mapping(self, prepend_self=False) -> dict[str, Namespace]:
        self._update_vars()
        p = "self." if prepend_self else ""
        mapping: dict[str, Namespace] = {}
        mapping['c'] = Namespace(**{c.name: p + c.var for c in self.constants})
        mapping['l'] = Namespace(**{loc.name: p + loc.var for loc in self.locals})
        mapping['p'] = Namespace(**{n.name: f"params[{i}]" for i, n in enumerate(self.params)})
        if self.blobs:
            raise NotImplementedError
        return mapping

    def generate_prior_ppf(self) -> str:
        mapping = self.get_mapping(True)
        result: list[str] = []
        result.append("cpdef double[:] prior_transform(self, double[:] params):")
        params = self._collect_vars(self._like_components)[1]['p']
        prior_params = {list(c.vars)[0] for c in self._prior_components}
        assert not params - prior_params, f"Priors were not set for param(s) {params-prior_params}."
        assert not prior_params - params, f"Priors were set for unrecognized param(s) {prior_params-params}."
        for comp in self._prior_components:
            code: str = comp.generate_ppf().format(**mapping)
            result.append("\n".join("    " + loc for loc in code.splitlines()))
        result.append("    return params\n")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_log_prior(self) -> str:
        mapping = self.get_mapping(True)
        result: list[str] = []
        result.append("cpdef double log_prior(self, double[:] params):")
        result.append("    cdef double logP = 0.")
        params = self._collect_vars(self._like_components)[1]['p']
        prior_params = {list(c.vars)[0] for c in self._prior_components}
        assert not params - prior_params, f"Priors were not set for param(s) {params-prior_params}."
        assert not prior_params - params, f"Priors were set for unrecognized param(s) {prior_params-params}."
        for comp in self._prior_components:
            code: str = comp.generate_pdf().format(**mapping)
            result.append("\n".join("    " + loc for loc in code.splitlines()))
        result.append("    return logP")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_forward_model(self) -> str:
        mapping = self.get_mapping(True)
        # Write the function header
        result: list[str] = []
        result.append("cdef void _forward_model(self, double[:] params):")
        # Generate the code for each component, sorted to satisfy their interdependencies
        components = [c for c in self._like_components if type(c) is not DistributionComponent]
        components = self._sort_by_dependency(components)
        for comp in components:
            code: str = comp.generate_code().format(**mapping)
            result.append("\n".join("    " + loc for loc in code.splitlines()))
        result[-1] = result[-1] + "\n"
        result.append("cpdef dict forward_model(self, double[:] params):")
        result.append("    self._forward_model(params)")
        ret_locals = ", ".join([f"{k}={v}" for k, v in mapping['l'].__dict__.items()])
        result.append(f"    return dict({ret_locals})\n")
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate_log_like(self) -> str:
        mapping = self.get_mapping(True)
        # Write the function header
        result: list[str] = []
        result.append("cdef double _log_like(self, double[:] params):")
        result.append("    cdef double logL = 0.")
        for comp in self._like_components:
            if type(comp) is DistributionComponent:
                code: str = comp.generate_code().format(**mapping)
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
        mapping = self.get_mapping(True)
        result: list[str] = []
        # Organize function arguments and constants to be set
        args = ['self']
        definitions = []
        for n, c in mapping['c']:
            ct = self.constant_types.get(n, "double")
            if n in self.auto_constants:
                definitions.append(f"    if '{n}' in args:")
                definitions.append(f"        {c} = args['{n}']")
                definitions.append("    else:")
                definitions.append(f"        {c} = {self.auto_constants[n]}")
            else:
                args.append(f"{ct} {c[5:]}")
                definitions.append(f"    {c} = {c[5:]}")
        args.append("**args")
        args = ", ".join(args)
        # Write the function
        result.append(f"def __init__({args}):")
        param_names = list(mapping['p'].__dict__.keys())
        result.append(f"    self.param_names = {param_names}")
        result += definitions
        result = ["    " + r for r in result]
        return "\n".join(result)

    def generate(self) -> str:
        mapping = self.get_mapping()
        result: list[str] = []
        result.append("# Generated by Starlord.  Versions:")
        result.append(f"# Starlord {__version__}, Cython {cython.__version__}, Python {sys.version}")
        result.append("\n".join(self.imports) + "\n")

        # Class and constant declarations
        result.append("cdef class Model:")
        result.append("    cdef public object param_names")
        for n, c in mapping['c']:
            ct = self.constant_types.get(n, "double")
            result.append(f"    cdef public {ct} {c}")
        result.append("")

        # Local vairable declarations
        for n, l in mapping['l']:
            result.append(f"    cdef public double {l}")
        result.append("")

        result.append(self.generate_forward_model())
        result.append(self.generate_log_like())
        result.append(self.generate_prior_ppf())
        result.append(self.generate_log_prior())
        result.append(self.generate_log_prob())
        result.append(self.generate_init())
        return "\n".join(result)

    def compile(self) -> ModuleType:
        hash = CodeGenerator._compile_to_module(self.generate())
        return CodeGenerator._load_module(hash)

    def summary(self, fancy=False) -> str:
        txt = config.text_format if fancy else config.text_format_off
        result: list[str] = []
        result += [f"    {txt.underline}Variables{txt.end}"]
        if self.params:
            result += ["Params:".ljust(12) + ", ".join([txt.yellow + p[2:] + txt.end for p in self.params])]
        if self.constants:
            consts = []
            for c in self.constants:
                if c in self.constant_types:
                    consts.append(txt.blue + c[2:] + txt.end + " (" + self.constant_types[c] + ")")
                else:
                    consts.append(txt.blue + c[2:] + txt.end)
            result += ["Constants:".ljust(12) + ", ".join(consts)]
        if self.blobs:
            result += ["Blobs:".ljust(12) + ", ".join([txt.green + b[2:] + txt.end for b in self.blobs])]
        if self.locals:
            result += ["Locals:".ljust(12) + ", ".join([txt.green + loc[2:] + txt.end for loc in self.locals])]
        result += [f"\n    {txt.underline}Likelihood{txt.end}"]
        result += [str(i) for i in self._sort_by_dependency(self._like_components)]
        result += [f"\n    {txt.underline}Prior{txt.end}"]
        result += [str(c) for c in self._prior_components]
        return self._fancy_format("\n".join(result), fancy)

    def expression(self, expr: str) -> None:
        '''Specify a general expression to add to the code.  Assignments and variables used will be
        automatically detected so long as they are formatted properly (see CodeGenerator doc)'''
        provides = set()
        # Finds assignment blocks like "l.foo = " and "l.bar, l.foo = "
        assigns = re.findall(r"^\s*[pcbl]\.[A-Za-z_]\w*\s*(?:,\s*[pcbl]\.[A-Za-z_]\w*)*\s*=(?!=)", expr, flags=re.M)
        assigns += re.findall(
            r"^\s*\(\s*[pcbl]\.[A-Za-z_]\w*\s*(?:,\s*[pcbl]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr, flags=re.M)
        # Same as above but covers when vars are enclosed by parentheses like "(l.a, l.b) ="
        assigns += re.findall(
            r"^\s*\(\s*[pcbl]\.[A-Za-z_]\w*\s*(?:,\s*[pcba]\.[A-Za-z_]\w*)*\s*\)\s*=(?!=)", expr, flags=re.M)
        for block in assigns:
            # Handles parens, multiple assignments, extra whitespace, and removes the "="
            block = block[:-1].strip(" ()")
            # Block now looks like "l.foo" or "l.foo, l.bar"
            for var in block.split(","):
                var = var.strip()
                # Verify that the result is a local or blob formatted as "l.foo" or "b.bar"
                assert var[0] in "lb" and var[1] == ".", var
                provides.add(Symb(var))
        code, variables = self._extract_params(expr)
        requires = variables - provides
        comp = Component(requires, provides, code, self._mark_autogen)
        if self.verbose:
            print(self._fancy_format("\n".join(["    " + line for line in str(comp).split("\n")])))
        self._like_components.append(comp)

    def assign(self, var: str, expr: str) -> None:
        # If l or b is omitted, l is implied
        var = Symb(var if re.match(r"^[bl]\.", var) is not None else f"l.{var}")
        code, variables = self._extract_params(expr)
        comp = AssignmentComponent.create(var, code, variables - {var}, self._mark_autogen)
        if self.verbose:
            print("    " + self._fancy_format(str(comp)))
        self._like_components.append(comp)

    def constraint(self, var: str, dist: str, params: list[str | float]) -> None:
        var = Symb(var)
        assert len(params) == 2
        pars: list[Symb] = [Symb(i) for i in params]
        comp = DistributionComponent.create(var, dist, pars, self._mark_autogen)
        if self.verbose:
            print("    " + self._fancy_format(str(comp)))
        self._like_components.append(comp)

    def prior(self, var: str, dist: str, params: list[str | float]):
        var = Symb(var)
        assert len(params) == 2
        pars: list[Symb] = [Symb(i) for i in params]
        comp = Prior.create(var, dist, pars)
        if self.verbose:
            print("    " + self._fancy_format(str(comp)))
        self._prior_components.append(comp)

    def remove_generated(self):
        self._like_components = [c for c in self._like_components if not c.autogenerated]
        self._vars_out_of_date = True

    @staticmethod
    def _fancy_format(source: str, fancy=False) -> str:
        '''Finds variables enclosed in curly braces and boldens and colors them based on their type'''
        txt = config.text_format if fancy else config.text_format_off
        result = re.sub(r"{(p\.\w+)}", f"{txt.bold}{txt.yellow}\\g<1>{txt.end}", source, flags=re.M)
        result = re.sub(r"{(c\.\w+)}", f"{txt.bold}{txt.blue}\\g<1>{txt.end}", result, flags=re.M)
        result = re.sub(r"{([bl]\.\w+)}", f"{txt.bold}{txt.green}\\g<1>{txt.end}", result, flags=re.M)
        result = re.sub(r"{([+-]?([0-9]*[.])?[0-9]+)}", f"{txt.blue}\\g<1>{txt.end}", result, flags=re.M)
        return result

    @staticmethod
    def _sort_by_dependency(components: list[Component]) -> list[Component]:
        '''Takes a list of components and returns a new one sorted such that components which provide
        variables are listed before those that require them. Beyond this the sort is stable
        (components which could appear in any order appear in the order found in their input list).'''
        _, vars = CodeGenerator._collect_vars(components)
        # Check that every local and blob used is initialized somewhere
        for v in vars['l'] | vars['b']:
            for comp in components:
                if v in comp.provides:
                    break
            else:
                raise LookupError(f"Variable {v} is used but never initialized.")
        # Sort components according to their initialization requirements
        result = []
        initialized = set()
        components = components.copy()
        while len(components) > 0:
            for comp in components:
                reqs = {c for c in comp.requires if c[0] in "bl" and c not in initialized}
                if len(reqs) == 0:
                    initialized = initialized | comp.provides
                    result.append(comp)
                    components.remove(comp)
                    break
            else:
                raise LookupError(f"Circular dependencies in components {components}")
        return result

    @staticmethod
    def _collect_vars(target: list[Component]) -> tuple[set[Symb], dict[str, set[Symb]]]:
        variables = set()
        result: dict[str, set[Symb]] = {i: set() for i in 'pcbl'}
        for comp in target:
            for sym in comp.requires | comp.provides:
                assert sym.label in 'pcbl', f"Bad symbol name {sym}"
                result[sym.label].add(sym)
                variables.add(sym)
        return variables, result

    @staticmethod
    def _extract_params(source: str) -> tuple[str, set[Symb]]:
        '''Extracts variables from the given string and replaces them with format brackets.
        Variables can be constants "c.name", blobs "b.name", parameters "p.name", or local variables "l.name".'''
        template: str = re.sub(r"(?<!\w)([pcbl]\.[A-Za-z_]\w*)", r"{\1}", source, flags=re.M)
        all_vars: list[str] = re.findall(r"(?<=\{)[pcbl]\.[A-Za-z_]\w*(?=\})", template, flags=re.M)
        variables: set[Symb] = {Symb(v) for v in all_vars}
        return template, variables

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
            os.system(f"cythonize -f -i {pyxfile}")
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
