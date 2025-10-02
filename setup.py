#!/usr/bin/env python
from setuptools import Extension, setup
from Cython.Build import cythonize

ext = Extension("cystar", ["starlord/cystar.pyx"])

setup(
    ext_modules=cythonize([ext], include_path=["starlord/"]),
    package_data={"starlord": ["starlord/cystar.pxd"]}
)
