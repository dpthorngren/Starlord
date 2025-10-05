import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = Extension("starlord.cy_tools", ["src/starlord/cy_tools.pyx"], include_dirs=[numpy.get_include()])

# Hack to detect if coverage data is being generated
directives = {}
if os.path.exists("./htmlcov"):
    ext.define_macros = [("CYTHON_TRACE_NOGIL", "1")]
    directives['linetrace'] = True

setup(ext_modules=cythonize([ext], compiler_directives=directives))
