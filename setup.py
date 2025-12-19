import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext = Extension("starlord.cy_tools", ["src/starlord/cy_tools.pyx"], include_dirs=[numpy.get_include()])

directives = {
    'embedsignature': True,
    'cdivision': True,
    'initializedcheck': False,
    'boundscheck': False,
    'binding': True,
}
# Hack to detect if coverage data is being generated
if os.path.exists("./htmlcov"):
    ext.define_macros = [("CYTHON_TRACE_NOGIL", "1")]
    directives['linetrace'] = True

setup(ext_modules=cythonize([ext], compiler_directives=directives, annotate=True))
