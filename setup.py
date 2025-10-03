from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension("starlord.cy_tools", ["src/starlord/cy_tools.pyx"], include_dirs = [numpy.get_include()])

setup(ext_modules=cythonize([ext]))
