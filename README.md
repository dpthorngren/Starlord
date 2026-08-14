Starlord
====================
[![Test and Report](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml)
[![Test Count](https://dpthorngren.github.io/Starlord/htmlcov/tests_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/index.html)
[![Test Coverage](https://dpthorngren.github.io/Starlord/htmlcov/coverage_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/index.html)
[![Cython Annotation](https://dpthorngren.github.io/Starlord/htmlcov/cython.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/cy_tools.html)

[**Documentation**](https://dpthorngren.github.io/Starlord/)

A Python library for Bayesian fits of models with gridded functions to data, with an emphasis on very flexible stellar model fitting.

This project has reached a beta release.  If you encounter any bugs, please report them in the issues tab or by emailing the author (Daniel Thorngren)

Installation
--------------------
All the dependencies can be handled by Pip, though note that we do not support Python 2.  This package is not yet on PyPi and so must be installed from this repository.  This can be done either by installing it directly with Pip:
```
pip install git+https://github.com/dpthorngren/Starlord.git#egg=starlord
```
or by cloning the repository and installing it with Pip:
```
git clone git@github.com:dpthorngren/Starlord.git
cd Starlord
pip install .
```

Finally, you must obtain grids to fit with.  To download the full set of standard grids from [Zenodo]([url](https://zenodo.org/records/21911646)), you can use:
```
starlord --download all
```
or replace all with the name of the grid to download just that one (the downloadable grids are listed if no argument is given)

Once installed, Starlord can be invoked in the terminal with `starlord`, which with no arguments prints basic help information.  For usage see the [Documentation](https://dpthorngren.github.io/Starlord/).

Roadmap
--------------------
This is a list of features I'd like to add, roughly ordered by priority and definitely subject to change.

 - **Python interpolation support** -- The grid system is faster than `scipy.RegularGridInterpolator`, but currently awkward to use directly in Python.
 - **Additional Standard Grids** -- Brown dwarfs, white dwarfs, and circumstellar disks are of particular interest.
 - **Vector Interpolation** -- For low-resolution spectra and faster interpolation of many outputs from the same grid.
 - **Vector Operations** -- Useful for more general Bayesian models.
