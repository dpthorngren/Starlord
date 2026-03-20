Starlord
====================
[![Test and Report](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml)
[![Test Count](https://dpthorngren.github.io/Starlord/htmlcov/tests_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/index.html)
[![Test Coverage](https://dpthorngren.github.io/Starlord/htmlcov/coverage_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/index.html)
[![Cython Annotation](https://dpthorngren.github.io/Starlord/htmlcov/cython.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/cy_tools.html)

[**Documentation**](https://dpthorngren.github.io/Starlord/)

A Python library for Bayesian fits of models with gridded functions to data, with an emphasis on very flexible stellar model fitting.

This project has reached an alpha release.  There are many missing features (see [roadmap](#roadmap)) and very likely some bugs.  Please let me know if you find any.

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
Once installed, Starlord can be invoked in the terminal with `starlord`, which with no arguments prints basic help information.  For usage see the [Documentation](https://dpthorngren.github.io/Starlord/).

> [!IMPORTANT]
> The current version of Starlord does not come with any grids pre-installed, including the MIST grid the documentation references.  This is because I have not yet sought permission from their respective maintainers to redistribute them publicly.  For now, contact me (Daniel Thorngren) if you would like to use them.

Roadmap
--------------------
This is a list of features I'd like to add, roughly ordered by priority and definitely subject to change.

 - **Prior Features** -- Truncated distributions and priors on transformed parameters.
 - **Stellar Physics Priors** -- Physically motivated priors for stars, such as the Chabrier (2003) stellar mass prior.
 - **Optional Likelihood Terms** -- In batch operations, some stars may not always have the full set of photometric observations.  This would allow you to skip missing values in the likelihood without recompiling.
 - **Implicit Variables** -- If mass is defined, then log_mass is pretty clear in its intent.
 - **Vector Interpolation** -- For low-resolution spectra and faster interpolation of many outputs from the same grid.
 - **Vector Operations** -- Useful for more general Bayesian models.
 - **Python interpolation support** -- The grid system is faster than `scipy.RegularGridInterpolator`, but currently awkward to use directly in Python.
