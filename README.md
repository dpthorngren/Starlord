Starlord
====================
[![Test and Report](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml)
[![Test Count](https://dpthorngren.github.io/Starlord/htmlcov/tests_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/index.html)
[![Test Coverage](https://dpthorngren.github.io/Starlord/htmlcov/coverage_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/index.html)
[![Cython Annotation](https://dpthorngren.github.io/Starlord/htmlcov/cython.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/htmlcov/cy_tools.html)

[**Documentation**](https://dpthorngren.github.io/Starlord/), [**Zenodo Record**](https://zenodo.org/records/21911646)

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

Finally, you must obtain grids to fit with.  To download the full set of standard grids from [Zenodo](https://zenodo.org/records/21911646), you can use:
```
starlord --download all
```
which will download ~1GB of data files to `~/.config/starlord/grids/`.  Alternatively, replace `all` with the name of the grid to download just that one; the downloadable grids are listed if no argument is given.

Basic Usage
--------------------
Once installed, Starlord can be invoked in the terminal with `starlord`, which with no arguments prints basic help information.  Starlord models are defined by [TOML files](https://dpthorngren.github.io/Starlord/models.html), and can be run with just `starlord my_model.toml` or with the Python API.

If you're interested in stellar characterization, see this [quickstart guide](https://dpthorngren.github.io/Starlord/quickstart/stars.html) and these basic input TOML files to get you started:
- [MIST1, HD 209458](./docs/examples/grid_examples/mist1_hd209458.toml)
- [MIST2, HD 209458](./docs/examples/grid_examples/mist2_hd209458.toml)
- [PARSEC, HD 209458](./docs/examples/grid_examples/parsec_hd209458.toml)

For planetary characterization, see this [example input TOML file](./docs/examples/grid_examples/hotJupiter.toml) and this [quickstart guide](https://dpthorngren.github.io/Starlord/quickstart/planets.html).

For more information see the [documentation](https://dpthorngren.github.io/Starlord/).

Roadmap
--------------------
This is a list of features I'd like to add, roughly ordered by priority and definitely subject to change.

 - **Python interpolation support** -- The grid system is faster than `scipy.RegularGridInterpolator`, but currently awkward to use directly in Python.
 - **Additional Standard Grids** -- Brown dwarfs, white dwarfs, and circumstellar disks are of particular interest.
 - **Vector Interpolation** -- For low-resolution spectra and faster interpolation of many outputs from the same grid.
 - **Vector Operations** -- Useful for more general Bayesian models.
