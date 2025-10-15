# Starlord

[![Test and Report](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml)
[![Test Count](https://dpthorngren.github.io/Starlord/tests_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/)
[![Test Coverage](https://dpthorngren.github.io/Starlord/coverage_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/)
[![Cython Annotation](https://dpthorngren.github.io/Starlord/cython.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/cy_tools.html)

A Python library for fitting stellar observations to models, varying the model grid and parameters.

This project is under active development and is not yet ready for general use.

## Installation
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


## Usage
Starlord may be invoked from the command line with `starlord`.  Typically this will be with a settings file defined in the [TOML format](https://toml.io/en/v1.0.0) with the sections `[model]`, `[sampling]`, and `[output]`, and invoked with `starlord run_settings.toml`.  Documentation on the settings file is to-do.  Additional command-line options can be viewed with `starlord --help`.  It may also be employed programmatically with `import starlord` via the `starlord.StarFitter()` class.  That documentation is also to-do -- I meant it when I said "active development".
