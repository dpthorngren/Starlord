Starlord
====================

[![Test and Report](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml/badge.svg?branch=main)](https://github.com/dpthorngren/Starlord/actions/workflows/python-test.yml)
[![Test Count](https://dpthorngren.github.io/Starlord/tests_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/)
[![Test Coverage](https://dpthorngren.github.io/Starlord/coverage_badge.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/)
[![Cython Annotation](https://dpthorngren.github.io/Starlord/cython.svg?dummy=unused)](https://dpthorngren.github.io/Starlord/cy_tools.html)

A Python library for Bayesian fits of models with gridded functions to data, with an emphasis on very flexible stellar model fitting.

This project is approaching an alpha release -- see the [roadmap](#roadmap).

Motivation
--------------------
In astronomy, it is common to need to fit the parameters of a computationally expensive forward model, accomplishing this by pre-computing the model on a grid of input values and interpolating the results.  Stellar structure grids like [MIST](https://waps.cfa.harvard.edu/MIST/model_grids.html) are a prominent example, where stellar atmosphere models are computed for a set of effective temperatures, metallicity, and surface gravities and output a wide range of photometric band luminosities.  These in turn depend on stellar evolution grids in mass, metallicity, and age that output radius, bolometric luminosity, etc.  While fitting Bayesian models using these grids as part of the likelihood function is quite common, small changes to the grid necessitates fitting code changes that can be cumbersome and error-prone when experimenting.  In particular, adding parameters to a model or changing out which grid is being used can force users to rewrite large portions of their statistical code.

Starlord is a model fitting library that resolves these issues by integrating a grid management system, a code generator for Cython, and Bayesian model samplers into a coherent whole.  The goal is to allow domain experts to sample posteriors using their model grids (and rapidly iterate on the results) without getting bogged down in the computational boilerplate.  The key to this is the grid management system, which incorporates enough metadata that Starlord can infer how constraints on grid outputs should be handled.

For example, suppose a user wishes to fit a number of photometric band observations of a star to the MIST grid.  By specifying an observation and uncertainty on the bands Starlord not only constructs the corresponding log Likelihood function, but also generates the interpolator and works out what inputs it requires.  Moreover, the grid metadata tells Starlord how to construct the inputs from the MIST stellar tracks grid outputs, so an interpolator for that grid is generated as well *automatically* (if the user doesn't override the relevant setting). Finally, it can initialize and run a selected sampler.  Thus, the user only has to specify the grid(s), the constraints, and priors for the resulting model parameters to obtain a sampled posterior distribution.

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

Usage
--------------------
Starlord may be invoked from the command line with `starlord`.  Typically this will be with a settings file defined in the [TOML format](https://toml.io/en/v1.0.0) with the sections `[model]`, `[sampling]`, and `[output]`, and invoked with `starlord run_settings.toml`.  Documentation on the settings file is to-do.  Additional command-line options can be viewed with `starlord --help`.  It may also be employed programmatically with `import starlord` via the `starlord.StarFitter()` class.  That documentation is also to-do -- I meant it when I said "active development".

Roadmap
--------------------
This is a list of features I'd like to add, roughly grouped by priority and definitely subject to change.

<ins>**Alpha Release Goals**</ins>
1. **Core documentation** -- I need to write up the basic documentation on what Starlord is and to use it.  This will probably use Sphinx and be hosted on Github pages (since I'm already using that for dev tools).
2. **Prior rework** -- The current prior system is largely a placeholder awkwardly borrowing from the likelihood class.  Desired features include truncated distributions, priors on transformed parameters, and a clearer summary system.
3. **EMCEE support** -- Currently only Dynesty is supported as a sampler, but nested sampling has a lot of peculiarities that make having a more traditional sampler strongly desirable.

<ins>**Beta Release Goals**</ins>
1. **Data Structure for Constants** -- Currently, all constants are passed as arguments to the logLikelihood function, which works for generated code but makes the function extremely awkward to call directly by the user.  It probably adds overhead to the function calls from Python as well.  I need to build a data structure Cython understands to agglomerate these.
2. **Builtin Sampler** -- Ensemble and nested sampling are powerful techniques, but calling back and forth from Python (including type conversions) adds overhead that matters when the likelihood and prior functions are fairly simple.  Having a builtin Metropolis-Hastings sampler would both allow for fast, pure Cython sampling that is more readily analyzed (due to being older and simpler than the more advanced sampling techniques).
3. **Vector Interpolation** -- Currently all interpolation is done on a single output value.  This is simple and efficient for a small number of outputs, but breaks cache coherency for many output values.  Interpolating low-resolution spectra is a common use-case, so interpolating the vector (of wavelengths for that case) all in one go would be much more efficient; structuring the grid so that the final index is the vector index would preserve cache coherency.

<ins>**Subsequent Goals**</ins>
1. **Vector Operations** -- It is already possible to define operations on vectors in the sampler, but users unfamiliar with Cython will likely struggle to produce efficient code. Some simple looping components with an associated TOML syntax would help a lot.
2. **Multiple Grid Interpolations** -- Currently each grid is interpolated at exactly one point, but this fails to support multi-star systems.  While this can be worked around with manual expressions, it is extremely cumbersome so proper support would be very helpful.
3. **Implicit Variables** -- If mass is defined, then logMass or log_mass are pretty clear in their intent.  Given how often simple transforms are applied to model parameters, it would be nice to directly identify and resolve them without requiring explicit definitions (albeit with the option to shut this off for finer control).
4. **Python interpolation support** -- The grid system is faster than `scipy.RegularGridInterpolator` and very generally applicable, but has currently only been written with usage within Cython in mind.  A handful of convenience methods would go a long way to making it generally useful, such as broadcasting interpolation, composition of multiple grids, and creation of grids from generated parameters.
