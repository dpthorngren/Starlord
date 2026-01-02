Quickstart
====================
This page covers the absolute basics of Starlord usage.

Installation
--------------------
It's generally good practice to install this sort of tool in its own virtual environment (here's a `virtual environment tutorial <https://www.w3schools.com/python/python_virtualenv.asp>`_).  Starlord is not yet listed on PyPy, so you'll need to install it from the Github repository.  This can be done with pip:

.. code-block:: console

    pip install git+https://github.com/dpthorngren/Starlord.git#egg=starlord

Alternatively, if you'd like to download the git repository somewhere specific, you can go to that directory and use:

.. code-block:: console

    git clone git@github.com:dpthorngren/Starlord.git
    cd Starlord
    pip install .

If you'd like to run the tests, install the test dependencies with ``pip install .[develop]``, then run them with ``pytest`` from the Starlord directory.  Either way, you should now be able to run Starlord with ``starlord`` in the command line.

.. tip::

   You can see a list of starlord command line options and their use with ``starlord --help``, and the Starlord :doc:`ref/index` can also be viewed with ``python -m pydoc starlord``.

TOML Input Files
--------------------
There are two ways to use Starlord: importing it in Python and using the API, or defining a model in a TOML file and passing it to Starlord on the command line.  The latter is an easier place to start.  Here's a minimal example:

.. warning::
   todo example

Notice that it has three sections: ``model``, ``sampling``, and ``output``.  The first sets up the statistical model to be fitted, including any likelihood terms, priors, and intermediate variable definitions (see :doc:`models`). The next section, ``sampling`` lets you select the sampler and the options for it, as well as run-time constants (see :doc:`sampling`). Finally, the ``output`` section defines what Starlord should actually do with the data.  For now this is quite minimal -- you can set terminal output on or off and select a file to write the results to in the `npz format <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_.  If you omit this, no output file is written.

.. note::
    When you specify a grid variable, Starlord will handle interpolation and set inputs automatically according to the grid's defaults.  This can including interpolating additional grids or defining new parameters.  Use ``starlord -da your_model.toml`` to see how Starlord interprets your model.

Recommended Workflow
--------------------
Starlord can do fairly complicated things with your model specification, so we suggest this workflow to 

1. Select or create the grid you wish to work with (see :doc:`grids`).  Use ``starlord -g`` to see available grids and ``starlord -g grid_name`` to examine one in detail.
2. Create a TOML file and list your constraints under the heading ``[models]`` like ``grid.var = ['normal', 3.4, 2.0]`` (distribution and parameters, see :doc:`models`).
3. Check that your model is being interpreted as you expect with ``starlord -da your_model.toml``, and review the list of parameters.
4. For each model parameter, set a prior with ``prior.param_name = ['normal', 2.3, 5.3]``
5. Check your model with a test case using ``starlord -dt 1.,2.,3. your_model.toml``; parameters are comma-separated with no spaces, in order listed by step 3.
6. Run the sampler with ``starlord your_model.toml -o samples.npz``.

.. note::

   This mentions only the basic options.  See :doc:`sampling` for configuring the sampler and :doc:`models` for other modelling options like grid input overrides and runtime constants.
