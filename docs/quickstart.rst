Quickstart
====================

This page covers the absolute basics of Starlord usage.

Installation
--------------------
It's generally good practice to install this sort of tool in its own virtual environment (here's a `virtual environment tutorial <https://www.w3schools.com/python/python_virtualenv.asp>`_).  Starlord is not yet listed on PyPy, so you'll need to install it from the Github repository.  This can be done with pip:

.. code-block:: console

    $ pip install git+https://github.com/dpthorngren/Starlord.git#egg=starlord

Alternatively, if you'd like to download the git repository somewhere specific, you can go to that directory and use:

.. code-block:: console

    git clone git@github.com:dpthorngren/Starlord.git
    cd Starlord
    pip install .

If you'd like to run the tests, install the test dependencies with ``pip install .[develop]``, then run them with ``pytest`` from the Starlord directory.

Example
--------------------
You can see the grids you have installed with ``starlord -l`` in the command line.  To inspect an individual grid, use ``starlord -l some_grid_name``.  Importantly, this will show the output variables you can put constraints on and the input variables you will need to fix or set priors for.  You can see more command line options by running ``starlord`` with no arguments.

There are two ways to use Starlord: importing it in Python and using the API, or defining a model in a TOML file and passing it to Starlord on the command line via ``starlord my_model.toml``.  The latter is an easy place to start.  The general TOML file specification is defined `here <https://toml.io/en/v1.1.0>`_ but the key thing to understand is that it's a standardized way to specify program settings that are both human and machine readable.  Here's an example of a Starlord TOML file:

.. code:: toml

    [model]
    # This is a comment!
    # The likelihood terms (implicitly normal distributions):
    mist.WISE_W1 = [7.203, 0.029]
    mist.WISE_W2 = [7.322, 0.020]
    mist.WISE_W3 = [7.297, 0.018]
    mist.WISE_W4 = [7.196, 0.0114]

    # Fix the extinction parameter to 0:
    override.mist.Av = 0.

    # The priors for the remaining parameters:
    prior.eep = ["uniform", 200, 400]
    prior.logMass0 = ["uniform", -0.5, 0.5]
    prior.feh0 = ["uniform", -0.2, 0.5]
    prior.logDist = [1.65, 0.02]

    [sampling]
    sampler = "emcee"
    emcee_init.nwalkers = 32
    emcee_run.nsteps = 2000

    [output]
    terminal = true
    file = "hd80606.npz"

Notice that it has three sections: ``model``, ``sampling``, and ``output``.  The first sets up the statistical model to be fitted, in this case four WISE band observations and priors for the model parameters.  Random variables, whether likelihood terms (here the WISE observations) or priors are specified as a list with a distribution name and the parameters.  If no distribution is specified, the normal distribution is assumed.

When you specify a grid variable, Starlord will handle interpolation and selecting the inputs automatically according to the grid's defaults.  This can including interpolating additional grids or defining new parameters.  To know what parameters are inferred for your model, you can use ``starlord my_model.toml --dry-run`` which will print general information about the model, including the list of parameters.  You read more about this section in :doc:`models`.

.. Note:: Every parameter must have a prior. However, you can remove parameters by fixing them to a constant value with ``override.grid_name.input_name = some_value``, as demonstrated above for the extinction parameter `Av`.

The next section, ``sampling`` lets you select the sampler and the options for it.  In this case we have selected the Emcee sampler, and then show how you can set initialization parameters with ``samplername_init.parameter = value`` and run parameters with ``samplername_run.parameter = value``.  You can read more on this in :doc:`sampling`.

Finally, the ``output`` section defines what Starlord should actually do with the data.  For now this is quite minimal -- you can set terminal output on or off and select a file to write the results to in the `npz format <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_.  If you omit this, no output file is written.  In future versions this section will specify additional output values to compute and report/save in addition to the model parameters.
