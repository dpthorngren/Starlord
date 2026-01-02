Defining a Model
====================

.. warning::

   This page is not yet finished.

Here I'll describe how the model definition system works, beginning with a basic Bayesian model to illustrate the sort problems we're trying to solve.

1. Describe variable and grid system.
2. Component system and sorting.
3. Parameter overrides

.. Notice that we can directly refer to the grid and variable; organizing the grid interpolation and required input parameters are one of the major conveniences offered by Starlord.

Likelihood Terms
--------------------
The likelihood terms describe how our data constrains the model posterior.  For our above example, you would define the likelihood with:

.. code-block:: toml

    ExampleGrid.A = ['normal', 1.5, 0.2]
    ExampleGrid.B = ['normal', 2.3, 0.15]
    ExampleGrid.C = ['normal', -0.5, 0.05]

The syntax is ``grid_name.output_name = ['distribution_name', param_1, param_2, ...]``, and the supported distributions are ``normal``, ``uniform``, ``gamma``, and ``beta``.

.. tip::

    You can see the grids available and their outputs with the the terminal command ``starlord -g``.
    
If you're building a model using the Python API (see :doc:`ref/index`), the function :meth:`ModelBuilder.constraint() <starlord.ModelBuilder.constraint>` is the corresponding function.

.. code-block:: python

    builder = starlord.ModelBuilder()
    builder.constraint('ExampleGrid.A', 'normal', [1.5, 0.2])
    builder.constraint('ExampleGrid.B', 'normal', [2.3, 0.15])
    builder.constraint('ExampleGrid.C', 'normal', [-0.5, 0.05])

Priors
--------------------
Priors are set in a similar manner, except that they start with `prior` instead of the grid name:

.. code-block:: toml

    prior.x = ['normal', 0, 10.0]
    prior.y = ['uniform', 0, 1]
    prior.z = ['normal', 5.32, 0.5]

The Python API equivalent is :meth:`ModelBuilder.prior() <starlord.ModelBuilder.prior>`:

.. code-block:: python

    builder.prior("p.x", 'normal', [0, 10.0)
    builder.prior("p.x", 'uniform', [0, 1])
    builder.prior("p.x", 'normal', [5.32, 0.5])

.. tip::

    When using grids, the parameters are auto-generated from their inputs (see :doc:`grids`), so it might not be immediately obvious what parameters your model uses.  After writing your likelihood terms, Starlord can list them for you either via the command line ``starlord --dry-run your_model.toml`` or in Python with :meth:`ModelBuilder.summary() <starlord.ModelBuilder.summary>`


Local Variables
--------------------

Expressions
--------------------
In some cases, you want to do something that doesn't fall into the other categories.  The expression component allows you to insert Cython code directly into the likelihood function.  Note that variable name resolution still occurs, and Starlord will still sort these components according to their dependencies (if any).  As an example, if you'd like to debug by spamming to the terminal, here's how you might do it:

.. code-block:: toml

    expr.x = """
    print(p.x, p.y, p.z)
    print(ExampleGrid.A, ExampleGrid.B, ExampleGrid.C)"
    """

Or in Python:

.. code-block:: python

    builder.expression("""
    print(p.x, p.y, p.z)
    print(ExampleGrid.A, ExampleGrid.B, ExampleGrid.C)"
    """)
