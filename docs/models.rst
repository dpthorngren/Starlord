Defining a Model
====================
Bayesian models are defined by their likelihoods, the model parameters, and the priors thereof.  Starlord's model specification system attempts to cut as much of the boilerplate out of this as possible, especially where there are grids to be interpolated.  This section describes how this specification system works, but you might also find the examples in :doc:`quickstart` and :doc:`stars` helpful, as well as the API documentation for :class:`starlord.ModelBuilder`.

A key feature of Starlord's model specification is the *implicit declaration of variables*.  If you use a variable, Starlord will infer its purpose from the prefix (every valid variable is of the form ```type.varname```) and generate code accordingly.  That is, you do not need to list your model parameters, constants, or grid outputs, you just *use them* and Starlord will handle it.  The types of variables are:

:Parameters: ``p.[name]``, these are model parameters to be sampled from.
:Constants: ``c.[name]``, these are set when the sampler is run and don't
   change.
:Local Variables: ``l.[name]`` these are calculated for each log likelihood call
   but not recorded
:Grid Variables: ``[grid_name].[output_name]``, these indicate a value obtained by interpolating from the specified grid.

These variables are used in defining the model via one of the five valid entries for the ``[model]`` section of the TOML file:

1. `Grid Likelihood Terms`_: ``grid_name.output_name = ['distribution', param_1, param_2, ...]``
2. `Local variables`_: ``var.variable_name = ['code', 'distribution', param_1, param_2, ...]``
3. `Expressions`_: ``expr.unique_identifier = 'code'``
4. `Input Mapping Overrides`_: ``override.grid_name.input_name = 'code'``
5. `Priors`_: ``prior.parameter_name = ['distribution', param_1, param_2, ...]``

Each will be discussed below.  The order within a section is completely irrelevant --- Starlord will sort the inputs so that variables are never used before they are defined.

.. warning::

    A peculiarity of TOML files is that numbers *must not* have leading or trailing decimal points.  That is, ``0.5`` if fine but ``.5`` and ``5.`` are not.

Grid Likelihood Terms
---------------------
The likelihood terms describe how our data constrains the model posterior.  For our above example, you would define the likelihood with:

.. code-block:: toml

    # Implicitly a normal distribution
    ExampleGrid.A = [1.5, 0.2]
    ExampleGrid.B = ['uniform', 0, 5.0]
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

Local Variables
--------------------
Not every likelihood term will be the output of a grid.  Local variables let you compute some new value and use *that* to set a likelihood.  For example, suppose we had a constraint not on ``ExampleGrid.A`` but on its square root.  We could define a new ``sqrt_A`` variable and add a likelihood term for it with:

.. code-block:: toml

   var.sqrt_A = ['math.sqrt(A)', 'normal', 1.5, 0.2]

In the Python API this is split into two function calls:

.. code-block:: python

   builder.assign("l.sqrt_A", "math.sqrt(A)")
   builder.constraint("l.sqrt_A", "normal", [1.5, 0.2])

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

Input Mapping Overrides
-----------------------
When a grid is used, the inputs to the interpolator are determined by the grid ``input_mappings`` metadata.  If this is omitted or lacks an entry for a given input, the default is a new parameter with the name of the input, e.g. ``p.x``.  If you wish to use something else (a constant value, another parameter name, the output variable of another grid, etc.), you can use the input override system.  In the TOML file this looks like:

.. code-block:: toml

   overrides.ExampleGrid.x = "2*p.y"

In Python the setup is nearly the same:

.. code-block:: python

   builder.override("ExampleGrid", "x", "2*p.y")

Either way, the generated code would now interpolate at ``(2*p.y, p.y, p.z)`` instead of ``(p.x, p.y, p.z)``.

Priors
--------------------
Priors are set in a similar manner to grid likelihood terms, except that they start with `prior` instead of the grid name:

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

    When using grids, the parameters are auto-generated from their inputs (see :doc:`grids`), so it might not be immediately obvious what parameters your model uses.  After writing your likelihood terms, Starlord can list them for you either via the command line ``starlord -da your_model.toml`` or in Python with :meth:`ModelBuilder.summary() <starlord.ModelBuilder.summary>`
