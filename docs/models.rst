Defining a Model
====================

Here I'll describe how the model definition system works, beginning with a basic Bayesian model to illustrate the sort problems we're trying to solve.

TODO:

1. Describe variable and grid system.
2. Component system and sorting.
3. Parameter overrides
4. Move example to its own page, reference from here

.. Notice that we can directly refer to the grid and variable; organizing the grid interpolation and required input parameters are one of the major conveniences offered by Starlord.

Basic Bayesian Model
--------------------
Suppose you are interested in some parameters `x`, `y`, and `z`, but you can't measure them directly. However, you can measure some functions of these variables up to some normally-distributed uncertainty:

.. math::
    A(x,y,z) &= 1.5 \pm 0.2 \\
    B(x,y,z) &= 2.3 \pm 0.15 \\
    C(x,y,z) &= -0.5 \pm 0.05

This kind of problem pops up a lot in astronomy. For example, we often cannot measure a star's mass, metallicity, etc. directly, but we can it's brightness at different wavelengths, which depend on the those unmeasured properties in complicated ways.  Complicating matters, these functions can be expensive to compute, so it can help to run the models across a grid of input values and interpolate them.  Starlord is focused on using these grids (see :doc:`grids`), so let's pretend you've already set up a grid that looks like this:

.. math::
   \mathrm{ExampleGrid} (x, y, z \rightarrow A, B, C)

A Bayesian interpretation of this situation is that we have observed random variables :math:`A_\mathrm{obs} = 1.5`, :math:`B_\mathrm{obs} = 2.3`, :math:`C_\mathrm{obs} = -0.5`, which were generated from a normal distribution centered on their true values.  These true values were in turn generated :abbr:`exactly (You can add uncertainty onto the functions themselves too, but we'll keep things simple here.)` from their functions of `x`, `y`, and `z` .  If our observations are independent, the likelihood of obtaining these observations given the model parameters is:

.. math::
   p(A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs} | x, y, z) = \mathcal{N}(A(x,y,z), \sigma_A) \times \mathcal{N}(B(x,y,z), \sigma_B) \times \mathcal{N}(C(x,y,z), \sigma_C)

This is really just restating things in fancy stats notation.  Notice that there are three multiplicative terms here, one for each observation.

Still, what we *really* want is the probability distributions for `x`, `y`, and `z`, given our observations.  Bayes Theorem gets us there:

.. math::
   p(x, y, z | A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs}) \propto p(x, y, z | A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs}) \times p(x, y, z)

That is, the posterior probability is :abbr:`proportional to (You don't need to worry about the normalizing constant unless you're trying to use Bayes factors for model comparison.  For that you'll want to use a tool like nested sampling rather than trying to calculate it by hand, as it is usually intractable.)` the likelihood times the prior.  Assuming our priors are independent, we can split them up and write the posterior as:

.. math::
   p(x, y, z | A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs}) \propto\; &\mathcal{N}(A(x,y,z), \sigma_A) \times \mathcal{N}(B(x,y,z), \sigma_B) \times \mathcal{N}(C(x,y,z), \sigma_C) \\ &\times p(x) \times p(y) \times p(z)

So in the end our model was simple enough that we're just multiplying the three likelihood terms and the three prior terms.  Now we just need to sample the distribution.

Likelihood Terms
--------------------
The likelihood terms describe how our data constrains the model posterior.  For our above example, you would define the likelihood with:

.. code-block:: toml

    ExampleGrid.A = ['normal', 1.5, 0.2]
    ExampleGrid.B = ['normal', 2.3, 0.15]
    ExampleGrid.C = ['normal', -0.5, 0.05]

The syntax is ``grid_name.output_name = ['distribution_name', param_1, param_2, ...]``, and the supported distributions are ``normal``, ``uniform``, ``gamma``, and ``beta``.

.. tip::

    You can see the grids available and their outputs with the the terminal command ``starlord -l``.
    
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
