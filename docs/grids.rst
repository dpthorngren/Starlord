Grid Management
====================

The grids used by Starlord are designed for linear interpolation in up to 5 dimensions, as is done with ``scipy.interpolate.RegularGridInterpolator``. Unlike that interpolator, Starlord grids have an associated npz storage schema, which includes grid metadata for Starlord to use in constructing the interpolator and during code generation. The components of a grid are:

:name:          The name of the grid, which is also the name of its npz file.
:inputs:        Up to 5 1-d arrays that define the axes (interpolation inputs) of the grid.
:outputs:       The values to be interpolated as arrays with each dimension matching the length of the associated input.
:derived:       Values that can be calculated from the outputs; the npz file stores the Cython code to generate each value in terms of the grid's outputs and inputs, or even those of other grids.  For :class:`~starlord.ModelBuilder` inputs, these can be treated the same as outputs.
:design:        Describes how the grid is structured as a string containing the inputs, outputs, and derived value names.  E.g. ``x, y, z -> A, B, C; Derived1, Derived2``
:input_mappings:      During code generation, these determine what is used for each input into the interpolator (e.g. what to sub in for ``x``, ``y``, and ``z``).

Grid Creation
--------------------
You can create your own grids with :func:`starlord.GridGenerator.create_grid`.  The name, inputs, and outputs are required, but the others may be omitted if you aren't going to use them.  The function will check its inputs for validity and then generate the metadata for you.  Here's an example of how you can generate a grid:

.. literalinclude:: /examples/demo_grid.py

Usage
--------------------

.. code-block:: python

   import starlord

   # Load from Starlord's known grids list (those in its directory)
   grid1 = starlord.GridGenerator.get_grid("grid1")
   # Load from a particular file
   grid2 = starlord.GridGenerator("./path/to/grid2.npz")

   # Create an interpolator and use it
   get_foo = grid1.build_grid("foo")
   xt = np.array([0.5, 2.3])
   print(xt, get_foo(xt))

.. note::
   A :class:`starlord.GridGenerator` will build :class:`starlord.GridInterpolator` objects.  These are *much* faster than scipy.interpolate options, but are less flexible -- they do not support Numpy-style array broadcasting.  This was an intentional trade-off to achieve maximum sampling speeds.

