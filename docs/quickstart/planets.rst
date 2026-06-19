Fitting Planets
====================
Although Starlord is designed for fitting stellar models to observations, it is perfectly capable of fitting planet evolution models instead.  If anything, planet models are simpler as the mass and/or radius are usually measured directly, rather than inferred from a collection of observed bands.  Existing model grids can be readily input into Starlord by the user, allowing modellers to avoid the laborious process of retrieval setup.

For illustrative purposes we'll use the hot Jupiter evolution model grids of `Thorngren & Fortney 2018 <https://ui.adsabs.harvard.edu/abs/2018AJ....155..214T/abstract>`_; these will be made publicly available when the Starlord built-in grid repository is setup, but for now may be obtained by contacting the developers.  This grid is 5 dimensional, in mass, metallicity, incident flux, internal heating power (as a fraction of flux), and age.  The outputs are the specific entropy, luminosity, and radius.

Defining the Model Grid
-----------------------
We'll start by showing how to convert your planet grid into the Starlord format (required to fit models to it).  This is also covered in :doc:`../grids`, but will be so common for planet modelling as to merit a tailored example.  This mostly consists of naming the axes, defining any derived parameters, and handing the data to :func:`starlord.GridGenerator.create_grid` for processing.  For this example, we'll assume the data was stored in a csv file and open it with Pandas (in reality it wasn't but this is an important case to cover).

.. literalinclude:: ../examples/grid_hot_jupiters.py
   :language: python

The steps here are:

1. Load your data from whatever format it is saved in.
2. Rearrange it so that the input axes are 1d and sorted and the outputs are nd-arrays arranged so that each axis has the same length as the corresponding input axis.
3. Define the input names with an ``OrderedDict``, optionally transforming axes (e.g. taking the log) for better interpolation.
4. Name the output axes with a ``dict``, again optionally transforming them.
5. If desired, define formulas for derived parameters as Cython code, using ``math`` for math operations and ``d.gridname.input_or_output`` to refer to other grid data.
6. Dump all of this along with a grid name into :func:`starlord.GridGenerator.create_grid`.

If all went well, you should see your new grid listed when you run ``starlord -g``.  You would be wise to run ``starlord -g gridname`` and double-check that the axes, minima, maxima, and lengths all make sense to you -- it's very easy to make unit errors here.  For this example, you would see:

.. code:: none

    Grid hotJupiters
        Input                       Min        Max     Length     Default Mapping
      0 log_mass                 -1.326      1.301         30     p.log_mass--i
      1 zpl                           0          1         10     p.zpl--i
      2 log_flux                      6         11         30     p.log_flux--i
      3 heating                       0       0.05         30     p.heating--i
      4 log_age                      -1      1.176         12     p.log_age--i
    Outputs
        Output                      Min        Max
      5 entropy                   6.148         14
      6 log_luminosity           -42.73      32.65
      7 log_radius              -0.6131      2.109
    Derived
        Derived              Code
      8 age                  10**g.hotJupiters.log_age
      9 flux                 10**g.hotJupiters.log_flux
     10 luminosity           10**g.hotJupiters.log_luminosity
     11 mass                 10**g.hotJupiters.log_mass
     12 radius               10**g.hotJupiters.log_radius
     13 tint                 math.pow(g.hotJupiters.luminosity / (7.125593e-4 * (g.hotJupiters.radius * 6.991 ...
     14 typical_heating      0.0237 * math.exp(-(g.hotJupiters.log_flux - 9.14)**2 / (2 * .37**2))

Specifying the Model
--------------------
To actually fit a planet using this grid, you'll need to define a model.  This can be done with the Python API, but we'll use the ``toml`` file approach here (see :doc:`../models`).  You should start by writing down your likelihood terms as ``gridname.output = ["distribution", param1, param2,...]``.  For a transiting planet this will generally include the radius and mass.

.. literalinclude:: ../examples/planet.toml
   :language: toml
   :lines: 1-5
   :linenos:

You can already run ``starlord -da planet.toml`` to get a sense of how Starlord will interpret your model.  Pay close attention to the ``Params:     p.log_age, p.log_mass, ...`` line; you need to either set priors for every parameter or change the model to not use it by e.g. fixing them.  Below, we set ``log_flux`` to a constant value and fix ``heating`` to a formula defined in the grid (see above) which is itself a function of flux (you could also have directly input the formula on this line).  Finally we set the priors in the same manner as the likelihood terms were set.

.. literalinclude:: ../examples/planet.toml
   :language: toml
   :lines: 6-16
   :linenos:
   :lineno-start: 6

It can be helpful for Starlord to output additional derived values like the intrinsic temperature even though they aren't parameters of the model.  This is set in the model ``outputs`` section as shown.  Finally, sampling and output options are set in the ``[sampling]`` and ``[output]`` sections respectively -- we'll just use a simple setup here, see :doc:`../sampling` for more information.

.. literalinclude:: ../examples/planet.toml
   :language: toml
   :lines: 16-32
   :linenos:
   :lineno-start: 16

Running and Reading Outputs
---------------------------
Running the model with ``starlord planet.toml``, we obtain:

.. code:: none

    Pre-run and burn-in done.
    Sampling. done.
         Name                            Mean         Std         16%         50%         84%
       0 log_age                       0.2871      0.3986    -0.08877      0.4047      0.6274
       1 log_mass                    -0.05023     0.04902    -0.09838    -0.04799   -0.001777
       2 zpl                           0.1247     0.03185     0.09322       0.124      0.1565
    -----------------------------------------------------------------------------------------
       3 log_like                      0.4117       1.271     -0.7866      0.6778       1.632
       4 log_prior                    -0.6931   1.382e-12     -0.6931     -0.6931     -0.6931
       5 hotJupiters__mass             0.8964     0.09973      0.7973      0.8954      0.9959
       6 hotJupiters__age                 2.6       1.574      0.8151       2.539       4.241
       7 hotJupiters__entropy           9.816     0.04124       9.776       9.813       9.857
       8 hotJupiters__tint              636.1     0.09113         636       636.1       636.1

The three model parameters are listed first, then the log_likelihood, log_prior, and output values requested in the ``toml`` file.  We can see, for example, that this planet was inferred to have a metallicity of 0.1247 +/- 0.032, and an intrinsic temperature of 636 K.  Because we fixed the flux and heating, the uncertainties on the latter are very small -- relaxing that assumption would get us more realistic uncertainties.

In the model file we also specified an output file of ``hotJupiter.npz`` -- this was saved in the directory the model was run in.  The data can be loaded in using ``np.load`` or with :func:`starlord.load_to_frame` to obtain a nicely-formatted Pandas data frame of the posterior.
