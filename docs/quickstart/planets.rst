Fitting Planets
====================
Although Starlord is designed for fitting stellar models to observations, it is perfectly capable of fitting planet evolution models instead.  If anything, planet models are simpler as the mass and/or radius are usually measured directly, rather than inferred from a collection of observed bands.  Existing model grids can be readily input into Starlord by the user, allowing modellers to avoid the laborious process of retrieval setup.

For this example we'll use the hot Jupiter evolution model grids of `Thorngren & Fortney 2018 <https://ui.adsabs.harvard.edu/abs/2018AJ....155..214T/abstract>`_, which is provided in the standard Starlord grids (or download with ``starlord --download hotJupiters``).  This grid is 5 dimensional, in mass, metallicity, incident flux, internal heating power (as a fraction of flux), and age.  The outputs are the specific entropy, luminosity, and radius.

Specifying the Model
--------------------
The first step defining the model to be fitted.  This can be done with the Python API, but we'll use the ``toml`` file approach here (see :doc:`../models`). The likelihood terms are written as ``hotJupiters.output = ["distribution", param1, param2,...]``, and the distribution defaults to ``"normal"``.  For a transiting planet the outputs you want to fit to will generally include the radius and mass.

.. literalinclude:: ../examples/planet.toml
   :language: toml
   :lines: 1-5
   :linenos:

You can already run ``starlord -da planet.toml`` to get a sense of how Starlord will interpret your model.  The variables section is most important -- pay close attention to the ``Params:     p.log_age, p.log_mass, ...`` line; you need to either set priors for every parameter or change the model to not use it by e.g. fixing them (again see `../models`).  Finally we set the priors in the same manner as the likelihood terms were set.

.. literalinclude:: ../examples/planet.toml
   :language: toml
   :lines: 7-13
   :linenos:
   :lineno-start: 7

It can be helpful for Starlord to output additional derived values like the intrinsic temperature even though they aren't parameters of the model.  This is set in the model ``outputs`` list as follows:

.. literalinclude:: ../examples/planet.toml
   :language: toml
   :lines: 15-22
   :linenos:
   :lineno-start: 16

Finally, sampling and output options are set in the ``[sampling]`` and ``[output]`` sections respectively -- we'll just use a simple setup here, see :doc:`../sampling` for more information.

.. literalinclude:: ../examples/planet.toml
   :language: toml
   :lines: 23-30
   :linenos:
   :lineno-start: 23

Running and Reading Outputs
---------------------------
Running the model with ``starlord planet.toml``, we obtain:

.. code:: none

    Pre-run and burn-in done.
    Sampling. done.
    Grid Citations:
        hotJupiters: Thorngren & Fortney (2018; 10.3847/1538-3881/aaba13)
    Convergence Stats:  pseudo_gr = 1.0031
         Name                            Mean         Std         16%         50%         84%
       0 heating                      0.02347    0.005577     0.01806     0.02289     0.02886
       1 log_age                       0.2812      0.4015     -0.1042      0.3988       0.627
       2 log_flux                        9.21     0.09072        9.12       9.209         9.3
       3 log_mass                    -0.05159     0.04804    -0.09901    -0.04893   -0.004114
       4 zpl                           0.1307     0.03864     0.09162       0.129      0.1698
    -----------------------------------------------------------------------------------------
       5 log_like                      0.4513       1.234     -0.7271      0.7129       1.637
       6 log_prior                      5.422      0.9755       4.598       5.719       6.231
       7 hotJupiters__mass             0.8934     0.09736      0.7961      0.8935      0.9906
       8 hotJupiters__age               2.578       1.576      0.7867       2.505       4.236
       9 hotJupiters__entropy           9.833      0.1483       9.685       9.833       9.978
      10 hotJupiters__tint              637.5       42.03       595.8       637.2       679.1
 

The model parameters are listed first, then the logl ikelihood, log prior, and output values requested in the ``toml`` file.  We can see, for example, that this planet was inferred to have a metallicity of 0.1307 +/- 0.03864, and an intrinsic temperature of 637 K.

In the model file we also specified an output file of ``hotJupiter.npz`` -- this was saved in the directory the model was run in.  The data can be loaded in using ``np.load`` or with :func:`starlord.load_to_frame` to obtain a nicely-formatted Pandas data frame of the posterior.

Defining a Model Grid
-----------------------
Users may have their own planet grids they'd like to use, so we'll discuss how to convert grids into the Starlord format (required to fit models to it).  This is also covered in :doc:`../grids`, but will be so common for planet modelling as to merit a tailored example.  This mostly consists of naming the axes, defining any derived parameters, and handing the data to :func:`starlord.GridGenerator.create_grid` for processing.  For this example, we'll assume the data was stored in a csv file and open it with Pandas (in reality it wasn't but this is an important case to cover).

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

