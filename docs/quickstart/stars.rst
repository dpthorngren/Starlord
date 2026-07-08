Fitting Stars
====================
The primary purpose of Starlord is to flexibly fit stellar models to observations, so this merits a worked example.  We'll use HD 80606 data straight from the `NASA Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/overview/HD%2080606>`_ for simplicity.

.. literalinclude:: /examples/hd80606.toml
   :language: toml

We have two examples of fixing grid input parameters, both simplifying approximations. First, we fix the extinction parameter to 0, and second, we set the alpha-to-iron ratio to zero. We set a power-law prior for the distance, a builtin prior for [Fe/H] from `Casagrande (2011) <https://ui.adsabs.harvard.edu/abs/2011A%26A...530A.138C/abstract>`, another builtin prior for mass from `Chabrier (2003) <https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract>`, and a uniform prior for the log-age.

Model Checks
--------------------
We want to make sure there aren't any syntax errors and that Starlord is interpreting the model the way we expect.  For this we can use the analysis output (``starlord -da hd80606.toml``), which in this case looks like:

.. code:: none

        Grids
    mist2 TWOMASS_H, TWOMASS_J, TWOMASS_Ks, WISE_W1, WISE_W2, WISE_W3, WISE_W4, bc_TWOMASS_H, bc_TWOMASS_J, bc_TWOMASS_Ks, bc_WISE_W1, bc_WISE_W2, bc_WISE_W3, bc_WISE_W4, distance, parallax
    mist2InvAge eep
    mist2Tracks log_g, log_lum, log_mass, log_radius, log_teff

        Forward Model
    v.mist2InvAge__eep = c.grid__mist2InvAge__eep._interp4d(p.log_mass0, p.feh, 0.0, p.log_age)
    v.mist2Tracks__log_radius = c.grid__mist2Tracks__log_radius._interp4d(p.log_mass0, p.feh, 0.0, v.mist2InvAge__eep)
    v.mist2Tracks__log_teff = c.grid__mist2Tracks__log_teff._interp4d(p.log_mass0, p.feh, 0.0, v.mist2InvAge__eep)
    v.mist2Tracks__log_lum = -15.045182 + 2*(v.mist2Tracks__log_radius) + 4*(v.mist2Tracks__log_teff)
    v.mist2Tracks__log_mass = c.grid__mist2Tracks__log_mass._interp4d(p.log_mass0, p.feh, 0.0, v.mist2InvAge__eep)
    v.mist2Tracks__log_g = 4.43785118 + v.mist2Tracks__log_mass - 2*v.mist2Tracks__log_radius
    v.mist2__bc_WISE_W2 = c.grid__mist2__bc_WISE_W2._interp5d(v.mist2Tracks__log_teff, v.mist2Tracks__log_g, p.feh, 0.0, 0.0)
    v.mist2__distance = p.distance
    v.mist2__WISE_W2 = -2.5 * v.mist2Tracks__log_lum - 0.26 - v.mist2__bc_WISE_W2 + 5*math.log10(v.mist2__distance)
    v.mist2__bc_WISE_W4 = c.grid__mist2__bc_WISE_W4._interp5d(v.mist2Tracks__log_teff, v.mist2Tracks__log_g, p.feh, 0.0, 0.0)
    v.mist2__WISE_W4 = -2.5 * v.mist2Tracks__log_lum - 0.26 - v.mist2__bc_WISE_W4 + 5*math.log10(v.mist2__distance)
    v.mist2__parallax = 1000 / (v.mist2__distance)
    v.mist2__bc_TWOMASS_J = c.grid__mist2__bc_TWOMASS_J._interp5d(v.mist2Tracks__log_teff, v.mist2Tracks__log_g, p.feh, 0.0, 0.0)
    v.mist2__TWOMASS_J = -2.5 * v.mist2Tracks__log_lum - 0.26 - v.mist2__bc_TWOMASS_J + 5*math.log10(v.mist2__distance)
    v.mist2__bc_TWOMASS_Ks = c.grid__mist2__bc_TWOMASS_Ks._interp5d(v.mist2Tracks__log_teff, v.mist2Tracks__log_g, p.feh, 0.0, 0.0)
    v.mist2__TWOMASS_Ks = -2.5 * v.mist2Tracks__log_lum - 0.26 - v.mist2__bc_TWOMASS_Ks + 5*math.log10(v.mist2__distance)
    v.mist2__bc_TWOMASS_H = c.grid__mist2__bc_TWOMASS_H._interp5d(v.mist2Tracks__log_teff, v.mist2Tracks__log_g, p.feh, 0.0, 0.0)
    v.mist2__TWOMASS_H = -2.5 * v.mist2Tracks__log_lum - 0.26 - v.mist2__bc_TWOMASS_H + 5*math.log10(v.mist2__distance)
    v.mist2__bc_WISE_W1 = c.grid__mist2__bc_WISE_W1._interp5d(v.mist2Tracks__log_teff, v.mist2Tracks__log_g, p.feh, 0.0, 0.0)
    v.mist2__WISE_W1 = -2.5 * v.mist2Tracks__log_lum - 0.26 - v.mist2__bc_WISE_W1 + 5*math.log10(v.mist2__distance)
    v.mist2__bc_WISE_W3 = c.grid__mist2__bc_WISE_W3._interp5d(v.mist2Tracks__log_teff, v.mist2Tracks__log_g, p.feh, 0.0, 0.0)
    v.mist2__WISE_W3 = -2.5 * v.mist2Tracks__log_lum - 0.26 - v.mist2__bc_WISE_W3 + 5*math.log10(v.mist2__distance)

        Likelihood
    Normal(v.mist2__TWOMASS_H | 7.4, 0.034)
    Normal(v.mist2__TWOMASS_J | 7.702, 0.03)
    Normal(v.mist2__TWOMASS_Ks | 7.316, 0.02)
    Normal(v.mist2__WISE_W1 | 7.203, 0.029)
    Normal(v.mist2__WISE_W2 | 7.322, 0.02)
    Normal(v.mist2__WISE_W3 | 7.297, 0.018)
    Normal(v.mist2__WISE_W4 | {c.wise_w4_mean}, 0.0114)
    Normal(v.mist2__parallax | 15.0153, 0.03612)

        Prior
    Trunc_Power(p.distance | -2.0, 10.0, 200.0)
    Binorm(p.feh | 0.8, 0.016, -0.15, 0.15, 0.22)
    Uniform(p.log_age | -0.5, 1.0)
    Chabrier(p.log_mass0 | 0.0, -1.10237, 0.69, 5.295945)

        Variables
    Params:     p.distance, p.feh, p.log_age, p.log_mass0
    Constants:  c.grid__mist2InvAge__eep, c.grid__mist2Tracks__log_mass, c.grid__mist2Tracks__log_radius, c.grid__mist2Tracks__log_teff, c.grid__mist2__bc_TWOMASS_H, c.grid__mist2__bc_TWOMASS_J, c.grid__mist2__bc_TWOMASS_Ks, c.grid__mist2
    __bc_WISE_W1, c.grid__mist2__bc_WISE_W2, c.grid__mist2__bc_WISE_W3, c.grid__mist2__bc_WISE_W4, c.wise_w4_mean
    Locals:     v.mist2InvAge__eep, v.mist2Tracks__log_g, v.mist2Tracks__log_lum, v.mist2Tracks__log_mass, v.mist2Tracks__log_radius, v.mist2Tracks__log_teff, v.mist2__TWOMASS_H, v.mist2__TWOMASS_J, v.mist2__TWOMASS_Ks, v.mist2__WISE_W1, 
    v.mist2__WISE_W2, v.mist2__WISE_W3, v.mist2__WISE_W4, v.mist2__bc_TWOMASS_H, v.mist2__bc_TWOMASS_J, v.mist2__bc_TWOMASS_Ks, v.mist2__bc_WISE_W1, v.mist2__bc_WISE_W2, v.mist2__bc_WISE_W3, v.mist2__bc_WISE_W4, v.mist2__distance, v.mist2
    __parallax

        Constant Values
    c.wise_w4_mean = 7.196


Notice that Starlord shows you in the ``Forward Model`` section its work to calculate grid inputs, interpolate on the grids, and then apply the bolometric corrections and distance effects.  These were entirely defined within the grid and implicitly invoked when we set a likelihood on the derived outputs of the mist grid.  It also added some constants and local variables for the grids themselves -- Starlord will set these, so you don't need to worry about them.

The key thing to verify is that the likelihoods match your expectations, that you have set a prior for every parameter, and that the constants are properly set.  Now we can test the model.  We can see parameters in order are ``distance``, ``feh``, ``log_age``, ``log_mass0``; let's test the values ``(65, 0, 0.3, 0)`` with ``starlord -dt 65,0,.3,0 hd80606.toml``.  It will take a second to compile, but you should get:

.. code:: none

    p.distance                 65.0
    p.feh                      0.0
    p.log_age                  0.3
    p.log_mass0                0.0
    v.mist2InvAge__eep         323.503
    v.mist2Tracks__log_g       4.49844
    v.mist2Tracks__log_lum     -0.0616264
    v.mist2Tracks__log_mass    -2.58637e-05
    v.mist2Tracks__log_radius  -0.0303056
    v.mist2Tracks__log_teff    3.76104
    v.mist2__TWOMASS_H         7.52015
    v.mist2__TWOMASS_J         7.85203
    v.mist2__TWOMASS_Ks        7.47578
    v.mist2__WISE_W1           7.45961
    v.mist2__WISE_W2           7.49128
    v.mist2__WISE_W3           7.45061
    v.mist2__WISE_W4           7.44897
    v.mist2__bc_TWOMASS_H      1.43848
    v.mist2__bc_TWOMASS_J      1.1066
    v.mist2__bc_TWOMASS_Ks     1.48286
    v.mist2__bc_WISE_W1        1.49902
    v.mist2__bc_WISE_W2        1.46736
    v.mist2__bc_WISE_W3        1.50802
    v.mist2__bc_WISE_W4        1.50966
    v.mist2__distance          65.0
    v.mist2__parallax          15.3846
    log_like                   -437.805
    log_prior                  -7.32281

You can see the parameters first, then the evolution track outputs, then the magnitudes, the bolometric corrections they came were derived from, and the log likelihood and log prior.  Well our guess was a little off but everything seems to be calculating properly.

Sampling
--------------------
For the sake of example we explicitly set the sampler as "builtin" and an init and run parameter.  Running the model with ``starlord hd80606.toml``, we get:

.. code:: none

    Pre-run and burn-in done.
    Sampling. done.
    Grid Citations:
        mist2InvAge: [See mist2 grid citations.]
        mist2Tracks: [See mist2 grid citations.]
        mist2: Dotter (2016; 10.3847/0067-0049/222/1/8), Choi et al. (2016; 10.3847/0004-637X/823/2/102), Dotter et al. (2026; 10.3847/1538-4365/ae48f3), Bauer et al. (2026; 10.3847/1538-4365/ae401e)
         Name                            Mean         Std         16%         50%         84%
       0 distance                       66.59      0.1594       66.43       66.59       66.75
       1 feh                          0.05859      0.1486    -0.08629      0.0648      0.2033
       2 log_age                       0.9186     0.08639      0.8517      0.9458      0.9868
       3 log_mass0                   -0.01979     0.01499    -0.03414    -0.02029   -0.005853
    -----------------------------------------------------------------------------------------
       4 log_like                     -0.5589       1.512      -1.937     -0.2356      0.8231
       5 log_prior                     -7.817      0.6701      -8.309      -7.559      -7.342
       6 mist2Tracks__log_radius      0.02337     0.00413     0.01933      0.0236     0.02742
       7 mist2Tracks__log_g             4.371      0.0153       4.358       4.368       4.385
       8 mist2Tracks__log_lum         0.02397      0.0219    0.002495     0.02267     0.04523

Wasn't that delightfully fast?  The mass is about what we expected and the metallicity is plausible but wider than what is listed on the NASA exoplanet archive -- presumably a side-effect of choosing the input bands semi-arbitrarily.  We also get an npz file to use for subsequent analysis.  This can be opened in Python with :func:`load_to_frame` or directly by numpy with ``post_sample = np.load("hd80606.npz")['samples']``.
