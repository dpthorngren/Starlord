Fitting Stars
====================
The primary purpose of Starlord is to flexibly fit stellar models to observations, so this merits a worked example.  We'll use HD 80606 data straight from the `NASA Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/overview/HD%2080606>`_ for simplicity.

.. literalinclude:: /examples/hd80606.toml

We have two examples of fixing grid input parameters, both simplifying approximations. First, we fix the extinction parameter to 0, and second, we set the initial metallicity (for the stellar evolution grid) to the present-day photospheric metallicity. We set a normal prior for the parallax (from the observations), and uniform priors for the remaining parameters.

Model Checks
--------------------
We want to make sure there aren't any syntax errors and that Starlord is interpreting the model the way we expect.  For this we can use the analysis output (``starlord -da hd80606.toml``), which in this case looks like:

.. code:: none

           Grids
    mist 2MASS_H, bc_2MASS_J, bc_WISE_W2, WISE_W4, bc_WISE_W3, WISE_W1, WISE_W2, bc_WISE_W1, 2MASS_Ks, WISE_W3, 2MASS_J, bc_2MASS_Ks, bc_2MASS_H, bc_WISE_W4
    mistTracks logRadius, logL, logMass, logTeff, logG

        Variables
    Params:     eep, feh, logMass0, parallax
    Constants:  grid_mistTracks_logMass, grid_mistTracks_logRadius, grid_mistTracks_logTeff, grid_mist_bc_2MASS_H, grid_mist_bc_2MASS_J, grid_mist_bc_2MASS_Ks, grid_mist_bc_WI
    SE_W1, grid_mist_bc_WISE_W2, grid_mist_bc_WISE_W3, grid_mist_bc_WISE_W4, wise_w4_mean
    Locals:     mistTracks_logG, mistTracks_logL, mistTracks_logMass, mistTracks_logRadius, mistTracks_logTeff, mist_2MASS_H, mist_2MASS_J, mist_2MASS_Ks, mist_WISE_W1, mist_W
    ISE_W2, mist_WISE_W3, mist_WISE_W4, mist_bc_2MASS_H, mist_bc_2MASS_J, mist_bc_2MASS_Ks, mist_bc_WISE_W1, mist_bc_WISE_W2, mist_bc_WISE_W3, mist_bc_WISE_W4

        Forward Model
    l.mistTracks_logMass = c.grid_mistTracks_logMass._interp3d(p.logMass0, p.feh, p.eep)
    l.mistTracks_logRadius = c.grid_mistTracks_logRadius._interp3d(p.logMass0, p.feh, p.eep)
    l.mistTracks_logG = 4.43785118 + l.mistTracks_logMass - 2*l.mistTracks_logRadius
    l.mistTracks_logTeff = c.grid_mistTracks_logTeff._interp3d(p.logMass0, p.feh, p.eep)
    l.mistTracks_logL = 2*(l.mistTracks_logRadius-1) + 4*(l.mistTracks_logTeff - 3.761777)
    l.mist_bc_2MASS_H = c.grid_mist_bc_2MASS_H._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh, 0.0)
    l.mist_2MASS_H = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_2MASS_H - 5*(math.log10(p.parallax)-1)
    l.mist_bc_2MASS_J = c.grid_mist_bc_2MASS_J._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh, 0.0)
    l.mist_2MASS_J = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_2MASS_J - 5*(math.log10(p.parallax)-1)
    l.mist_bc_2MASS_Ks = c.grid_mist_bc_2MASS_Ks._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh, 0.0)
    l.mist_2MASS_Ks = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_2MASS_Ks - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W1 = c.grid_mist_bc_WISE_W1._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh, 0.0)
    l.mist_WISE_W1 = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_WISE_W1 - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W2 = c.grid_mist_bc_WISE_W2._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh, 0.0)
    l.mist_WISE_W2 = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_WISE_W2 - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W3 = c.grid_mist_bc_WISE_W3._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh, 0.0)
    l.mist_WISE_W3 = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_WISE_W3 - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W4 = c.grid_mist_bc_WISE_W4._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh, 0.0)
    l.mist_WISE_W4 = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_WISE_W4 - 5*(math.log10(p.parallax)-1)

        Likelihood
    Normal(l.mist_2MASS_H | 7.4, 0.034)
    Normal(l.mist_2MASS_J | 7.702, 0.03)
    Normal(l.mist_2MASS_Ks | 7.316, 0.02)
    Normal(l.mist_WISE_W1 | 7.203, 0.029)
    Normal(l.mist_WISE_W2 | 7.322, 0.02)
    Normal(l.mist_WISE_W3 | 7.297, 0.018)
    Normal(l.mist_WISE_W4 | c.wise_w4_mean, 0.0114)

        Prior
    Uniform(p.eep | 200.0, 400.0)
    Uniform(p.feh | -0.2, 0.5)
    Uniform(p.logMass0 | -0.5, 0.5)
    Normal(p.parallax | 15.0153, 0.03612)

        Constant Values
    c.wise_w4_mean = 7.196

Notice that Starlord shows you in the ``Forward Model`` section its work to calculate grid inputs, interpolate on the grids, and then apply the bolometric corrections and distance effects.  These were entirely defined within the grid and implicitly invoked when we set a likelihood on the derived outputs of the mist grid.  It also added some constants and local variables for the grids themselves -- Starlord will set these, so you don't need to worry about them.

The key thing to verify is that the likelihoods match your expectations, that you have set a prior for every parameter, and that the constants are properly set.  Now we can test the model.  We can see parameters in order are ``eep``, ``feh``, ``logMass0``, ``parallax``; let's test the values ``(250, 0, 0, 15)`` with ``starlord -dt 250,0,0,15 hd80606.toml``.  It will take a second to compile, but you should get:

.. code:: none

       Test Case
    p.eep                   250.0
    p.feh                   0.0
    p.logMass0              0.0
    p.parallax              15.0
    l.mistTracks_logG       4.53267
    l.mistTracks_logL       -2.11003
    l.mistTracks_logMass    -2.92512e-06
    l.mistTracks_logRadius  -0.0474104
    l.mistTracks_logTeff    3.75797
    l.mist_2MASS_H          7.6958
    l.mist_2MASS_J          8.03596
    l.mist_2MASS_Ks         7.65735
    l.mist_WISE_W1          7.64546
    l.mist_WISE_W2          7.66841
    l.mist_WISE_W3          7.63476
    l.mist_WISE_W4          7.63409
    l.mist_bc_2MASS_H       1.43883
    l.mist_bc_2MASS_J       1.09866
    l.mist_bc_2MASS_Ks      1.47727
    l.mist_bc_WISE_W1       1.48916
    l.mist_bc_WISE_W2       1.46621
    l.mist_bc_WISE_W3       1.49986
    l.mist_bc_WISE_W4       1.50054
    log_like                -1405.96
    log_prior               -2.62939

You can see the parameters first, then the evolution track outputs, then the magnitudes, the bolometric corrections they came were derived from, and the log likelihood and log prior.Well our guess was a little off but everything seems to be calculating properly.

Sampling
--------------------
For the sake of example we explicitly set the sampler as "emcee" and an init and run parameter.  Normally EMCEE requires that you specify an initial state for the walkers but Starlord can just draw that from the priors.  Running the model, we get:

.. code:: none

    100%|█████████████████████████████████████| 4500/4500 [00:01<00:00, 3943.58it/s]
      Name               Mean         Std         16%         50%         84%
    0 eep               351.2       72.71       216.8       388.9       397.7
    1 feh              0.3611      0.1103      0.2473      0.3839      0.4727
    2 logMass0        0.02392     0.02928   -0.000881     0.01403     0.07333
    3 parallax          15.02     0.03589       14.98       15.02       15.05

Wasn't that delightfully fast?  The mass is about what we expected and the metallicity is plausible but wider than what is listed on the NASA exoplanet archive -- presumably a side-effect of choosing the input bands semi-arbitrarily.  We also get an npz file to use for subsequent analysis.  This can be opened in Python with ``post_sample = np.load("hd80606.npz")['samples']``.
