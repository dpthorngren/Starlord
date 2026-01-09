Fitting Stars
====================
The primary purpose of Starlord is to flexibly fit stellar models to observations, so this merits a worked example.  We'll use HD 80606 data straight from the `NASA Exoplanet Archive <https://exoplanetarchive.ipac.caltech.edu/overview/HD%2080606>`_ for simplicity.

.. literalinclude:: /examples/hd80606.toml
   :language: toml

We have two examples of fixing grid input parameters, both simplifying approximations. First, we fix the extinction parameter to 0, and second, we set the initial metallicity (for the stellar evolution grid) to the present-day photospheric metallicity. We set a normal prior for the parallax (from the observations), and uniform priors for the remaining parameters.

Model Checks
--------------------
We want to make sure there aren't any syntax errors and that Starlord is interpreting the model the way we expect.  For this we can use the analysis output (``starlord -da hd80606.toml``), which in this case looks like:

.. code:: none

        Grids
    mist 2MASS_H, 2MASS_J, 2MASS_Ks, WISE_W1, WISE_W2, WISE_W3, WISE_W4, bc_2MASS_H, bc_2MASS_J, bc_2MASS_Ks, bc_WISE_W1, bc_WISE_W2, bc_WISE_W3, bc_WISE_W4
    mistInvAge eep
    mistTracks logG, logL, logMass, logRadius, logTeff

        Variables
    Params:     feh0, logAge, logMass0, parallax
    Constants:  grid_mistInvAge_eep, grid_mistTracks_logMass, grid_mistTracks_logRadius, grid_mistTracks_logTeff, grid_mist_bc_2MASS_H, grid_mist_bc_2MASS_J, grid_mist_bc_2MASS_Ks, grid_mist_bc_WISE_W1, grid_mist_bc_WISE_W2, grid_mist_bc_WISE_W3, grid_mist_bc_WISE_W4, wise_w4_mean
    Locals:     mistInvAge_eep, mistTracks_logG, mistTracks_logL, mistTracks_logMass, mistTracks_logRadius, mistTracks_logTeff, mist_2MASS_H, mist_2MASS_J, mist_2MASS_Ks, mist_WISE_W1, mist_WISE_W2, mist_WISE_W3, mist_WISE_W4, mist_bc_2MASS_H, mist_bc_2MASS_J, mist_bc_2MASS_Ks, mist_bc_WISE_W1, mist_bc_WISE_W2, mist_bc_WISE_W3, mist_bc_WISE_W4

        Forward Model
    l.mistInvAge_eep = c.grid_mistInvAge_eep._interp3d(p.logMass0, p.feh0, p.logAge)
    l.mistTracks_logMass = c.grid_mistTracks_logMass._interp3d(p.logMass0, p.feh0, l.mistInvAge_eep)
    l.mistTracks_logRadius = c.grid_mistTracks_logRadius._interp3d(p.logMass0, p.feh0, l.mistInvAge_eep)
    l.mistTracks_logG = 4.43785118 + l.mistTracks_logMass - 2*l.mistTracks_logRadius
    l.mistTracks_logTeff = c.grid_mistTracks_logTeff._interp3d(p.logMass0, p.feh0, l.mistInvAge_eep)
    l.mistTracks_logL = 2*(l.mistTracks_logRadius-1) + 4*(l.mistTracks_logTeff - 3.761777)
    l.mist_bc_2MASS_H = c.grid_mist_bc_2MASS_H._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh0, 0.0)
    l.mist_2MASS_H = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_2MASS_H - 5*(math.log10(p.parallax)-1)
    l.mist_bc_2MASS_J = c.grid_mist_bc_2MASS_J._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh0, 0.0)
    l.mist_2MASS_J = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_2MASS_J - 5*(math.log10(p.parallax)-1)
    l.mist_bc_2MASS_Ks = c.grid_mist_bc_2MASS_Ks._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh0, 0.0)
    l.mist_2MASS_Ks = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_2MASS_Ks - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W1 = c.grid_mist_bc_WISE_W1._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh0, 0.0)
    l.mist_WISE_W1 = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_WISE_W1 - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W2 = c.grid_mist_bc_WISE_W2._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh0, 0.0)
    l.mist_WISE_W2 = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_WISE_W2 - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W3 = c.grid_mist_bc_WISE_W3._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh0, 0.0)
    l.mist_WISE_W3 = -2.5 * l.mistTracks_logL + 4.74 - l.mist_bc_WISE_W3 - 5*(math.log10(p.parallax)-1)
    l.mist_bc_WISE_W4 = c.grid_mist_bc_WISE_W4._interp4d(l.mistTracks_logTeff, l.mistTracks_logG, p.feh0, 0.0)
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
    Uniform(p.feh0 | -0.2, 0.5)
    Uniform(p.logAge | -0.5, 1.0)
    Uniform(p.logMass0 | -0.5, 0.5)
    Normal(p.parallax | 15.0153, 0.03612)

        Constant Values
    c.wise_w4_mean = 7.196


Notice that Starlord shows you in the ``Forward Model`` section its work to calculate grid inputs, interpolate on the grids, and then apply the bolometric corrections and distance effects.  These were entirely defined within the grid and implicitly invoked when we set a likelihood on the derived outputs of the mist grid.  It also added some constants and local variables for the grids themselves -- Starlord will set these, so you don't need to worry about them.

The key thing to verify is that the likelihoods match your expectations, that you have set a prior for every parameter, and that the constants are properly set.  Now we can test the model.  We can see parameters in order are ``feh``, ``logAge``, ``logMass0``, ``parallax``; let's test the values ``(0, 0.3, 0, 15)`` with ``starlord -dt 0,.3,0,15 hd80606.toml``.  It will take a second to compile, but you should get:

.. code:: none

        Test Case
    p.feh0                  0.0
    p.logAge                0.3
    p.logMass0              0.0
    p.parallax              15.0
    l.mistInvAge_eep        325.56
    l.mistTracks_logG       4.49019
    l.mistTracks_logL       -2.05148
    l.mistTracks_logMass    -2.67794e-05
    l.mistTracks_logRadius  -0.0261845
    l.mistTracks_logTeff    3.762
    l.mist_2MASS_H          7.57698
    l.mist_2MASS_J          7.90794
    l.mist_2MASS_Ks         7.54013
    l.mist_WISE_W1          7.52884
    l.mist_WISE_W2          7.5497
    l.mist_WISE_W3          7.51846
    l.mist_WISE_W4          7.51794
    l.mist_bc_2MASS_H       1.41127
    l.mist_bc_2MASS_J       1.0803
    l.mist_bc_2MASS_Ks      1.44812
    l.mist_bc_WISE_W1       1.4594
    l.mist_bc_WISE_W2       1.43855
    l.mist_bc_WISE_W3       1.46979
    l.mist_bc_WISE_W4       1.47031
    log_like                -681.972
    log_prior               2.26347

You can see the parameters first, then the evolution track outputs, then the magnitudes, the bolometric corrections they came were derived from, and the log likelihood and log prior.  Well our guess was a little off but everything seems to be calculating properly.

Sampling
--------------------
For the sake of example we explicitly set the sampler as "emcee" and an init and run parameter.  Normally EMCEE requires that you specify an initial state for the walkers but Starlord can just draw that from the priors.  Running the model, we get:

.. code:: none

    100%|█████████████████████████████████████| 4500/4500 [00:00<00:00, 5402.18it/s]
         Name               Mean         Std         16%         50%         84%
       0 feh0             0.3544      0.1196      0.2297      0.3856      0.4691
       1 logAge           0.8458      0.2542      0.7762      0.9292      0.9819
       2 logMass0       0.005823     0.01787   -0.007341    0.002378      0.0193
       3 parallax          15.02     0.03559       14.98       15.02       15.05

Wasn't that delightfully fast?  The mass is about what we expected and the metallicity is plausible but wider than what is listed on the NASA exoplanet archive -- presumably a side-effect of choosing the input bands semi-arbitrarily.  We also get an npz file to use for subsequent analysis.  This can be opened in Python with ``post_sample = np.load("hd80606.npz")['samples']``.
