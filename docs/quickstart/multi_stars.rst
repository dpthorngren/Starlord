Multiple Star Systems
=====================
Starlord is capable of modeling multi-star systems, but specifying the model for such systems requires additional work. The key factors are:

:Multiplicity: The number of stars in the system is set with e.g. ``multiplicity2.mist2 = 2``; in this case a binary system using the Mist2 grid.
:Indexing: Grid variables, parameters, etc are indexed with ``--i``, where ``i`` is the star's index, between 1 and the specified multiplicity. This may be used for e.g. setting separate mass priors or unblended photometry.
:Blending: If a photometric band blends the light from all stars, this can be specified with ``--blend`` instead of an index, which computes the bands for all stars and combines them appropriately.
:Shared Parameters: Grid inputs and parameters may be shared among stars or kept separate. The provided Starlord grids are set up to assume physically associated stars, with metallicity, age, distance, and extinction shared, but different masses. To relax some or all of these assumptions see `Separating the Parameters`_ and `Physically Unassociated Stars`_.
:Disambiguation: To obtain unambiguous outputs, it is helpful to write the model such that the stars cannot swap parameters without changing the model probability. This may be done in the priors, but the example below (``var.mass_difference``) shows how to specify that one star must be more massive than the other.

Stellar Companions
-----------------------------
Here is a full example input toml file for the star WASP-77. All available photometry is blended and the stars are physically associated (the default setup). This means that the only model parameter that differs between the two stars is the mass. However, output values, such as the luminosity, are still computed separately for each star. In order to avoid the stars swapping places during sampling, we add a likelihood term that forces star 1 to always be higher mass than star 2.

.. literalinclude:: ../examples/wasp77.toml
   :language: toml

Running this with `starlord wasp77.toml`, we obtain:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av                           0.02769   0.0009729     0.02682      0.0277     0.02864
       1 distance                       105.2      0.2065         105       105.2       105.3
       2 feh                          -0.1674       0.106     -0.2821     -0.1612    -0.06954
       3 log_age                       -0.339      0.6067       -1.07     -0.3099      0.3673
       4 log_mass0__1               -0.006204      0.1213    -0.05565    -0.03424     -0.0178
       5 log_mass0__2                 -0.1666     0.07102     -0.1997     -0.1656     -0.1368
    -----------------------------------------------------------------------------------------
       6 log_like                      -27.83       44.19      -26.49       -23.9      -22.66
       7 log_prior                     -8.888        1.39      -9.827      -8.584      -7.818
       8 mist2__log_radius__1         -0.2136      0.4718     -0.1074    -0.08344    -0.06868
       9 mist2Tracks__log_g__1          4.816      0.9051        4.55       4.569       4.595
      10 mist2__log_lum__1            -0.4459      0.8703      -0.266     -0.2038     -0.1718
      11 mist2Tracks__log_teff__1       3.757     0.02602        3.75       3.752       3.754
      12 mist2__log_radius__2         -0.1951     0.05425     -0.2248     -0.1975      -0.177
      13 mist2Tracks__log_g__2          4.661     0.05335        4.64       4.671       4.694
      14 mist2__log_lum__2            -0.7921      0.2634     -0.9824     -0.8107      -0.626
      15 mist2Tracks__log_teff__2       3.661     0.04081       3.628       3.659       3.694

Separating the Parameters
-----------------------------
In some cases, it can be helpful to prevent starlord from assuming the stars have the same age and metallicities. This requires overriding the default grid inputs from ignoring index to copying it -- e.g. from ``p.log_age`` to ``p.log_age--i``. You will need to override the latter variables for the ``mist2InvAge`` grid (used internally to obtain an equivalent evolutionary point or EEP), as this is the only one that references the age.

.. code:: toml

    override.mist2.feh = "p.feh--i"
    override.mist2InvAge.log_age = "p.log_age--i"

This is admittedly a little esoteric if you don't deeply understand Starlord, but amounts to saying "if this variable is referred to by an indexed variable, propagate that index instead of omitting it". Regardless, these lines can be copied to other binary systems.

Naturally, splitting the stars to use separate metallicities and ages means that they now require separate priors:

.. code:: toml

    prior.feh--1 = ["casagrande_disk"]
    prior.feh--2 = ["casagrande_disk"]
    prior.log_age--1 = ["uniform", -1.5, 1.2]
    prior.log_age--2 = ["uniform", -1.5, 1.2]

Unfortunately, it is easy to miss a parameter override while modifying Starlord's variable resolution. It is helpful to run ``starlord -da wasp77_separate.toml`` and verify that there are no unexpected parameters in the model before running. Starlord will also output an error if you've set priors for variables that don't exist, or haven't set priors for variables that do.

Once the overrides and prior specifications are complete, we can again run the model in the same manner to obtain:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av                           0.02823    0.003948     0.02669     0.02766     0.02867
       1 distance                       105.2      0.2605         105       105.2       105.3
       2 feh__1                       -0.1176     0.07016     -0.1821     -0.1182    -0.05839
       3 feh__2                       0.07426        0.18      -0.111     0.08235      0.2562
       4 log_age__1                   -0.4398      0.5808      -1.077     -0.4626      0.1214
       5 log_age__2                   -0.2697      0.7668      -1.173     -0.2743      0.6165
       6 log_mass0__1                -0.02261     0.02579    -0.03595     -0.0252    -0.01527
       7 log_mass0__2                 -0.1759     0.06033     -0.2036     -0.1702     -0.1432
    -----------------------------------------------------------------------------------------
       8 log_like                      -29.51       24.95      -25.85      -23.56      -22.36
       9 log_prior                     -18.49       56.19      -10.68      -8.939      -8.155
      10 mist2__log_radius__1         -0.1277      0.3095    -0.08583    -0.07395    -0.06411
      11 mist2Tracks__log_g__1           4.66      0.5804       4.548       4.561       4.574
      12 mist2__log_lum__1            -0.2869      0.6117      -0.207     -0.1806     -0.1604
      13 mist2Tracks__log_teff__1       3.753     0.03015       3.752       3.753       3.754
      14 mist2__log_radius__2         -0.2007     0.05231     -0.2225     -0.1978      -0.177
      15 mist2Tracks__log_g__2          4.663     0.05811       4.629       4.666       4.693
      16 mist2__log_lum__2            -0.9187      0.2238       -1.05     -0.9122     -0.7884
      17 mist2Tracks__log_teff__2       3.632     0.03435       3.607       3.633       3.656

Physically Unassociated Stars
-----------------------------
For stars which are apparent binaries but are not physically associated, we will generally wish to use not only separate metallicities and ages, but also distance and extinction values. This is not the case for WASP-77, but for consistency of example we'll pretend it is.

This is done exactly as it was above, except that both of these variables are direct members of the Mist grid rather than the supporting evolution tracks or inverse-age grids.

.. code:: toml

    override.mist2.distance = "p.distance--i"
    override.mist2.Av = "p.Av--i"

Now we must also set separate priors for distance and extinction.

.. code:: toml

    prior.Av--1 = ["trunc_normal", 0.02687, 0.03, 0, 20]
    prior.Av--2 = ["trunc_normal", 0.02687, 0.03, 0, 20]
    prior.distance--1 = ["trunc_power", -2, 10, 200]
    prior.distance--2 = ["trunc_power", -2, 10, 200]

These parameters are allowed to vary separately, but since we're using the same informative priors as before, we don't expect the posterior to change much. Running the model we see that indeed it doesn't:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av__1                        0.02772     0.00105     0.02674      0.0277     0.02863
       1 Av__2                        0.02771    0.001295     0.02675     0.02763      0.0285
       2 distance__1                    104.9       2.372         105       105.2       105.4
       3 distance__2                    104.7       4.416         105       105.2       105.4
       4 feh__1                       -0.1147     0.07223     -0.1818     -0.1033     -0.0473
       5 feh__2                       0.09336      0.1743    -0.07401     0.09694      0.2723
       6 log_age__1                   -0.3268      0.5238     -0.9486     -0.2564      0.2266
       7 log_age__2                   -0.4218      0.7932      -1.308     -0.5373      0.5665
       8 log_mass0__1                -0.02342     0.03184    -0.03752    -0.02427    -0.01473
        9 log_mass0__2                 -0.1905     0.07248      -0.217     -0.1778     -0.1486
    -----------------------------------------------------------------------------------------
      10 log_like                       -1063   1.034e+04       -26.3      -24.12      -22.71
      11 log_prior                     -14.54       6.717      -15.29      -13.57      -12.54
      12 mist2__log_radius__1        -0.08529      0.1317    -0.08313    -0.07043    -0.06117
      13 mist2Tracks__log_g__1           4.58      0.2423        4.54       4.556       4.571
      14 mist2__log_lum__1            -0.1496      0.2922     -0.2005     -0.1739     -0.1551
      15 mist2Tracks__log_teff__1       3.767      0.1383       3.751       3.753       3.754
      16 mist2__log_radius__2         -0.2097     0.05685     -0.2268     -0.2016     -0.1794
      17 mist2Tracks__log_g__2          4.667     0.06034       4.626       4.668       4.698
      18 mist2__log_lum__2            -0.9745      0.2162      -1.096     -0.9534     -0.8145
      19 mist2Tracks__log_teff__2       3.623     0.03295       3.593       3.626       3.651

For completeness, here is the toml file for this case:

.. literalinclude:: ../examples/wasp77_unassoc.toml
   :language: toml
