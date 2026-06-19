Multiple Star Systems
=====================
Starlord is capable of modeling multi-star systems, but specifying the model for such systems requires additional work. The key factors are:

:Multiplicity: The number of stars in the system is set with e.g. ``multiplicity2.mist = 2``; in this case a binary system using the Mist2 grid.
:Indexing: Grid variables, parameters, etc are indexed with ``--i``, where ``i`` is the star's index, between 1 and the specified multiplicity. This may be used for e.g. setting separate mass priors or unblended photometry.
:Blending: If a photometric band blends the light from all stars, this can be specified with ``--blend`` instead of an index, which computes the bands for all stars and combines them appropriately.
:Shared Parameters: Grid inputs and parameters may be shared among stars or kept separate. The provided Starlord grids are set up to assume physically associated stars, with metallicity, age, parallax, and extinction shared, but different masses. To relax some or all of these assumptions see `Separating the Parameters`_ and `Physically Unassociated Stars`_.
:Disambiguation: To obtain unambiguous outputs, it is helpful to write the model such that the stars cannot swap parameters without changing the model probability. This may be done in the priors, but the example below (``var.mass_difference``) shows how to specify that one star must be more massive than the other.

Stellar Companions
-----------------------------
Here is a full example input toml file for the star WASP-77. All available photometry is blended and the stars are physically associated (the default setup). This means that the only model parameter that differs between the two stars is the mass. However, output values, such as the luminosity, are still computed separately for each star. In order to avoid the stars swapping places during sampling, we add a likelihood term that forces star 1 to always be higher mass than star 2.

.. literalinclude:: ../examples/wasp77.toml
   :language: toml

Running this with `starlord wasp77.toml`, we obtain:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av                           0.02775    0.001071     0.02681     0.02774     0.02872
       1 feh                          -0.1007     0.09443     -0.1763     -0.1273    -0.03533
       2 log_age                      -0.2151      0.7554       -1.12     -0.2062      0.6564
       3 log_mass0__1                 0.03902      0.1421    -0.03652    -0.02643      0.2117
       4 log_mass0__2                 -0.1543     0.07182     -0.1965     -0.1727    -0.06715
       5 parallax                       9.508     0.01835        9.49       9.508       9.527
    -----------------------------------------------------------------------------------------
       6 log_like                      -37.86       26.37       -53.2      -30.21      -29.01
       7 log_prior                      4.023       3.095       3.295       4.502       5.062
       8 mist2__log_radius__1         -0.4779       0.758      -1.887    -0.07939    -0.06953
       9 mist2Tracks__log_g__1          5.321       1.436       4.551       4.567       7.971
      10 mist2__log_lum__1            -0.9354       1.471      -2.631      -0.195     -0.1729
      11 mist2Tracks__log_teff__1       3.766      0.1057        3.75       3.752       3.754
      12 mist2__log_radius__2         -0.1738     0.08178     -0.2226     -0.2026    -0.03824
      13 mist2Tracks__log_g__2          4.631     0.09883       4.506       4.674       4.695
      14 mist2__log_lum__2            -0.7392      0.3541     -0.9715     -0.8488     -0.1284
      15 mist2Tracks__log_teff__2       3.663     0.04897       3.628       3.651       3.744

Separating the Parameters
-----------------------------
In some cases, it can be helpful to prevent starlord from assuming the stars have the same age and metallicities. This requires overriding the default grid inputs from ignoring index to copying it -- e.g. from ``p.log_age`` to ``p.log_age--i``. You will need to override the latter variables for the ``mistInvAge`` grid (used internally to obtain an equivalent evolutionary point or EEP), as this is the only one that references the age.

.. code:: toml

    override.mist2.feh = "p.feh--i"
    override.mist2InvAge.log_age = "p.log_age--i"

This is admittedly a little esoteric if you don't deeply understand Starlord, but amounts to saying "if this variable is referred to by an indexed variable, propagate that index instead of omitting it". Regardless, these lines can be copied to other binary systems. Extending to triple and greater multiplicity systems is as simple as adding additional lines for each star.

Naturally, splitting the stars to use separate metallicities and ages means that they now require separate priors:

.. code:: toml

    prior.feh--1 = ["uniform", -0.2, 0.5]
    prior.feh--2 = ["uniform", -0.2, 0.5]
    prior.log_age--1 = ["uniform", -1.5, 1.2]
    prior.log_age--2 = ["uniform", -1.5, 1.2]

Unfortunately, it is easy to miss a parameter override while modifying Starlord's variable resolution. It is helpful to run ``starlord -da wasp77_separate.toml`` and verify that there are no unexpected parameters in the model before running. Starlord will also output an error if you've set priors for variables that don't exist, or haven't set priors for variables that do.

Once the overrides and prior specifications are complete, we can again run the model in the same manner to obtain:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av                           0.02803    0.002153     0.02685     0.02777     0.02874
       1 feh__1                      -0.03209      0.1518     -0.1433    -0.07972     0.09614
       2 feh__2                        0.2493      0.1875      0.0378      0.2827      0.4564
       3 log_age__1                   -0.3223      0.7732        -1.2     -0.4052      0.6475
       4 log_age__2                   -0.2475      0.7535      -1.096     -0.2868        0.66
       5 log_mass0__1                 0.04062      0.1304     -0.0303    -0.01864      0.1827
       6 log_mass0__2                 -0.1356     0.07949     -0.1894     -0.1538    -0.02225
       7 parallax                       9.509     0.01813       9.492       9.509       9.528
    -----------------------------------------------------------------------------------------
       8 log_like                      -39.06       21.34      -64.71      -28.93      -27.51
       9 log_prior                       1.17        19.9       2.511       3.859       4.422
      10 mist2__log_radius__1         -0.4734       0.759      -1.889    -0.07313    -0.06314
      11 mist2Tracks__log_g__1          5.316       1.432       4.548       4.562       7.973
      12 mist2__log_lum__1            -0.9372       1.474      -2.893      -0.178     -0.1576
      13 mist2Tracks__log_teff__1       3.764     0.08656       3.751       3.753       3.755
      14 mist2__log_radius__2          -0.168     0.07701     -0.2177     -0.1889    -0.04643
      15 mist2Tracks__log_g__2          4.638     0.08012       4.534       4.664       4.694
      16 mist2__log_lum__2            -0.7878      0.3846      -1.062     -0.9159     -0.1301
      17 mist2Tracks__log_teff__2       3.648     0.05944       3.604       3.628        3.75

Physically Unassociated Stars
-----------------------------
For stars which are apparent binaries but are not physically associated, we will generally wish to use not only separate metallicities and ages, but also parallaxes and extinction values. This is not the case for WASP-77, but for consistency of example we'll pretend it is.

This is done exactly as it was above, except that both of these variables are direct members of the Mist grid rather than the supporting evolution tracks or inverse-age grids.

.. code:: toml

    override.mist2.parallax = "p.parallax--i"
    override.mist2.Av = "p.Av--i"

Now we must also set separate priors for parallax and extinction.

.. code:: toml

    prior.Av--1 = ["trunc_normal", 0.02687, 0.03, 0, 20]
    prior.Av--2 = ["trunc_normal", 0.02687, 0.03, 0, 20]
    prior.parallax--1 = [9.508, 0.018]
    prior.parallax--2 = [9.508, 0.018]

These parameters are allowed to vary separately, but since we're using the same informative priors as before, we don't expect the posterior to change much. Running the model we see that indeed it doesn't:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av__1                        0.02762    0.001253     0.02674     0.02767     0.02864
       1 Av__2                         0.0277    0.001997     0.02684     0.02782     0.02879
       2 feh__1                      -0.02954      0.1485     -0.1387    -0.07833      0.1068
       3 feh__2                        0.2332      0.1705     0.05285      0.2124      0.4412
       4 log_age__1                   -0.2596      0.6405      -1.009     -0.1615      0.3589
       5 log_age__2                    -0.178      0.7287      -1.082     -0.1153      0.6158
       6 log_mass0__1                  0.1062      0.1881    -0.02766    -0.01529      0.3771
       7 log_mass0__2                 -0.1122     0.08198     -0.1827     -0.1489    0.001123
       8 parallax__1                    9.508     0.01732       9.491       9.509       9.524
       9 parallax__2                     9.51     0.01821       9.492        9.51       9.528
    -----------------------------------------------------------------------------------------
      10 log_like                      -40.86       17.28      -64.43      -29.37      -27.63
      11 log_prior                      6.096       18.93       6.784       8.728       9.788
      12 mist2__log_radius__1         -0.6953      0.8709      -1.909     -0.0754    -0.06471
      13 mist2Tracks__log_g__1          5.742       1.652       4.551       4.565       8.041
      14 mist2__log_lum__1             -1.129       1.405      -2.595     -0.1831     -0.1613
      15 mist2Tracks__log_teff__1       3.827      0.1507       3.752       3.754       4.074
      16 mist2__log_radius__2         -0.1448     0.07689     -0.2117     -0.1832    -0.04045
      17 mist2Tracks__log_g__2          4.615     0.07809       4.518       4.653       4.687
      18 mist2__log_lum__2            -0.6801      0.4042      -1.029     -0.8874     -0.1295
      19 mist2Tracks__log_teff__2       3.664     0.06361        3.61       3.634       3.749

For completeness, here is the toml file for this case:

.. literalinclude:: ../examples/wasp77_unassoc.toml
   :language: toml
