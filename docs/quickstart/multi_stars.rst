Multiple Star Systems
=====================
Starlord is capable of modeling multi-star systems, but specifying the model for such systems requires additional work. The key factors are:

:Multiplicity: The number of stars in the system is set with e.g. ``multiplicity.mist = 2``; in this case a binary system using the Mist grid.
:Indexing: Grid variables, parameters, etc are indexed with ``--i``, where ``i`` is the star's index, between 1 and the specified multiplicity. This may be used for e.g. setting separate mass priors or unblended photometry.
:Blending: If a photometric band blends the light from all stars, this can be specified with ``--blend`` instead of an index, which computes the bands for all stars and combines them appropriately.
:Shared Parameters: Grid inputs and parameters may be shared among stars or kept separate. The provided Starlord grids are set up to assume physically associated stars, with metallicity, age, parallax, and extinction shared, but different masses. To relax some or all of these assumptions see `Separating the Parameters`_ and `Physically Unassociated Stars`_.
:Disambiguation: To obtain unambiguous outputs, it is helpful to write the model such that the stars cannot swap parameters without changing the model probability. This may be done in the priors, but the example below (``var.mass_Difference``) shows how to specify that one star must be more massive than the other.

Stellar Companions
-----------------------------
Here is a full example input toml file for the star WASP-77. All available photometry is blended and the stars are physically associated (the default setup). This means that the only model parameter that differs between the two stars is the mass. However, output values, such as the luminosity, are still computed separately for each star. In order to avoid the stars swapping places during sampling, we add a likelihood term that forces star 1 to always be higher mass than star 2.

.. literalinclude:: ../examples/wasp77.toml
   :language: toml

Running this with `starlord wasp77.toml`, we obtain:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av                           0.02797    0.002072     0.02681     0.02774     0.02867
       1 feh0                        -0.09599      0.0764     -0.1736     -0.1041     -0.0304
       2 logAge                       -0.2873      0.5584     -0.9424     -0.2681      0.3373
       3 logMass0__1                 -0.03729     0.02678    -0.04788     -0.0338    -0.01953
       4 logMass0__2                  -0.1535     0.06353     -0.1867     -0.1609     -0.1402
       5 parallax                       9.506     0.01798       9.488       9.506       9.524
    -----------------------------------------------------------------------------------------
       6 mistTracks__logRadius__1    -0.07982     0.02426    -0.09093    -0.07622    -0.06298
       7 mistTracks__logG__1             4.56     0.02583        4.54       4.558       4.577
       8 mistTracks__logL__1          -0.2224       0.134     -0.2323      -0.198     -0.1695
       9 mistTracks__logTeff__1         3.746     0.02226       3.748        3.75       3.751
      10 mistTracks__logRadius__2     -0.1808     0.07839     -0.2102     -0.1917     -0.1786
      11 mistTracks__logG__2            4.646     0.09619        4.65       4.663       4.678
      12 mistTracks__logL__2          -0.7686      0.2992     -0.9428     -0.8009     -0.6878
      13 mistTracks__logTeff__2          3.66     0.03794       3.631       3.657       3.679

Separating the Parameters
-----------------------------
In some cases, it can be helpful to prevent starlord from assuming the stars have the same age and metallicities. This requires overriding the default grid inputs from ignoring index to copying it -- e.g. from ``p.logAge`` to ``p.logAge--i``. You will need to override these variables for both the ``mistTracks`` (evolution tracks used to obtain e.g. bolometric luminosities) and ``mistInvAge`` (used internally to obtain an equivalent evolutionary point or EEP) grids.

.. code:: toml

    override.mist.feh = "p.feh0--i"
    override.mistTracks.feh0 = "p.feh0--i"
    override.mistTracks.logAge = "mistInvAge.logAge--i"
    override.mistInvAge.feh0 = "p.feh0--i"
    override.mistInvAge.logAge = "p.logAge--i"

This is admittedly a little esoteric if you don't deeply understand Starlord, but amounts to saying "if this variable is referred to by an indexed variable, propagate that index instead of omitting it". Regardless, these lines can be copied to other binary systems. Extending to triple and greater multiplicity systems is as simple as adding additional lines for each star.

Naturally, splitting the stars to use separate metallicities and ages means that they now require separate priors:

.. code:: toml

    prior.feh0--1 = ["uniform", -0.2, 0.5]
    prior.feh0--2 = ["uniform", -0.2, 0.5]
    prior.logAge--1 = ["uniform", -1.5, 1.2]
    prior.logAge--2 = ["uniform", -1.5, 1.2]

Unfortunately, it is easy to miss a parameter override while modifying Starlord's variable resolution. It is helpful to run ``starlord -da wasp77.toml`` and verify that there are no unexpected parameters in the model before running; Starlord will also output an error if you've set priors for variables that don't exist, or haven't set priors for variables that do.

Once the overrides and prior specifications are complete, we can again run the model in the same manner to obtain:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av                           0.02817    0.003346     0.02653     0.02764     0.02871
       1 feh0__1                     -0.02224     0.08135    -0.09855    -0.02123     0.03637
       2 feh0__2                       0.2277      0.1969   -0.002923      0.2677      0.4411
       3 logAge__1                    -0.4263      0.5289      -1.036     -0.4433      0.1759
       4 logAge__2                    -0.2511       0.696      -1.038     -0.3106      0.5727
       5 logMass0__1                 -0.03152      0.0385    -0.03986    -0.01988   -0.009047
       6 logMass0__2                  -0.1354     0.09882     -0.1897     -0.1605     -0.1268
       7 parallax                       9.509      0.0171       9.493        9.51       9.526
    -----------------------------------------------------------------------------------------
       8 mistTracks__logRadius__1    -0.07559     0.03614    -0.08602    -0.06366    -0.05414
       9 mistTracks__logG__1            4.557     0.03576       4.533       4.549       4.572
      10 mistTracks__logL__1          -0.2351      0.1994     -0.2181     -0.1678     -0.1497
      11 mistTracks__logTeff__1          3.74     0.03236       3.749       3.751       3.752
      12 mistTracks__logRadius__2     -0.1571      0.1087      -0.211     -0.1875     -0.1625
      13 mistTracks__logG__2            4.617       0.123       4.608       4.656        4.68
      14 mistTracks__logL__2          -0.7945      0.4794      -1.072      -0.936     -0.7232
      15 mistTracks__logTeff__2         3.641     0.06761       3.597       3.621       3.667

Physically Unassociated Stars
-----------------------------
For stars which are apparent binaries but are not physically associated, we will generally wish to use not only separate metallicities and ages, but also parallaxes and extinction values. This is not the case for WASP-77, but for consistency of example we'll pretend it is.

This is done exactly as it was above, except that both of these variables are direct members of the Mist grid rather than the supporting evolution tracks or inverse-age grids.

.. code:: toml

    override.mist.parallax = "p.parallax--i"
    override.mist.Av = "p.Av--i"

Now we must also set separate priors for parallax and extinction.

.. code:: toml

    prior.Av--1 = ["trunc_normal", 0.02687, 0.03, 0, 20]
    prior.Av--2 = ["trunc_normal", 0.02687, 0.03, 0, 20]
    prior.parallax--1 = [9.508, 0.018]
    prior.parallax--2 = [9.508, 0.018]

These parameters are allowed to vary separately, but since we're using the same informative priors as before, we don't expect the posterior to change much. Running the model we see that indeed it doesn't:

.. code:: none

         Name                            Mean         Std         16%         50%         84%
       0 Av__1                        0.02786      0.0018     0.02681     0.02772     0.02864
       1 Av__2                        0.02799     0.00217     0.02683     0.02774     0.02869
       2 feh0__1                     -0.02614     0.06685    -0.08296    -0.02221     0.02234
       3 feh0__2                       0.2526      0.1954     0.01271      0.3027      0.4513
       4 logAge__1                    -0.5668      0.4855      -1.128     -0.5781  -0.0001668
       5 logAge__2                    -0.1798      0.7035     -0.9855     -0.2278      0.6546
       6 logMass0__1                 -0.02113     0.02546    -0.02824    -0.01593   -0.007553
       7 logMass0__2                  -0.1553     0.06178     -0.1888     -0.1629     -0.1371
       8 parallax__1                    9.507     0.01773       9.489       9.507       9.525
       9 parallax__2                    9.507     0.01773       9.489       9.507       9.524
    -----------------------------------------------------------------------------------------
      10 mistTracks__logRadius__1    -0.06794     0.02329    -0.07554    -0.06253    -0.05513
      11 mistTracks__logG__1            4.553     0.02244       4.538       4.549       4.563
      12 mistTracks__logL__1          -0.1915      0.1279     -0.1935     -0.1648     -0.1501
      13 mistTracks__logTeff__1         3.747     0.02101        3.75       3.751       3.752
      14 mistTracks__logRadius__2     -0.1813     0.06311     -0.2112       -0.19     -0.1698
      15 mistTracks__logG__2            4.645     0.06936       4.631       4.658        4.68
      16 mistTracks__logL__2          -0.9035      0.2958      -1.068     -0.9535     -0.8111
      17 mistTracks__logTeff__2         3.626     0.04479       3.598       3.618       3.648

For completeness, here is the toml file for this case:

.. literalinclude:: ../examples/wasp77_unassoc.toml
   :language: toml
