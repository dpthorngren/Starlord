Sampling
====================
Starlord uses thin wrapper classes around samplers provided by other libraries.  This is to allow Starlord to automatically initialize and run the codes while still flexibly passing settings to the samplers themselves.

Settings can be passed to the samplers in the ``[sampling]`` section of the TOML files.  ``[sampler_name_init].[option]`` entries are passed to the sampler initializer, and ``[sampler_name]_run.[option]`` entries are passed to the run command. If you are calling Starlord from within Python, you have the additional option to retrieve and use the underlying sampler ``sampler``, if you wish.

Emcee Sampler
--------------------
The `Emcee sampler <https://emcee.readthedocs.io/en/stable/>`_ is a popular MCMC sampler in astronomy,

**Initialization Parameters**

:nwalkers:          The number of walkers, which must be at least twice the
                    number of dimensions; more complex posteriors can benefit
                    from higher walker counts.

**Run Parameters**

:initial_state:     The initial walker states -- if not set Starlord will draw
                    this from the prior PPF. 
:nsteps:            The number of steps per walker to take during the run.

Dynesty Sampler
--------------------
The `Dynesty sampler <https://dynesty.readthedocs.io/en/v3.0.0/>`_ 

**Initialization Parameters**

:nlive:             Asdf
:bound:             Sdfg
:sample:            Gfds
:update_interval:   Wdfg

**Run Parameters**

:nlive_init:        sdfg
:maxiter_init:      gfds
:dlogz_init:        hrt
:n_effective:       rdhtgfb
:print_progress:    gfds
