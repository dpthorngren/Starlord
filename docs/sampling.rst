Sampling
====================
Starlord uses thin wrapper classes around samplers provided by other libraries.  This is to allow Starlord to automatically initialize and run the codes while still flexibly passing settings to the samplers themselves.

Settings can be passed to the samplers in the ``[sampling]`` section of the TOML files.  ``sampler_name.init`` entries are passed to the sampler initializer, and ``sampler_name.run`` entries are passed to the run command. If you are calling Starlord from within Python, you have the additional option to retrieve and use the underlying sampler ``sampler``, if you wish.

Emcee Sampler
--------------------
The `Emcee sampler <https://emcee.readthedocs.io/en/stable/>`_ is a popular MCMC sampler in astronomy,

Init:

1. nwalkers
2. ndim

Run:
1. initial_state
2. nsteps

Dynesty Sampler
--------------------
The `Dynesty sampler <https://dynesty.readthedocs.io/en/v3.0.0/>`_ 

Initialize:

1. nlive
2. bound
3. sample
4. update_interval

Run:
1. nlive_init
2. maxiter_init
3. dlogz_init
4. n_effective
5. print_progress
