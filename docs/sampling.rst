Sampling
====================
Starlord uses thin wrapper classes around samplers provided by other libraries.  This is to allow Starlord to automatically initialize and run the codes while still flexibly passing settings to the samplers themselves. Settings can be passed to the samplers in the ``[sampling]`` section of the TOML files.  ``[sampler_name_init].[option]`` entries are passed to the sampler initializer, and ``[sampler_name]_run.[option]`` entries are passed to the run command. If you are calling Starlord from within Python, you have the additional option to retrieve and use the underlying sampler ``sampler``, if you wish.

The ``[sampling]`` is also where you may set the values of any constants you used during model specification.  The syntax is just ``const.constant_name = value``, where value is a float.  The :doc:`quickstart/stars` page has an example of this.  However, note that constants may *also* be set at the command line via the repeatable argument ``-s constant_name=value``; if a constant is specified in both places, the CLI argument gets priority.

This page lists some common sampling parameters you may wish to set; however, it is not an exhaustive list of the sampler options.

Builtin Sampler
--------------------
The builtin sampler is a simple variant of the affine invariant ensemble sampler, written in Cython and built directly into Starlord to reduce the calling overhead.  It features an adaptive system that runs the sampler until a convergence criterion (similar to the Gelman-Rubin statistic) is satisfied. It is a good default choice as it requires little tuning to work well, but will struggle with more complex distributions.

**Initialization Parameters**

:nwalkers:          The number of walkers, which must be at least twice the number of dimensions; more complex posteriors can benefit from higher walker counts.

**Run Parameters**

:n_samples:             The number of samples to record for each walker during the run.
:burnin:                The number of samples to take before beginning to record the output samples.
:thin:                  The number of samples to take for each sample that is actually recorded.
:alpha:                 The scaling distance for stretch moves -- the default of 2 is usually best.
:initial_state:         The initial walker states -- if not set Starlord will draw this from the prior PPF. 
:progress:              Whether to show a very minimal indicator of sampling progress while running.
:metropolis_frac:       The fraction of samples to be made as a metropolis step rather than a stretch move, 0.2 by default.
:metropolis_presamples: The number of samples used in calculating a proposal covariance for the metropolis steps, n_samples//10 by default.
:adaptive_pgr_thresh:   The threshold convergence statistic at which to end convergence, 1.1 by default, and numbers closer to zero are more aggressive requirements.  If set to less than 1, no adaptation is done.
:max_adapt_iter:        How many failures to meet the convergence requirement before the sampler gives up, 6 by default. Each failure *doubles* the thinning, so the final sample run before giving up will take `(n_samples + burnin) * thin * 2**max_adapt_iter` samples.


Emcee Sampler
--------------------
The `Emcee sampler <https://emcee.readthedocs.io/en/stable/>`_ is a popular MCMC sampler in astronomy which uses a collection of walkers to propose jumps that are invariant to affine transformations of the distribution being sampled.  It's a good choice so long as your posterior isn't too complex (donuts, bananas, holes, etc).

**Initialization Parameters**

:nwalkers:          The number of walkers, which must be at least twice the number of dimensions; more complex posteriors can benefit from higher walker counts.
:burnin:            The number of samples to take before beginning to record the output samples.
:thin:              The number of samples to take for each sample that is actually recorded.

**Run Parameters**

:initial_state:     The initial walker states -- if not set Starlord will draw this from the prior PPF. 
:nsteps:            The number of steps per walker to take during the run.
:alpha:             The scaling distance for stretch moves -- the default of 2 is usually best.
:progress:          Whether to print the progress of the sampler as it runs.

Dynesty Sampler
--------------------
The `Dynesty sampler <https://dynesty.readthedocs.io/en/v3.0.0/>`_ is an alternative sampler that can handle more complex posteriors.  Technically it is not an MCMC algorithm.  This version is the static sampler; a Starlord wrapper the dynamic sampler is not yet implemented.

**Initialization Parameters**

:nlive:             The number of live points, larger values give larger posterior samples and better convergence, but take longer.
:bound:             The method used to approximately bound the prior based on the live points: 'none', 'single', 'multi', 'balls', or 'cubes'.  See the Dynesty documentation for more information.
:sample:            The method used to sample within the likelihood constraint: 'auto', 'unif', 'rwalk', 'slice', or 'rslice'.  See the Dynesty documentation for more information.
:update_interval:   How often to update the posterior distribution estimate, which can be useful for very fast posterior evaluations.

**Run Parameters**

:maxiter_init:      The maximum number of iterations to use during the run (default is no limit).
:dlogz_init:        Determines the estimated uncertainty at which the sampler will complete.
:n_effective:       The number of effective posterior samples at which the sampler will complete.
:print_progress:    Whether to print the progress of the sampler as it runs.
