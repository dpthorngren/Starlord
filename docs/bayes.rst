Bayesian Interpretation
-----------------------

.. warning::

   This page is not yet finished.

Suppose you are interested in some parameters `x`, `y`, and `z`, but you can't measure them directly. However, you can measure some functions of these variables up to some normally-distributed uncertainty:

.. math::
    A(x,y,z) &= 1.5 \pm 0.2 \\
    B(x,y,z) &= 2.3 \pm 0.15 \\
    C(x,y,z) &= -0.5 \pm 0.05

This kind of problem pops up a lot in astronomy. For example, we often cannot measure a star's mass, metallicity, etc. directly, but we can it's brightness at different wavelengths, which depend on the those unmeasured properties in complicated ways.  Complicating matters, these functions can be expensive to compute, so it can help to run the models across a grid of input values and interpolate them.  Starlord is focused on using these grids (see :doc:`grids`), so let's pretend you've already set up a grid that looks like this:

.. math::
   \mathrm{ExampleGrid} (x, y, z \rightarrow A, B, C)

A Bayesian interpretation of this situation is that we have observed random variables :math:`A_\mathrm{obs} = 1.5`, :math:`B_\mathrm{obs} = 2.3`, :math:`C_\mathrm{obs} = -0.5`, which were generated from a normal distribution centered on their true values.  These true values were in turn generated :abbr:`exactly (You can add uncertainty onto the functions themselves too, but we'll keep things simple here.)` from their functions of `x`, `y`, and `z` .  If our observations are independent, the likelihood of obtaining these observations given the model parameters is:

.. math::
   p(A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs} | x, y, z) = \mathcal{N}(A(x,y,z), \sigma_A) \times \mathcal{N}(B(x,y,z), \sigma_B) \times \mathcal{N}(C(x,y,z), \sigma_C)

This is really just restating things in fancy stats notation.  Notice that there are three multiplicative terms here, one for each observation.

Still, what we *really* want is the probability distributions for `x`, `y`, and `z`, given our observations.  Bayes Theorem gets us there:

.. math::
   p(x, y, z | A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs}) \propto p(x, y, z | A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs}) \times p(x, y, z)

That is, the posterior probability is :abbr:`proportional to (You don't need to worry about the normalizing constant unless you're trying to use Bayes factors for model comparison.  For that you'll want to use a tool like nested sampling rather than trying to calculate it by hand, as it is usually intractable.)` the likelihood times the prior.  Assuming our priors are independent, we can split them up and write the posterior as:

.. math::
   p(x, y, z | A_\mathrm{obs}, B_\mathrm{obs}, C_\mathrm{obs}) \propto\; &\mathcal{N}(A(x,y,z), \sigma_A) \times \mathcal{N}(B(x,y,z), \sigma_B) \times \mathcal{N}(C(x,y,z), \sigma_C) \\ &\times p(x) \times p(y) \times p(z)

So in the end our model was simple enough that we're just multiplying the three likelihood terms and the three prior terms.  Now we just need to sample the distribution.

