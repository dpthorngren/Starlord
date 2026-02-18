from __future__ import annotations

from abc import ABC, abstractmethod
from collections import namedtuple
from types import ModuleType
from typing import Callable, Optional

import dynesty
import emcee
import numpy as np
from dynesty.results import Results as DynestyResults

ResultStats = namedtuple("ResultStats", ["mean", 'cov', 'std', 'p16', 'p50', 'p84'])
_dummyModel = namedtuple(
    "DummyModel",
    ['param_names', 'forward_model', 'log_like', 'prior_transform', 'log_prior', 'output_names', 'postprocess'])


class _Sampler(ABC):
    '''Abstract class for objects which can sample from probability distributions.'''
    ndim: int
    param_names: list[str]
    model: _dummyModel
    post: Optional[np.ndarray]

    @property
    @abstractmethod
    def results(self) -> object:
        pass

    @abstractmethod
    def run(self, **args):
        pass

    @abstractmethod
    def stats(self) -> ResultStats:
        pass

    @abstractmethod
    def save_results(self, filename: str) -> None:
        pass

    def summary(self) -> str:
        # TODO: Convergence statistics
        stats = self.stats()
        out = ["     Name".ljust(29) + "Mean".rjust(12) + "Std".rjust(12)]
        out[0] += "16%".rjust(12) + "50%".rjust(12) + "84%".rjust(12)
        for i in range(self.ndim):
            line = f"{i:4d} {self.param_names[i]:24}"
            line += f" {stats.mean[i]:11.4g} {stats.std[i]:11.4g}"
            line += f" {stats.p16[i]:11.4g} {stats.p50[i]:11.4g} {stats.p84[i]:11.4g}"
            out += [line]
        if self.model.output_names and self.post is not None:
            out += [89 * "-"]
            for i, name in enumerate(self.model.output_names):
                i = i + self.ndim
                line = f"{i:4d} {name:24}"
                line += f" {stats.mean[i]:11.4g} {stats.std[i]:11.4g}"
                line += f" {stats.p16[i]:11.4g} {stats.p50[i]:11.4g} {stats.p84[i]:11.4g}"
                out += [line]
        return "\n".join(out)


class SamplerEnsemble(_Sampler):
    '''Thin wrapper for EMCEE's EnsembleSampler'''
    sampler: emcee.EnsembleSampler
    prior_transform: Callable | None
    burn_in: int
    thin: int

    def __init__(
        self,
        log_prob: Callable,
        ndim: int,
        param_names: list[str] = [],
        prior_transform: Callable | None = None,
        burn_in: int = 500,
        thin: int = 1,
        pool: int = 1,
        model: object = None,
        **args,
    ) -> None:
        self.ndim = ndim
        self.prior_transform = prior_transform
        self.param_names = param_names if param_names else [""] * ndim
        self.burn_in = burn_in
        self.thin = thin
        self.model = model  # type: ignore
        assert len(param_names) == ndim
        args.setdefault('nwalkers', max(100, 5 * ndim))
        args.setdefault('ndim', ndim)
        args.setdefault('log_prob_fn', log_prob)
        self.sampler = emcee.EnsembleSampler(**args)

    @classmethod
    def create_from_module(cls, module: ModuleType, constants: list, **args):
        model = module.Model(*constants)
        return SamplerEnsemble(
            model.log_prob,
            len(model.param_names),
            model.param_names,
            model.prior_transform,
            model=model,
            **args,
        )

    @property
    def results(self) -> object:
        return self.sampler.get_chain(flat=True, discard=self.burn_in, thin=self.thin)

    def run(self, **args):
        args.setdefault('nsteps', 2500)
        args.setdefault('progress', True)
        if "initial_state" not in args:
            assert self.prior_transform is not None, "Must provide initial_state or prior_transform."
            args['initial_state'] = 0.3 + 0.4 * np.random.rand(self.sampler.nwalkers, self.ndim)
            [self.prior_transform(s) for s in args['initial_state']]
        args['nsteps'] += self.burn_in
        self.sampler.run_mcmc(**args)
        post = self.results
        assert type(post) is np.ndarray
        postprocessed = np.zeros((post.shape[0], len(self.model.output_names)))
        self.model.postprocess(post, postprocessed)
        self.post = np.hstack([post, postprocessed])

    def stats(self) -> ResultStats:
        assert type(self.post) is np.ndarray, "Must run sampler before computing stats!"
        mean = self.post.mean(axis=0)
        std = self.post.std(axis=0)
        cov = np.cov(self.post.T)
        std = np.sqrt(np.diag(cov))
        q = np.quantile(self.post, [.16, .5, .84], axis=0)
        return ResultStats(mean, cov, std, q[0], q[1], q[2])

    def save_results(self, filename: str):
        np.savez_compressed(filename, samples=self.post)  # type: ignore


class SamplerNested(_Sampler):
    '''Thin wrapper for the Dynesty NestedSampler'''
    sampler: dynesty.NestedSampler

    def __init__(
        self,
        log_like: Callable,
        prior_transform: Callable,
        ndim: int,
        param_names: list[str] = [],
        model: object = None,
        **args,
    ) -> None:
        self.ndim = ndim
        self.param_names = param_names if param_names else [""] * ndim
        assert len(param_names) == ndim
        self.model = model  # type: ignore
        args.setdefault('ndim', ndim)
        args.setdefault('loglikelihood', log_like)
        args.setdefault('prior_transform', prior_transform)
        self.sampler = dynesty.NestedSampler(**args)

    @classmethod
    def create_from_module(cls, module: ModuleType, constants: list, **args):
        model = module.Model(*constants)
        return SamplerNested(
            model.log_like,
            model.prior_transform,
            len(model.param_names),
            model.param_names,
            model=model,
            **args,
        )

    @property
    def results(self) -> DynestyResults:
        return self.sampler.results

    def run(self, **args):
        self.sampler.run_nested(**args)

    def stats(self) -> ResultStats:
        samples = self.results['samples']
        weights = self.sampler.results.importance_weights()
        mean, cov = dynesty.utils.mean_and_cov(samples, weights)
        std = np.sqrt(np.diag(cov))
        q = np.array([
            dynesty.utils.quantile(samples[:, i], [0.16, 0.5, 0.84], weights=weights) for i in range(samples.shape[1])
        ])
        return ResultStats(mean, cov, std, q[:, 0], q[:, 1], q[:, 2])

    def save_results(self, filename: str):
        # TODO: Citation info.
        np.savez_compressed(
            filename,
            samples=self.results['samples'],
            weights=self.sampler.results.importance_weights(),
        )
