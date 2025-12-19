from abc import ABC, abstractmethod
from collections import namedtuple
from types import ModuleType
from typing import Callable

import dynesty
import emcee
import numpy as np
from dynesty.results import Results as DynestyResults

ResultStats = namedtuple("ResultStats", ["mean", 'cov', 'std', 'p16', 'p50', 'p84'])


class _Sampler(ABC):
    '''Abstract class for objects which can sample from probability distributions.'''
    ndim: int
    param_names: list[str]
    logl_args: list[object]

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
    def save(self, filename: str) -> None:
        pass

    def summary(self) -> str:
        # TODO: Convergence statistics
        stats = self.stats()
        out = ["     Name".ljust(16) + "Mean".rjust(12) + "Std".rjust(12)]
        out[0] += "16%".rjust(12) + "50%".rjust(12) + "84%".rjust(12)
        for i in range(self.ndim):
            line = f"{i:4d} {self.param_names[i]:11}"
            line += f" {stats.mean[i]:11.4g} {stats.std[i]:11.4g}"
            line += f" {stats.p16[i]:11.4g} {stats.p50[i]:11.4g} {stats.p84[i]:11.4g}"
            out += [line]
        return "\n".join(out)


class SamplerEnsemble(_Sampler):
    '''Thin wrapper for EMCEE's EnsembleSampler'''
    sampler: emcee.EnsembleSampler
    prior_transform: Callable | None

    def __init__(
        self,
        log_prob: Callable,
        ndim: int,
        logl_args: list[object] = [],
        param_names: list[str] = [],
        prior_transform: Callable | None = None,
        **args,
    ) -> None:
        self.ndim = ndim
        self.prior_transform = prior_transform
        self.logl_args = logl_args
        self.param_names = param_names if param_names else [""] * ndim
        assert len(param_names) == ndim
        args.setdefault('nwalkers', max(100, 5 * ndim))
        args.setdefault('ndim', ndim)
        args.setdefault('log_prob_fn', log_prob)
        args.setdefault('args', logl_args)
        self.sampler = emcee.EnsembleSampler(**args)

    @classmethod
    def create_from_module(cls, module: ModuleType, constants: list, **args):
        return SamplerEnsemble(
            module.log_prob,
            len(module.param_names),
            constants,
            module.param_names,
            module.prior_transform,
            **args,
        )

    @property
    def results(self) -> object:
        return self.sampler.get_chain(flat=True)

    def run(self, **args):
        args.setdefault('nsteps', 2500)
        args.setdefault('progress', True)
        if "initial_state" not in args:
            assert self.prior_transform is not None, "Must provide initial_state or prior_transform."
            args['initial_state'] = 0.3 + 0.4 * np.random.rand(self.sampler.nwalkers, self.ndim)
            [self.prior_transform(s) for s in args['initial_state']]
        self.sampler.run_mcmc(**args)

    def stats(self) -> ResultStats:
        results = self.results
        assert type(results) is np.ndarray, "Must run sampler before computing stats!"
        mean = results.mean(axis=0)
        std = results.std(axis=0)
        cov = np.cov(results.T)
        std = np.sqrt(np.diag(cov))
        q = np.quantile(results, [.16, .5, .84], axis=0)
        return ResultStats(mean, cov, std, q[0], q[1], q[2])

    def save(self, filename: str):
        print("TODO: Save run data.")


class SamplerNested(_Sampler):
    '''Thin wrapper for the Dynesty NestedSampler'''
    sampler: dynesty.NestedSampler

    def __init__(
        self,
        log_like: Callable,
        prior_transform: Callable,
        ndim: int,
        logl_args: list[object] = [],
        param_names: list[str] = [],
        **args,
    ) -> None:
        self.ndim = ndim
        self.logl_args = logl_args
        self.param_names = param_names if param_names else [""] * ndim
        assert len(param_names) == ndim
        args.setdefault('ndim', ndim)
        args.setdefault('logl_args', logl_args)
        args.setdefault('loglikelihood', log_like)
        args.setdefault('prior_transform', prior_transform)
        self.sampler = dynesty.NestedSampler(**args)

    @classmethod
    def create_from_module(cls, module: ModuleType, constants: list, **args):
        return SamplerNested(
            module.log_like,
            module.prior_transform,
            len(module.param_names),
            constants,
            module.param_names,
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

    def save(self, filename: str):
        # NOTE: Remember to include citation info.
        print("TODO: Save run data.")
