from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Callable

import dynesty
import emcee
import numpy as np

ResultStats = namedtuple("ResultStats", ["mean", 'cov', 'std', 'p16', 'p50', 'p84'])


class _Sampler(ABC):
    '''Abstract class for objects which can sample from probability distributions.'''
    # TODO: Init from CodeGenerator directly.
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

    def __init__(
        self,
        log_prob: Callable,
        ndim: int,
        logl_args: list[object],
        param_names: list[str] = [],
        **args,
    ) -> None:

        self.ndim = ndim
        self.logl_args = logl_args
        self.param_names = param_names if param_names else [""] * ndim
        assert len(param_names) == ndim
        nwalkers = args.pop('nwalkers', max(100, 5 * ndim))
        self.sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob, args=logl_args, parameter_names=param_names, **args)

    @property
    def results(self) -> object:
        return self.sampler.flatchain

    def run(self, **args):
        # TODO: More default arguments
        self.sampler.run_mcmc(**args)

    def stats(self) -> ResultStats:
        results = self.sampler.flatchain
        assert type(results) is np.ndarray, "Must run sampler before computing stats!"
        mean = results.mean(axis=0)
        std = results.std(axis=0)
        cov = np.cov(results)
        std = np.sqrt(np.diag(cov))
        q = np.quantile(results, [.16, .5, .84], axis=0)
        return ResultStats(mean, cov, std, q[:, 0], q[:, 1], q[:, 2])

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
        args.setdefault('logl_args', logl_args)
        self.sampler = dynesty.NestedSampler(log_like, prior_transform, ndim, **args)

    @property
    def results(self) -> object:
        return self.sampler.results

    def run(self, **args):
        self.sampler.run_nested(**args)

    def stats(self) -> ResultStats:
        samples = self.sampler.results['samples']
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
