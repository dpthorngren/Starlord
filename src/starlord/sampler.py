from abc import ABC, abstractmethod
from typing import Callable

import dynesty
import numpy as np


class _Sampler(ABC):
    '''Abstract class for objects which can sample from probability distributions.'''
    # TODO: Init from CodeGenerator directly.

    @property
    @abstractmethod
    def sampler(self) -> object:
        pass

    @property
    @abstractmethod
    def results(self) -> object:
        pass

    @abstractmethod
    def run(self, options: dict):
        pass

    @abstractmethod
    def stats(self) -> np.ndarray:
        pass

    @abstractmethod
    def summary(self) -> str:
        pass

    @abstractmethod
    def save(self, options: dict):
        pass


class SamplerNested(_Sampler):
    '''Thin wrapper for the Dynesty NestedSampler'''

    def __init__(self, loglike: Callable, ptform: Callable, ndim: int, config: dict, logl_args=[]) -> None:
        # TODO: Parameter names
        config.setdefault('logl_args', logl_args)
        self._sampler = dynesty.NestedSampler(loglike, ptform, ndim, **config)

    @property
    def sampler(self):
        return self._sampler

    @property
    def results(self):
        return self._sampler.results

    def run(self, options: dict):
        self._sampler.run_nested(**options)

    def stats(self) -> np.ndarray:
        samples = self.results['samples']
        weights = self.results.importance_weights()
        mean, cov = dynesty.utils.mean_and_cov(samples, weights)
        q = [dynesty.utils.quantile(samples[:, i], [0.16, 0.5, 0.84], weights=weights) for i in range(len(mean))]
        return np.column_stack([mean, np.sqrt(np.diag(cov)), q, cov])

    def summary(self) -> str:
        # TODO: Convergence statistics
        stats = self.stats()
        out = [" Dim" + "Mean".rjust(12) + "Std".rjust(12) + "16%".rjust(12) + "50%".rjust(12) + "84%".rjust(12)]
        for i in range(self.sampler.ndim):
            line = f"{i:4d} {stats[i, 0]:11.4g} {stats[i, 1]:11.4g}"
            line += f" {stats[i, 2]:11.4g} {stats[i, 3]:11.4g} {stats[i, 4]:11.4g}"
            out += [line]
        return "\n".join(out)

    def save(self, options: dict):
        # NOTE: Remember to include citation info.
        print("TODO: Save run data.")
