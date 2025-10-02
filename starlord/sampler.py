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
    def summary(self) -> str:
        pass

    @abstractmethod
    def save(self, options: dict):
        pass


class SamplerNested(_Sampler):
    '''Thin wrapper for the Dynesty NestedSampler'''

    def __init__(self, loglike: Callable, ptform: Callable, ndim: int, config: dict) -> None:
        # TODO: Parameter names
        self._sampler = dynesty.NestedSampler(loglike, ptform, ndim, **config)

    @property
    def sampler(self):
        return self._sampler

    @property
    def results(self):
        return self._sampler.results

    def run(self, options: dict):
        self._sampler.run_nested(**options)

    def summary(self) -> str:
        # TODO: Convergence statistics
        samples = self.results['samples']
        weights = self.results.importance_weights()
        out = [" Dim" + "Mean".rjust(12) + "Std".rjust(12) + "16".rjust(12) + "50".rjust(12) + "84".rjust(12)]
        mean, cov = dynesty.utils.mean_and_cov(samples, weights)
        for i in range(self.sampler.ndim):
            line = f"{i:4d} {mean[i]:11.4g} {np.sqrt(cov[i,i]):11.4g}"
            q = dynesty.utils.quantile(samples[:, i], [0.16, 0.5, 0.84], weights=weights)
            line += f" {q[0]:11.4g} {q[1]:11.4g} {q[2]:11.4g}"
            out += [line]
        return "\n".join(out)

    def save(self, options: dict):
        # NOTE: Remember to include citation info.
        print("TODO: Save run data.")
