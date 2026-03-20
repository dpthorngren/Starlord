from __future__ import annotations

from collections import namedtuple
from multiprocessing import Pool
from typing import Callable, Optional

import dynesty
import emcee
import numpy as np
from dynesty.results import Results as DynestyResults

ResultStats = namedtuple("ResultStats", ["mean", 'cov', 'std', 'p16', 'p50', 'p84'])
_dummyModel = namedtuple(
    "DummyModel", [
        'param_names', 'const_names', 'optional_consts', 'forward_model', 'log_like', 'log_prob', 'prior_transform',
        'log_prior', 'output_names', 'postprocess'
    ])


class _Sampler:
    '''Abstract class for objects which can sample from probability distributions.'''
    init_args: dict
    _constants: dict[str, float]
    _check_constants: bool
    _model_class: Callable[..., _dummyModel]
    _post: Optional[np.ndarray]
    _model: Optional[_dummyModel]
    _stats: Optional[ResultStats]

    @property
    def constants(self) -> dict[str, float]:
        self._check_constants = True
        return self._constants

    @property
    def model(self) -> _dummyModel:
        if self._model is None:
            self._model = self._model_class(**self._constants)
        elif self._check_constants:
            for key, value in self._constants.items():
                setattr(self._model, "c__" + key, value)
        return self._model

    @property
    def param_names(self) -> list[str]:
        return self._model_class.param_names  # type: ignore

    @property
    def output_names(self) -> list[str]:
        return self._model_class.output_names  # type: ignore

    @property
    def const_names(self) -> list[str]:
        return self._model_class.const_names  # type: ignore

    @property
    def optional_consts(self) -> list[str]:
        return self._model_class.optional_consts  # type: ignore

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    @property
    def forward_model(self) -> Callable:
        return self.model.forward_model

    @property
    def log_prob(self) -> Callable:
        return self.model.log_prob

    @property
    def log_like(self) -> Callable:
        return self.model.log_like

    @property
    def log_prior(self) -> Callable:
        return self.model.log_prior

    @property
    def prior_transform(self) -> Callable:
        return self.model.prior_transform

    @property
    def postprocess(self) -> Callable:
        return self.model.postprocess

    @property
    def stats(self) -> ResultStats:
        assert self._stats is not None, "Cannot read stats before running the model"
        return self._stats

    @property
    def post(self) -> np.ndarray:
        assert self._post is not None, "Cannot read results before running the model"
        return self._post

    def __init__(self, model_class, constants={}, **init_args):
        self._model_class = model_class
        self._constants = constants
        self.init_args = init_args
        self._check_constants = False
        self._post = None
        self._model = None
        self._stats = None

    def validate_constants(self):
        expected = set(self.const_names) - set(self.optional_consts)
        missing = expected - set(self._constants.keys())
        extra = set(self._constants.keys()) - expected
        assert not missing, "Missing values for constant(s) " + ", ".join(missing)
        if extra:
            print("Warning, unused constants: " + ", ".join(extra))
        for cname in expected:
            val = self._constants[cname]
            assert np.isfinite(val), f"Invalid value for constant c.{cname} = {val}"

    def summary(self) -> str:
        # TODO: Convergence statistics
        out = ["     Name".ljust(29) + "Mean".rjust(12) + "Std".rjust(12)]
        out[0] += "16%".rjust(12) + "50%".rjust(12) + "84%".rjust(12)
        for i in range(self.ndim):
            line = f"{i:4d} {self.param_names[i]:24}"
            line += f" {self.stats.mean[i]:11.4g} {self.stats.std[i]:11.4g}"
            line += f" {self.stats.p16[i]:11.4g} {self.stats.p50[i]:11.4g} {self.stats.p84[i]:11.4g}"
            out += [line]
        if self.model.output_names and self.post is not None:
            out += [89 * "-"]
            for i, name in enumerate(self.model.output_names):
                i = i + self.ndim
                line = f"{i:4d} {name:24}"
                line += f" {self.stats.mean[i]:11.4g} {self.stats.std[i]:11.4g}"
                line += f" {self.stats.p16[i]:11.4g} {self.stats.p50[i]:11.4g} {self.stats.p84[i]:11.4g}"
                out += [line]
        return "\n".join(out)


class SamplerEnsemble(_Sampler):
    '''Thin wrapper for EMCEE's EnsembleSampler'''
    _sampler: emcee.EnsembleSampler | None
    burn_in: int
    thin: int

    @property
    def sampler(self) -> emcee.EnsembleSampler:
        assert self._sampler is not None, "Must run sampler before accessing it."
        return self._sampler

    def __init__(self, model_class, constants={}, burn_in=500, thin=1, **init_args) -> None:
        super().__init__(model_class, constants, **init_args)
        self._sampler = None
        self.burn_in = burn_in
        self.thin = thin

    @property
    def results(self) -> object:
        return self.sampler.get_chain(flat=True, discard=self.burn_in, thin=self.thin)

    def run(self, threads=1, **run_args):
        self.validate_constants()
        # Propagate sampler settings
        init_args = self.init_args.copy()
        init_args.setdefault('nwalkers', max(100, 5 * self.ndim))
        init_args.setdefault('ndim', self.ndim)
        init_args.setdefault('log_prob_fn', self.log_prob)
        run_args = run_args.copy()
        run_args.setdefault('nsteps', 2500)
        run_args.setdefault('progress', True)

        # Prepare an initial state matrix
        if "initial_state" not in run_args:
            assert self.prior_transform is not None, "Must provide initial_state or prior_transform."
            run_args['initial_state'] = 0.3 + 0.4 * np.random.rand(init_args['nwalkers'], self.ndim)
            [self.prior_transform(s) for s in run_args['initial_state']]
        run_args['nsteps'] += self.burn_in

        # Run the MCMC
        if threads > 1:
            with Pool(threads) as pool:
                self._sampler = emcee.EnsembleSampler(pool=pool, **init_args)
                self.sampler.run_mcmc(**run_args)
        else:
            self._sampler = emcee.EnsembleSampler(**init_args)
            self.sampler.run_mcmc(**run_args)

        # Process the results
        assert self.results is not None and type(self.results) is np.ndarray
        postprocessed = np.zeros((self.results.shape[0], len(self.output_names)))
        self.postprocess(self.results, postprocessed)
        self._post = np.hstack([self.results, postprocessed])
        mean = self.post.mean(axis=0)
        std = self.post.std(axis=0)
        cov = np.cov(self.post.T)
        std = np.sqrt(np.diag(cov))
        q = np.quantile(self.post, [.16, .5, .84], axis=0)
        self._stats = ResultStats(mean, cov, std, q[0], q[1], q[2])

    def save_results(self, filename: str):
        assert self.post is not None, "Cannot save results before running the sampler."
        np.savez_compressed(filename, samples=self.post)


class SamplerNested(_Sampler):
    '''Thin wrapper for the Dynesty NestedSampler'''
    _sampler: dynesty.NestedSampler | None

    @property
    def sampler(self) -> dynesty.NestedSampler:
        assert self._sampler is not None, "Must run sampler before accessing it."
        return self._sampler

    @property
    def results(self) -> DynestyResults:
        return self.sampler.results

    def __init__(self, model_class, constants={}, **init_args) -> None:
        super().__init__(model_class, constants, **init_args)
        self._sampler = None

    def run(self, **run_args):
        self.validate_constants()
        # Propagate sampler settings
        init_args = self.init_args.copy()
        init_args.setdefault('ndim', self.ndim)
        init_args.setdefault('loglikelihood', self.log_like)
        init_args.setdefault('prior_transform', self.prior_transform)
        self._sampler = dynesty.NestedSampler(**init_args)
        self.sampler.run_nested(**run_args)

        # Process the results
        assert self.results is not None and type(self.results) is DynestyResults
        post = self.results.samples  # type: ignore
        postprocessed = np.zeros((post.shape[0], len(self.output_names)))
        self.postprocess(post, postprocessed)
        self._post = np.hstack([post, postprocessed])
        samples = self.results['samples']
        weights = self.sampler.results.importance_weights()
        mean, cov = dynesty.utils.mean_and_cov(samples, weights)
        std = np.sqrt(np.diag(cov))
        q = np.array([
            dynesty.utils.quantile(samples[:, i], [0.16, 0.5, 0.84], weights=weights) for i in range(samples.shape[1])
        ])
        self._stats = ResultStats(mean, cov, std, q[:, 0], q[:, 1], q[:, 2])

    def save_results(self, filename: str):
        # TODO: Citation info.
        np.savez_compressed(filename, samples=self.post, weights=self.results.importance_weights())
