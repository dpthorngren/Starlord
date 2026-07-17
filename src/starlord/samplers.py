from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Optional, Type

import dynesty
import emcee
import numpy as np
from dynesty.results import Results as DynestyResults

from ._config import __version__
from .cy_tools import BaseModel, BuiltinSampler
from .grid_gen import GridGenerator


@dataclass
class ResultStats:
    mean: np.ndarray
    cov: np.ndarray
    std: np.ndarray
    p16: np.ndarray
    p50: np.ndarray
    p84: np.ndarray

    def summary(self, param_names=None, output_names=None):
        n_outputs = len(output_names) if output_names is not None else 0
        if param_names is not None:
            n_params = len(param_names)
        else:
            n_params = len(self.mean) - n_outputs
            param_names = [""] * n_params
        out = ["     Name".ljust(29) + "Mean".rjust(12) + "Std".rjust(12)]
        out[0] += "16%".rjust(12) + "50%".rjust(12) + "84%".rjust(12)
        for i in range(n_params):
            line = f"{i:4d} {param_names[i]:24}"
            line += f" {self.mean[i]:11.4g} {self.std[i]:11.4g}"
            line += f" {self.p16[i]:11.4g} {self.p50[i]:11.4g} {self.p84[i]:11.4g}"
            out += [line]
        if output_names:
            out += [89 * "-"]
            for i, name in enumerate(output_names):
                i = i + n_params
                line = f"{i:4d} {name:24}"
                line += f" {self.mean[i]:11.4g} {self.std[i]:11.4g}"
                line += f" {self.p16[i]:11.4g} {self.p50[i]:11.4g} {self.p84[i]:11.4g}"
                out += [line]
        return "\n".join(out)

    def to_array(self, include_cov=True):
        result = np.vstack([self.mean, self.std, self.p16, self.p50, self.p84])
        if include_cov:
            result = np.vstack([result, self.cov])
        return result.T

    @classmethod
    def create_from_array(cls, arr: np.ndarray):
        assert arr.ndim == 2
        s = arr.shape[0]
        assert arr.shape[1] == 5 + s
        return cls(arr[:, 0], arr[:, 5:], *arr[:, 1:5].T)

    @classmethod
    def create_from_post(cls, posterior: np.ndarray, weights: Optional[np.ndarray] = None):
        assert type(posterior) is np.ndarray
        if weights is not None:
            assert type(weights) is np.ndarray
            mean, cov = dynesty.utils.mean_and_cov(posterior, weights)
            q = np.array([dynesty.utils.quantile(p, [0.16, 0.5, 0.84], weights=weights) for p in posterior.T]).T
        else:
            mean = posterior.mean(axis=0)
            std = posterior.std(axis=0)
            cov = np.cov(posterior.T)
            q = np.quantile(posterior, [.16, .5, .84], axis=0)
        std = np.sqrt(np.diag(cov))
        result = ResultStats(mean, cov, std, q[0], q[1], q[2])
        return result


class _Sampler:
    '''Abstract class for objects which can sample from probability distributions.'''
    init_args: dict
    _constants: dict[str, float]
    _check_constants: bool
    _model_class: Type[BaseModel]
    _post: Optional[np.ndarray]
    _model: Optional[BaseModel]
    _stats: Optional[ResultStats]
    _last_run_args: dict
    _last_init_args: dict
    _last_constants: list[float]

    @property
    def constants(self) -> dict[str, float]:
        self._check_constants = True
        return self._constants

    @property
    def model(self) -> BaseModel:
        if self._model is None:
            self._model = self._model_class(**self._constants)
        elif self._check_constants:
            for key, value in self._constants.items():
                setattr(self._model, "c__" + key, value)
        return self._model

    @property
    def param_names(self) -> list[str]:
        return self._model_class.param_names

    @property
    def output_names(self) -> list[str]:
        return self._model_class.output_names

    @property
    def const_names(self) -> list[str]:
        return [c for c in self._model_class.const_names if not c.startswith("grid__")]

    @property
    def grids_used(self) -> dict[str, list[str]]:
        results = {}
        for c in self._model_class.const_names:
            if c.startswith("grid__"):
                _, grid_name, var = c.split("__")
                results.setdefault(grid_name, []).append(var)
        return results

    @property
    def optional_consts(self) -> list[str]:
        return self._model_class.optional_consts

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

    @property
    def results(self) -> object:
        return self.post

    def __init__(self, model_class, constants={}, **init_args):
        self._model_class = model_class
        self._constants = constants
        self.init_args = init_args
        self._check_constants = False
        self._post = None
        self._model = None
        self._stats = None
        self._last_init_args = {}
        self._last_run_args = {}
        self._last_constants = []

    def validate_constants(self, allow_nan=False):
        expected = set(self.const_names) - set(self.optional_consts)
        missing = expected - set(self._constants.keys())
        extra = set(self._constants.keys()) - expected
        assert not missing, "Missing values for constant(s) " + ", ".join(missing)
        if extra:
            print("Warning, unused constants: " + ", ".join(extra))
        for cname in expected:
            val = self._constants[cname]
            assert allow_nan or np.isfinite(val), f"Invalid value for constant c.{cname} = {val}"

    def summary(self) -> str:
        citations = self.get_citations()
        if citations:
            print("Grid Citations:")
            print("    " + "\n    ".join(citations))
        return self.stats.summary(self.param_names, self.output_names)

    def run(self, **run_args):
        raise NotImplementedError("Do not use _Sampler directly, pick a subclass.")

    def save_results(self, filename: str):
        np.savez_compressed(filename, **self._to_dict_())

    def save_corner(self, filename, **kwargs):
        from starlord.io import corner_plot
        kwargs.setdefault('labels', self.param_names)
        corner_plot(self.post[:, :self.ndim], filename, **kwargs)

    def get_citations(self) -> list[str]:
        citations = []
        for gridname in self.grids_used.keys():
            grid_citations = GridGenerator.get_grid(gridname).citations
            citations.append(f"{gridname}: {grid_citations}")
        return citations

    def _to_dict_(self) -> dict:
        grid_vars = []
        for gridname, keys in self.grids_used.items():
            grid_vars += ([f"{gridname}__{key}" for key in keys])
        return dict(
            params=self.post[:, :self.ndim],
            outputs=self.post[:, self.ndim:],
            consts=self._last_constants,
            output_names=self.output_names,
            param_names=self.param_names,
            const_names=self.const_names,
            code=self.model.code[0],
            code_hash=self.model.code_hash[0],
            grids=list(self.grids_used.keys()),
            grid_vars=grid_vars,
            stats=self.stats.to_array(),
            citations='\n'.join(self.get_citations()),
            time=str(datetime.datetime.now(datetime.timezone.utc).ctime() + " UTC"),
            starlord_version=__version__,
            python_version=sys.version,
        )

    def batch_run(
        self,
        run_args: dict,
        infile: str | Path,
        terminal_output: bool = True,
        postfile: Optional[str] = None,
        summaryfile: Optional[str] = None,
        threads: int = 1,
    ) -> np.ndarray:
        # Read in the constants data from the provided file
        data = np.genfromtxt(
            infile,
            delimiter=",",
            comments="#",
            autostrip=True,
            names=True,
            dtype=None,
            encoding="UTF-8",
        )
        columns = data.dtype.names
        assert columns is not None, f"Failed to read column names in {infile}."
        columns = [n for n in columns if n in self.const_names + ['name']]
        nongrid_consts = [c for c in self.const_names if not c.startswith("grid__")]
        if "name" not in columns:
            data['name'] = np.arange(len(data))
        names = data['name'].copy()
        work = [{c: row[c] for c in columns} for row in data]

        task = partial(
            self._run_single_,
            run_args=run_args,
            terminal_output=terminal_output,
            postfile=postfile,
            summary_cols=nongrid_consts,
        )

        if threads > 1:
            with Pool(threads) as pool:
                results = list(pool.map(task, work))
        else:
            results = list(map(task, work))

        if summaryfile is not None:
            assert summaryfile != infile, "Error: will not output to input csv file (would overwrite!)"
            assert results is not None
            header = ["name"] + nongrid_consts
            for p in self.param_names + self.output_names:
                header += [p + stat for stat in ('_mean', '_std', '_p16', '_p50', '_p84')]
            summary_rows = []
            for name, input, output in zip(names, work, results):
                assert output is not None
                row = [name]
                row += [f"{input[c]:.6f}" for c in nongrid_consts]
                row += [f"{v:.6f}" for v in output]
                summary_rows.append(", ".join(row))
            with open(summaryfile, 'w') as fd:
                fd.write(", ".join(header) + "\n")
                fd.write("\n".join(summary_rows) + "\n")
        return np.asarray(results)

    def _run_single_(
        self,
        constants,
        run_args: dict,
        terminal_output: bool = True,
        postfile: Optional[str] = None,
        summary_cols: list[str] = [],
    ) -> np.ndarray:
        name = constants.pop('name', '')
        print(name, ", ".join([f"{k} = {v}" for k, v in constants.items()]))
        self.constants.update(constants)
        try:
            self.run(**run_args)
            if terminal_output:
                print(name, self.summary())
            if postfile is not None:
                self.save_results(postfile + "_" + name.replace(" ", "_"))
            return self.stats.to_array(False).flatten()
        except Exception as e:
            print(f"Error: {name} raised exception {e}")
        return np.full(5 * len(summary_cols), np.nan)


class SamplerBuiltin(_Sampler):
    '''Wrapper for Starlord's built-in sampler, an Ensemble sampler.  The algorithm
    is similar to Emcee's (SamplerEnsemble) but has fewer features in exchange
    for very low calling overhead.'''
    _sampler: BuiltinSampler

    @property
    def sampler(self) -> BuiltinSampler:
        assert self._sampler is not None, "Must run sampler before accessing it."
        return self._sampler

    @property
    def results(self) -> np.ndarray:
        return self.post[:, :-2]

    def __init__(self, model_class, constants={}, **init_args) -> None:
        super().__init__(model_class, constants, **init_args)
        self.init_args.setdefault("nwalkers", max(40, 3 * self.ndim))
        self.init_args.setdefault("ndim", self.ndim)

    def run(self, **run_args):
        self.validate_constants(self._model_class.optional_likelihood_terms)
        self._last_init_args = self.init_args.copy()
        self._sampler = BuiltinSampler(self.model, self.init_args['ndim'], self.init_args['nwalkers'])
        run_args = run_args.copy()
        run_args.setdefault('n_samples', 4000)
        run_args.setdefault('burn_in', 500)
        run_args.setdefault('thin', 5)
        run_args.setdefault('alpha', 2.0)
        self._last_run_args = run_args.copy()
        self._last_constants = [getattr(self.model, f"c__{c}") for c in self.const_names if not c.startswith("grid")]

        # Prepare an initial state matrix
        if "initial_state" not in run_args:
            assert self.prior_transform is not None, "Must provide initial_state or prior_transform."
            run_args['initial_state'] = self.model.generate_initial_state(self.init_args['nwalkers'], 100)
            assert np.all(
                np.isfinite(run_args['initial_state'])
            ), "Failed to generate a valid initial state from 100 draws.  Check your prior bounds."
        self.sampler.run(**run_args)

        # Process the results
        results = self.sampler.get_samples(True)
        assert results is not None and type(results) is np.ndarray
        postprocessed = np.zeros((results.shape[0], len(self.output_names)))
        self.postprocess(results, postprocessed)
        self._post = np.hstack([results, postprocessed])
        self._stats = ResultStats.create_from_post(self._post)


class SamplerEnsemble(_Sampler):
    '''Thin wrapper for EMCEE's EnsembleSampler'''
    _sampler: emcee.EnsembleSampler | None
    burn_in: int
    thin: int

    @property
    def sampler(self) -> emcee.EnsembleSampler:
        assert self._sampler is not None, "Must run sampler before accessing it."
        return self._sampler

    @property
    def results(self) -> object:
        return self.sampler.get_chain(flat=True, discard=self.burn_in, thin=self.thin)

    def __init__(self, model_class, constants={}, burn_in=500, thin=1, **init_args) -> None:
        super().__init__(model_class, constants, **init_args)
        self._sampler = None
        self.burn_in = burn_in
        self.thin = thin

    def summary(self) -> str:
        try:
            convergence = self.sampler.get_autocorr_time(thin=self.thin, discard=self.burn_in)
            convergence = max(convergence)
            neff = self._last_run_args['nsteps'] / convergence
            summary = f"Convergence: Tau = {convergence:.2f}; N/Tau = {neff:.2f}\n"
        except emcee.autocorr.AutocorrError:
            summary = "Too few samples to estimate convergence.\n"
        summary += super().summary()
        return summary

    def run(self, threads=1, **run_args):
        self.validate_constants(self._model_class.optional_likelihood_terms)
        # Propagate sampler settings
        init_args = self.init_args.copy()
        init_args.setdefault('nwalkers', max(100, 5 * self.ndim))
        init_args.setdefault('ndim', self.ndim)
        init_args.setdefault('log_prob_fn', self.log_prob)
        self._last_init_args = init_args.copy()
        run_args = run_args.copy()
        run_args.setdefault('nsteps', 5000)
        run_args.setdefault('progress', True)
        self._last_run_args = run_args.copy()
        self._last_constants = [getattr(self.model, f"c__{c}") for c in self.const_names if not c.startswith("grid")]

        # Prepare an initial state matrix
        if "initial_state" not in run_args:
            assert self.prior_transform is not None, "Must provide initial_state or prior_transform."
            run_args['initial_state'] = self.model.generate_initial_state(init_args['nwalkers'], 100)
            assert np.all(
                np.isfinite(run_args['initial_state'])
            ), "Failed to generate a valid initial state from 100 draws.  Check your prior bounds."
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
        self._stats = ResultStats.create_from_post(self._post)


class SamplerNested(_Sampler):
    '''Thin wrapper for the Dynesty NestedSampler'''
    _sampler: dynesty.DynamicNestedSampler | None

    @property
    def sampler(self) -> dynesty.DynamicNestedSampler:
        assert self._sampler is not None, "Must run sampler before accessing it."
        return self._sampler

    @property
    def results(self) -> DynestyResults:
        return self.sampler.results

    def __init__(self, model_class, constants={}, **init_args) -> None:
        super().__init__(model_class, constants, **init_args)
        self._sampler = None

    def run(self, **run_args):
        self.validate_constants(self._model_class.optional_likelihood_terms)
        # Propagate sampler settings
        init_args = self.init_args.copy()
        init_args.setdefault('ndim', self.ndim)
        init_args.setdefault('loglikelihood', self.log_like)
        init_args.setdefault('prior_transform', self.prior_transform)
        self._last_init_args = init_args.copy()
        self._last_run_args = run_args.copy()
        self._sampler = dynesty.DynamicNestedSampler(**init_args)
        self._last_constants = [getattr(self.model, f"c__{c}") for c in self.const_names if not c.startswith("grid")]
        self.sampler.run_nested(**run_args)

        # Process the results
        assert self.results is not None and type(self.results) is DynestyResults
        post = self.results.samples  # type: ignore
        postprocessed = np.zeros((post.shape[0], len(self.output_names)))
        self.postprocess(post, postprocessed)
        self._post = np.hstack([post, postprocessed])
        weights = self.sampler.results.importance_weights()
        self._stats = ResultStats.create_from_post(self._post, weights)

    def save_results(self, filename: str):
        result = self._to_dict_()
        result['weights'] = self.results.importance_weights()
        np.savez_compressed(filename, **self._to_dict_())

    def save_corner(self, filename, **kwargs):
        from starlord.io import corner_plot
        assert self.post is not None, "Cannot generate a plot before running the sampler."
        kwargs.setdefault('labels', self.param_names)
        corner_plot(self.results, filename, weights=self.sampler.results.importance_weights(), **kwargs)
