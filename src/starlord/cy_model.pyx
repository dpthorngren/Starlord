cdef class BaseModel:
    # Static metadata (Python objects)
    # To be filled by subclass definition
    param_names: list[str] = []
    output_names: list[str] = []
    var_names: list[str] = []
    const_names: list[str] = []
    optional_consts: list[str] = []
    optional_likelihood_terms: bool = False
    # To be filled by module loader at load time
    code_hash: list[str] = []
    code: list[str] = []

    # ===== Functions overridden by subclasses =====
    cpdef double[:] prior_transform(self, double[:] params):
        return params

    cpdef double log_prior(self, double[:] params):
        return 0.0

    cdef void _forward_model(self, double[:] params):
        return

    cdef double _log_like(self, double[:] params):
        return -math.INFINITY

    cpdef postprocess(self, double[:,:] params, double[:,:] out):
        return

    # ===== Functions not overridden by subclasses =====
    cpdef dict forward_model(self, double[:] params):
        self._forward_model(params)
        return {k: getattr(self, "l__"+k) for k in self.var_names}

    cpdef double log_like(self, double[:] params):
        self._forward_model(params)
        return self._log_like(params)

    cpdef double log_prob(self, double[:] params):
        cdef double log_prior = self.log_prior(params)
        if not math.isfinite(log_prior):
            return -math.INFINITY
        return log_prior + self.log_like(params)

    cpdef load_constants(self, dict constants):
        from starlord import GridGenerator
        for c in self.const_names:
            if c in constants.keys():
                setattr(self, f"c__{c}", constants[c])
            elif c.startswith("grid__"):
                _, grid, outname = c.split("__")
                setattr(self, f"c__{c}", GridGenerator.get_grid(grid).build_grid(outname))
            else:
                raise ValueError(f"Value for constant {c} was not provided on initialization.")

    cpdef object generate_initial_state(self, samples=1, steps=100):
        '''Simple metropolis sampler proposing from the prior for initial state generation.'''
        cdef int i, j, k
        cdef double p, log_l, proposal_log_l
        cdef int ndim = self.ndim
        proposal = np.full([ndim], math.NAN)
        cdef double[:] proposal_view = proposal
        result = np.full([samples, ndim], math.NAN)
        cdef double[:, :] result_view = result

        np.random.seed(int(os.urandom(4).hex(),16))
        srand(np.random.rand() * int(os.urandom(4).hex(),16))
        for i in range(samples):
            log_l = -math.INFINITY
            for j in range(steps):
                for k in range(ndim):
                    proposal_view[k] = float(rand()) / RAND_MAX
                self.prior_transform(proposal_view)
                proposal_log_l = self.log_like(proposal_view)
                p = math.log(float(rand()) / RAND_MAX)
                if p < proposal_log_l - log_l:
                    copy_arr1d(proposal_view, result_view[i])
                    log_l = proposal_log_l
        return result

    def __init__(self, **constants):
        self.load_constants(constants)
