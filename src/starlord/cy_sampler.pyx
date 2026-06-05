cdef class BuiltinSampler:
    def __init__(self, BaseModel model, int n_dim, int n_walkers, double[:,:] metropolis_cov = None):
        assert n_dim > 0
        assert n_walkers > 2*n_dim
        self.model = model
        self.n_dim = n_dim
        self.n_walkers = n_walkers
        self.trials_metropolis = 0
        self.accepted_metropolis = 0
        self.trials_stretch = 0
        self.accepted_stretch = 0
        self._samples_memory_ = None
        self.samples = None
        self._init_working_memory()
        if metropolis_cov is not None:
            assert metropolis_cov.ndim == 2
            assert metropolis_cov.shape[0] == n_dim, "Proposal covariance must be [ndim x ndim]."
            assert metropolis_cov.shape[1] == n_dim, "Proposal covariance must be [ndim x ndim]."
            cov_chol = np.linalg.cholesky(metropolis_cov)
        else:
            x_median = np.full(n_dim, 0.5)
            model.prior_transform(x_median)
            x_offset = np.full(n_dim, 0.55)
            model.prior_transform(x_offset)
            cov_chol = np.diag(np.sqrt(np.abs(x_offset-x_median)))
        copy_arr2d(cov_chol, self.propose_chol)

    cdef int _init_working_memory(self) except -1:
        self._working_memory_ = np.empty([self.n_walkers+2+self.n_dim, self.n_dim+1])
        self.walkers = self._working_memory_[:self.n_walkers]
        self.x_propose = self._working_memory_[self.n_walkers, :self.n_dim]
        self.temp = self._working_memory_[self.n_walkers+1, :self.n_dim]
        self.propose_chol = self._working_memory_[self.n_walkers+2:, :self.n_dim]
        return 0

    cdef int stretch_step(self, double alpha=2.0) except -1:
        cdef double p, z, logp
        cdef int i, j, k
        cdef double root_alpha = math.sqrt(alpha)
        for i in range(self.n_walkers):
            # Select a walker to stretch towards
            j = rand() % (self.n_walkers-1)
            if j >= i:
                j += 1
            # Calculate stretch amount
            p = float(rand()) / RAND_MAX
            z = p*root_alpha + (1-p) / root_alpha
            z *= z
            # Get proposed position and log probability there
            for k in range(self.n_dim):
                self.x_propose[k] = self.walkers[j, k] + z * (self.walkers[i, k] - self.walkers[j, k])
            logp = self.model.log_prob(self.x_propose)
            # Calculate the metropolis correction and decide whether to accept the point
            adjust = math.log(z) * (self.n_dim - 1)
            p = math.log(float(rand()) / RAND_MAX)
            self.trials_stretch += 1
            if p < adjust + logp - self.walkers[i, self.n_dim]:
                self.accepted_stretch += 1
                copy_arr1d(self.x_propose, self.walkers[i, :self.n_dim])
                self.walkers[i, self.n_dim] = logp
        return 0

    cdef int metropolis_step(self) except -1:
        cdef int i, k
        cdef double accept_thresh, logp
        for i in range(self.n_walkers):
            # Get a proposed position and it's probability
            for k in range(self.n_dim):
                self.temp[k] = normal_ppf(float(rand()) / RAND_MAX, 0, 1.0)
            multinormal_zppf(self.propose_chol, self.temp, self.x_propose, self.walkers[i,:self.n_dim])
            logp = self.model.log_prob(self.x_propose)
            # Decide whether to accept the new position
            self.trials_metropolis += 1
            accept_thresh = math.log(float(rand()) / RAND_MAX)
            if accept_thresh < logp - self.walkers[i, self.n_dim]:
                self.accepted_metropolis += 1
                copy_arr1d(self.x_propose, self.walkers[i, :self.n_dim])
                self.walkers[i, self.n_dim] = logp
        return 0

    cdef int _sub_run_(self, int n_samples, int thin=1, bint record=False, bint progress=False, double alpha=2.0, double metropolis_frac=0.2) except -1:
        cdef int i
        cdef int total_steps = n_samples*thin

        if record:
            assert self.samples is not None and self.samples.shape[0] >= n_samples

        np.random.seed(int(os.urandom(4).hex(),16))
        srand(np.random.rand() * int(os.urandom(4).hex(),16))

        for i in range(total_steps):
            if (float(rand()) / RAND_MAX) < metropolis_frac:
                self.metropolis_step()
            else:
                self.stretch_step(alpha)
            if record and i % thin == 0:
                si = i / thin
                copy_arr2d(self.walkers, self.samples[si])
                if progress:
                    self._progress_bar(i, total_steps, "Sampling")
        return 0

    cpdef void run(self, double[:,:] initial_state, int n_samples, int burn_in, int thin=4, bint progress = False, double alpha=2.0, double metropolis_frac=0.2, int metropolis_presamples=-1):
        cdef int i, j, k, si
        # Validate inputs
        assert n_samples > 0
        assert burn_in >= 0
        assert thin >= 0
        assert alpha > 1.0
        assert initial_state.shape[0] == self.n_walkers and initial_state.shape[1] == self.n_dim
        if metropolis_presamples < 0:
            metropolis_presamples = n_samples // 10

        # Initialize output memory
        self._samples_memory_ = np.zeros([n_samples, self.n_walkers, self.n_dim+1])
        self.samples = self._samples_memory_

        # Copy initial state and get intial log probabilities
        copy_arr2d(initial_state, self.walkers[:, :self.n_dim])
        for j in range(self.n_walkers):
            self.walkers[j, self.n_dim] = self.model.log_prob(self.walkers[j])

        # Metropolis covariance pre-run
        if metropolis_frac > 0 and metropolis_presamples > 0:
            self._sub_run_(metropolis_presamples, thin, True, progress, alpha, metropolis_frac)
            covar = self._samples_memory_[:metropolis_presamples, :, :self.n_dim]
            covar = covar.reshape([metropolis_presamples*self.n_walkers, self.n_dim])
            covar = np.cov(covar.T)
            covar = np.linalg.cholesky(covar)
            copy_arr2d(covar, self.propose_chol)

        # Burn-in
        self._sub_run_(burn_in, thin, False, progress, alpha, metropolis_frac)
        # Collect samples
        self._sub_run_(n_samples, thin, True, progress, alpha, metropolis_frac)
        return

    cdef int _progress_bar(self, int i, int N, object header) except -1:
        '''Displays or updates a simple ASCII progress bar (code copied from dpthorngren/Sam).

        Args:
            i: the current iteration, should <= N.
            N: the total number of iterations to be done.
            header: a string to display before the progress bar.
        '''
        f = (10*i)//N
        sys.stdout.write('\r'+header+': <'+f*"="+(10-f)*" "+'> ('+str(i)+" / " + str(N) + ")          ")
        sys.stdout.flush()
        return 0

    cpdef object get_samples(self, bint flatten=False):
        assert self._samples_memory_ is not None, "Must run sampler before retrieving samples."
        assert type(self._samples_memory_) is np.ndarray
        if flatten:
            samples = self._samples_memory_.shape[0] * self._samples_memory_.shape[1]
            return self._samples_memory_[:, :, :self.n_dim].reshape([samples, self.n_dim])
        else:
            return self._samples_memory_[:, :, :self.n_dim]

    cpdef object get_log_prob(self, bint flatten=False):
        assert self._samples_memory_ is not None, "Must run sampler before retrieving samples."
        assert type(self._samples_memory_) is np.ndarray
        if flatten:
            return self._samples_memory_[:, :, self.n_dim].flatten()
        else:
            return self._samples_memory_[:, :, self.n_dim]

    cpdef (float, float) get_acceptance(self):
        cdef float metropolis_accept = float(self.accepted_metropolis) / self.trials_metropolis if self.trials_metropolis > 0 else math.NAN
        cdef float stretch_accept = float(self.accepted_stretch) / self.trials_stretch if self.trials_stretch > 0 else math.NAN
        return metropolis_accept, stretch_accept

    def __getstate__(self):
        '''Prepares internal memory for pickling, necessary for multiprocessing.'''
        info = (self.model, self.n_dim, self.n_walkers, self.acceptance, np.asarray(self.propose_chol))
        return info

    def __setstate__(self, info):
        '''Restores internal memory from pickle info, necessary for multiprocessing.'''
        (self.model, self.n_dim, self.n_walkers, self.acceptance, propose_chol) = info
        self._init_working_memory()
        copy_arr2d(self.propose_chol, propose_chol)

cpdef double gelman_rubin(x, warn=True) except -1.:
    x = np.asarray(x, np.double, "C")
    if x.ndim == 2:
        if warn:
            print("Warning: the G.R. diagnostic was not designed for " +\
                "the case where chains are not completely independent.")
        x = np.stack([x[:x.shape[0]//2],x[x.shape[0]//2:]],axis=0)
    elif x.ndim != 3:
        raise ValueError("Input has an invalid shape: must be 2-d or 3-d.")
    n = np.shape(x)[1]
    W = np.mean(np.var(x,axis=1,ddof=1),axis=0)
    B_n = np.var(np.mean(x,axis=1),axis=0,ddof=1)
    return np.max(np.sqrt((1.-1./n) + B_n/W))

