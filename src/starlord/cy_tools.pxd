cimport scipy.special.cython_special as special
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas
from libc cimport math
from libc.stdlib cimport rand, srand, RAND_MAX

cpdef void copy_arr1d(double[:] source, double[:] dest)
cpdef void copy_arr2d(double[:,:] source, double[:,:] dest)

cpdef double logsumexp(double x, double y, double c_x=?, double c_y=?) noexcept
cpdef double uniform_lpdf(double x, double xmin, double xmax) noexcept
cpdef double uniform_ppf(double x, double xmin, double xmax) noexcept
cpdef double normal_lpdf(double x, double mean, double sigma) noexcept
cpdef double normal_ppf(double p, double mean, double sigma) noexcept
cpdef double normal_cdf(double x, double mean, double sigma) noexcept
cpdef double beta_lpdf(double x, double alpha, double beta) noexcept
cpdef double beta_ppf(double p, double alpha, double beta) noexcept
cpdef double gamma_lpdf(double x, double alpha, double lamb) noexcept
cpdef double gamma_ppf(double p, double alpha, double lamb) noexcept
cpdef double exponential_lpdf(double x, double rate) noexcept
cpdef double exponential_ppf(double p, double rate) noexcept
cpdef double exponential_cdf(double x, double rate) noexcept
cpdef double trunc_power_lpdf(double x, double k, double a, double b) noexcept
cpdef double trunc_power_ppf(double p, double k, double a, double b) noexcept
cpdef double trunc_normal_lpdf(double x, double mean, double sigma, double a, double b) noexcept
cpdef double trunc_normal_ppf(double p, double mean, double sigma, double a, double b) noexcept
cpdef double trunc_exponential_lpdf(double x, double rate, double a, double b) noexcept
cpdef double trunc_exponential_ppf(double p, double rate, double a, double b) noexcept
cpdef double chabrier_lpdf(double log_mass, double log_m_switch, double mean, double sigma, double power) noexcept
cpdef double chabrier_ppf(double p, double log_m_switch, double mean, double sigma, double power) noexcept
cpdef void multinormal_zppf(double[:,:] cov_chol, double[:] z, double[:] out, double[:] mean=?)


cdef int _locatePoint_(double point, double[:] axis, int axLen, double* w) noexcept
cdef double _unit_interp3(double[:] values, int s, int xs, int ys, int zs, double xw, double yw, double zw) noexcept

cdef class GridInterpolator:
    cdef readonly int ndim
    cdef int x_len
    cdef int y_len
    cdef int z_len
    cdef int u_len
    cdef int v_len
    cdef int x_stride
    cdef int y_stride
    cdef int z_stride
    cdef int u_stride
    cdef double[:] x_axis
    cdef double[:] y_axis
    cdef double[:] z_axis
    cdef double[:] u_axis
    cdef double[:] v_axis
    cdef double[:] values
    cdef object _data
    cdef readonly object bounds
    cdef readonly object shape

    cpdef double interp(self, double[:] x)
    cpdef double _interp1d(self, double point) noexcept
    cpdef double _interp2d(self, double x, double y) noexcept
    cpdef double _interp3d(self, double x, double y, double z) noexcept
    cpdef double _interp4d(self, double x, double y, double z, double u) noexcept
    cpdef double _interp5d(self, double x, double y, double z, double u, double v) noexcept

cdef class BaseModel:

    # ===== Functions overridden by subclasses =====
    cpdef double[:] prior_transform(self, double[:] params)
    cpdef double log_prior(self, double[:] params)
    cdef void _forward_model(self, double[:] params)
    cdef double _log_like(self, double[:] params)
    cpdef postprocess(self, double[:,:] params, double[:,:] out)

    # ===== Functions not overridden by subclasses =====
    cpdef dict forward_model(self, double[:] params)
    cpdef double log_like(self, double[:] params)
    cpdef double log_prob(self, double[:] params)
    cpdef load_constants(self, dict constants)
    cpdef object generate_initial_state(self, samples=?, steps=?)

cdef class BuiltinSampler:
    # Internal data
    cdef BaseModel model
    cdef readonly int n_dim
    cdef readonly int n_walkers
    cdef object _working_memory_
    cdef double[:, :] walkers
    cdef double[:] x_propose
    cdef double[:] temp
    cdef double[:, :] propose_chol

    # Outputs
    cdef int trials_metropolis
    cdef int accepted_metropolis
    cdef int trials_stretch
    cdef int accepted_stretch
    cdef readonly object _samples_memory_
    cdef double[:, :, :] samples

    cdef int _init_working_memory(self) except -1
    cdef int _progress_bar(self, int i, int N, object header) except -1
    cdef int stretch_step(self, double alpha=?) except -1
    cdef int metropolis_step(self) except -1
    cpdef void run(self, double[:,:] initial_state, int n_samples, int burn_in, int thin=?, bint progress=?, double alpha=?, double metropolis_frac=?, int metropolis_presamples=?)
    cpdef object get_samples(self, bint flatten=?)
    cpdef object get_log_prob(self, bint flatten=?)
    cpdef (float, float) get_acceptance(self)
