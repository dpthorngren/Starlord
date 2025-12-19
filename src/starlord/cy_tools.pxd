cimport scipy.special.cython_special as special
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas
from libc cimport math

cpdef double uniform_lpdf(double x, double xmin, double xmax) noexcept
cpdef double uniform_ppf(double x, double xmin, double xmax) noexcept
cpdef double normal_lpdf(double x, double mean, double sigma) noexcept
cpdef double normal_ppf(double p, double mean, double sigma) noexcept
cpdef double beta_lpdf(double x, double alpha, double beta) noexcept
cpdef double beta_ppf(double p, double alpha, double beta) noexcept
cpdef double gamma_lpdf(double x, double alpha, double lamb) noexcept
cpdef double gamma_ppf(double p, double alpha, double lamb) noexcept

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
