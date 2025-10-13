cimport scipy.special.cython_special as special
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas
from libc cimport math

cpdef double uniform_lpdf(double x, double xmin, double xmax)
cpdef double uniform_ppf(double x, double xmin, double xmax)
cpdef double normal_lpdf(double x, double mean, double sigma)
cpdef double normal_ppf(double p, double mean, double sigma)
cpdef double beta_lpdf(double x, double alpha, double beta)
cpdef double beta_ppf(double p, double alpha, double beta)
cpdef double gamma_lpdf(double x, double alpha, double lamb)
cpdef double gamma_ppf(double p, double alpha, double lamb)

cdef inline int _locatePoint_(double point, double[:] axis, int axLen, double* w)

cdef class GridInterpolator:
    cdef int ndim
    cdef int x_len
    cdef int y_len
    cdef int z_len
    cdef int u_len
    cdef int v_len
    cdef int x_stride
    cdef int y_stride
    cdef int z_stride
    cdef int u_stride
    cdef int v_stride
    cdef double[:] x_axis
    cdef double[:] y_axis
    cdef double[:] z_axis
    cdef double[:] u_axis
    cdef double[:] v_axis
    cdef double[:] values
    cdef object _data

    cpdef double interp(self, double[:] x)
    cpdef double _interp1d(self, double point)
    cpdef double _interp2d(self, double x, double y)
