cimport scipy.special.cython_special
cimport scipy.linalg.cython_lapack as lapack
cimport scipy.linalg.cython_blas as blas
from libc cimport math

cpdef double normal_lpdf(double x, double mean, double sigma)
cpdef double normal_ppf(double p, double mean, double sigma)
