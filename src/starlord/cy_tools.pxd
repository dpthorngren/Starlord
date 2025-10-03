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
