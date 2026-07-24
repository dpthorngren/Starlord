import numpy as np
import sys
import os
cimport cython

cpdef inline void copy_arr1d(double[:] source, double[:] dest):
    cdef int i
    assert source.shape[0] == dest.shape[0]
    for i in range(dest.shape[0]):
        dest[i] = source[i]

cpdef inline void copy_arr2d(double[:,:] source, double[:,:] dest):
    cdef int i, j
    assert source.shape[0] == dest.shape[0]
    assert source.shape[1] == dest.shape[1]
    for i in range(dest.shape[0]):
        for j in range(dest.shape[1]):
            dest[i, j] = source[i, j]

cpdef inline void copy_arr3d(double[:,:,:] source, double[:,:,:] dest):
    cdef int i, j, k
    assert source.shape[0] == dest.shape[0]
    assert source.shape[1] == dest.shape[1]
    assert source.shape[2] == dest.shape[2]
    for i in range(dest.shape[0]):
        for j in range(dest.shape[1]):
            for k in range(dest.shape[2]):
                dest[i, j, k] = source[i, j, k]

cpdef double expit(double x) noexcept:
    return 1. / (1. + math.exp(-x))

cpdef double logit(double x) noexcept:
    return math.log(x / (1. - x))

cpdef double logddx_logit(double x) noexcept:
    return -math.log(x - x*x)

cpdef double smootherstep(double x, double start, double end) noexcept:
    '''A sigmoid function smoothly interpolated from 0 to 1 between start and end;
    end does not have to be greater than start.
    See en.wikipedia.org/wiki/Smoothstep'''
    if end == start:
        return 0.5
    x = math.fmax(math.fmin((x-start)/(end-start), 1.), 0.)
    return x * x * x * (x * (6. * x - 15.) + 10.)

cpdef double logsumexp(double x, double y, double c_x=1., double c_y=1.) noexcept:
    cdef double baseline = max(x, y)
    x = c_x*math.exp(x - baseline)
    y = c_y*math.exp(y - baseline)
    return baseline + math.log(x+y)

cpdef double uniform_lpdf(double x, double xmin, double xmax) noexcept:
    if x >= xmin and x <= xmax:
        return -math.log(xmax - xmin)
    return -math.INFINITY

cpdef double uniform_ppf(double x, double xmin, double xmax) noexcept:
    return xmin + x * (xmax - xmin)

cpdef double normal_lpdf(double x, double mean, double sigma) noexcept:
    if sigma <= 0:
        return math.NAN
    return -(x-mean)**2/(2*sigma*sigma) - .5*math.log(2*math.M_PI*sigma*sigma)

cpdef double normal_ppf(double p, double mean, double sigma) noexcept:
    return -math.sqrt(2.) * special.erfcinv(2.*p)*sigma + mean

cpdef double normal_cdf(double x, double mean, double sigma) noexcept:
    x = (x - mean) / sigma
    return (1.0 + special.erf(x/math.sqrt(2)))/ 2.0

cpdef double beta_lpdf(double x, double alpha, double beta) noexcept:
    return (alpha-1.)*math.log(x) + (beta-1.)*math.log(1-x) - special.betaln(alpha, beta)

cpdef double beta_ppf(double p, double alpha, double beta) noexcept:
    return special.betaincinv(alpha, beta, p)

cpdef double gamma_lpdf(double x, double alpha, double lamb) noexcept:
    return (alpha-1.)*math.log(x*lamb) + math.log(lamb) - lamb*x - special.gammaln(alpha)

cpdef double gamma_ppf(double p, double alpha, double lamb) noexcept:
    return special.gammaincinv(alpha, p)/lamb

cpdef double exponential_lpdf(double x, double rate) noexcept:
    if rate <= 0:
        return math.NAN
    if x < 0.:
        return -math.INFINITY
    return math.log(rate) - rate*x

cpdef double exponential_ppf(double p, double rate) noexcept:
    if p > 1 or p < 0 or rate <= 0:
        return math.NAN
    return -math.log(1 - p) / rate

cpdef double exponential_cdf(double x, double rate) noexcept:
    if rate <= 0:
        return math.NAN
    if x <= 0:
        return 0.
    return 1. - math.exp(-rate*x)

cpdef double binorm_lpdf(double x, double weight1, double mean1, double mean2, double sigma1, double sigma2):
    if sigma1 <= 0 or sigma2 <=0 or weight1 > 1 or weight1 < 0:
        return math.NAN
    cdef double a = normal_lpdf(x, mean1, sigma1)
    cdef double b = normal_lpdf(x, mean2, sigma2)
    return logsumexp(a, b, weight1, 1-weight1)

cpdef double binorm_cdf(double x, double weight1, double mean1, double mean2, double sigma1, double sigma2):
    if sigma1 <= 0 or sigma2 <=0 or weight1 > 1 or weight1 < 0:
        return math.NAN
    cdef double a = normal_cdf(x, mean1, sigma1)
    cdef double b = normal_cdf(x, mean2, sigma2)
    return weight1*a + (1-weight1)*b

cpdef double binorm_ppf(double p, double weight1, double mean1, double mean2, double sigma1, double sigma2):
    if p < 0 or p > 1 or sigma1 <= 0 or sigma2 <=0 or weight1 > 1 or weight1 < 0:
        return math.NAN
    cdef double weight2 = 1 - weight1
    cdef double a = normal_ppf(p, mean1, sigma1)
    cdef double b = normal_ppf(p, mean2, sigma2)
    if a > b:
        a, b = b, a
    cdef double p_a = weight1*normal_cdf(a, mean1, sigma1) + weight2*normal_cdf(a, mean2, sigma2)
    cdef double p_b = weight1*normal_cdf(b, mean1, sigma1) + weight2*normal_cdf(b, mean2, sigma2)
    cdef double p_guess
    cdef int count = 0
    while abs(p_a - p_b) > 1e-12:
        count += 1
        if count%2 == 0:
            guess = a + (b-a) * (p-p_a) / (p_b-p_a)
        else:
            guess = (a + b) / 2.
        p_guess = weight1 * normal_cdf(guess, mean1, sigma1) + weight2 * normal_cdf(guess, mean2, sigma2)
        if p_guess > p:
            b = guess
            p_b = p_guess
        else:
            a = guess
            p_a = p_guess
        if count > 100:
            return np.nan
    return (a+b) / 2.

cpdef double trunc_power_lpdf(double x, double k, double a, double b) noexcept:
    if (a <= 0 and k <= 0) or b < 0 or a < 0:
        return math.NAN
    if x < a or x > b:
        return -math.INFINITY
    if k == -1:
        return -math.log(x * math.log(b/a))
    return k*math.log(x) + math.log((k+1) / (b**(k+1) - a**(k+1)))

cpdef double trunc_power_ppf(double p, double k, double a, double b) noexcept:
    if (a <= 0 and k <= 0) or b < 0 or a < 0:
        return math.NAN
    if p > 1 or p < 0:
        return math.NAN
    if k == -1:
        return a * (b/a)**p
    k += 1
    return (p * b**k + (1-p) * a**k)**(1./k)

cpdef double trunc_normal_lpdf(double x, double mean, double sigma, double a, double b) noexcept:
    if a >= b:
        return math.NAN
    elif x < a or x > b:
        return -math.INFINITY
    x = (x-mean) / sigma
    a = (a-mean) / sigma
    a = math.erfc(-a/math.sqrt(2.))/2.
    b = (b-mean) / sigma
    b = math.erfc(-b/math.sqrt(2.))/2.
    return -(x-mean)**2/(2*sigma*sigma) - .5*math.log(2*math.M_PI*sigma*sigma) - math.log(b-a)

cpdef double trunc_normal_ppf(double p, double mean, double sigma, double a, double b) noexcept:
    if a >= b:
        return math.NAN
    if p > 1 or p < 0:
        return math.NAN
    a = (a-mean) / sigma
    a = math.erfc(-a/math.sqrt(2.))/2.
    b = (b-mean) / sigma
    b = math.erfc(-b/math.sqrt(2.))/2.
    p = p*(b-a) + a
    return -math.sqrt(2.) * special.erfcinv(2.*p)*sigma + mean

cpdef double trunc_exponential_lpdf(double x, double rate, double a, double b) noexcept:
    if rate <= 0:
        return math.NAN
    if x < a or x > b:
        return -math.INFINITY
    norm = logsumexp(-rate*a, -rate*b, 1., -1.)
    return math.log(rate) - rate*x - norm

cpdef double trunc_exponential_ppf(double p, double rate, double a, double b) noexcept:
    if p > 1 or p < 0 or rate < 0:
        return math.NAN
    a = exponential_cdf(a, rate)
    b = exponential_cdf(b, rate)
    p = a + p * (b-a)
    return -math.log(1 - p) / rate

cpdef double chabrier_lpdf(double log_mass, double log_m_switch, double mean, double sigma, double rate) noexcept:
    '''Initial mass function priors of the form from Chabrier (2002) table 2, as a log PDFs.

    That paper lists the following constant values for different groups of stars:

    ==========================  ============== ========== ======= ======= ==========
    Case                         log_m_switch   mean       sigma   power     rate
    ==========================  ============== ========== ======= ======= ==========
    Disk and Young Clusters          0.0        -1.10237    0.69    1.3    5.295945
    Globular Clusters            -0.04575749    -0.48148    0.34    1.3    5.295945
    Spheroid                     -0.15490195    -0.65757    0.33    1.3    5.295945
    ==========================  ============== ========== ======= ======= ==========

    In order to turn this into a prior on log_mass, the power-law distribution had to be transformed
    into an exponential distribution, hence the use of "rate" rather than "power".  The conversion
    is ``rate = (1+power)*ln(10)``.
    '''
    cdef double cdf_switch = normal_cdf(log_m_switch, mean, sigma)
    cdef double f = normal_lpdf(log_m_switch, mean, sigma)
    cdef double g = trunc_exponential_lpdf(log_m_switch, rate, log_m_switch, 2.)
    cdef double norm = math.exp(f) / math.exp(g)
    if log_mass <= log_m_switch:
        norm = cdf_switch + norm
        return normal_lpdf(log_mass, mean, sigma) - math.log(norm)
    else:
        norm = 1 + cdf_switch/norm
        return trunc_exponential_lpdf(log_mass, rate, log_m_switch, 2.0) - math.log(norm)

cpdef double chabrier_ppf(double p, double log_m_switch, double mean, double sigma, double rate) noexcept:
    '''Initial mass function priors of the form from Chabrier (2002) table 2, as PPFs.

    See :func:`starlord.cy_tools.chabrier_lpdf` for more information.'''
    cdef double cdf_switch = normal_cdf(log_m_switch, mean, sigma)
    if p <= cdf_switch:
        return normal_ppf(p/cdf_switch, mean, sigma)
    else:
        return trunc_exponential_ppf((p-cdf_switch)/(1-cdf_switch), rate, log_m_switch, math.INFINITY)

cpdef void multinormal_zppf(double[:,:] cov_chol, double[:] z, double[:] out, double[:] mean=None):
    '''Calculates a random vector from a normal distribution centered at mean and with the
    cholesky of the covariance cov_chol, given normally-distributed random numbers z, and
    writes the output to out.  If mean is None the mean vector is assumed to be zero.'''
    cdef int i, j
    ndim = cov_chol.shape[0]
    if mean is not None:
        copy_arr1d(mean, out)
    else:
        for i in range(ndim):
            out[i] = 0.
    cdef double temp = 0.
    for i in range(ndim):
        temp = 0.
        for j in range(i+1):
            temp += cov_chol[i, j] * z[j]
        out[i] += temp

include "cy_interpolator.pyx"
include "cy_model.pyx"
include "cy_sampler.pyx"
