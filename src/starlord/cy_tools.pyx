
cpdef double uniform_lpdf(double x, double xmin, double xmax):
    if x > xmin and x < xmax:
        return -math.log(xmax - xmin)
    return -math.INFINITY

cpdef double uniform_ppf(double x, double xmin, double xmax):
    return xmin + x * (xmax - xmin)

cpdef double normal_lpdf(double x, double mean, double sigma):
    if sigma <= 0:
        return math.NAN
    return -(x-mean)**2/(2*sigma*sigma) - .5*math.log(2*math.M_PI*sigma*sigma)

cpdef double normal_ppf(double p, double mean, double sigma):
    return -math.sqrt(2.) * special.erfcinv(2.*p)*sigma + mean

cpdef double beta_lpdf(double x, double alpha, double beta):
    return (alpha-1.)*math.log(x) + (beta-1.)*math.log(1-x) - special.betaln(alpha, beta)

cpdef double beta_ppf(double p, double alpha, double beta):
    return special.betaincinv(alpha, beta, p)

cpdef double gamma_lpdf(double x, double alpha, double lamb):
    return (alpha-1.)*math.log(x*lamb) + math.log(lamb) - lamb*x - special.gammaln(alpha)

cpdef double gamma_ppf(double p, double alpha, double lamb):
    return special.gammaincinv(alpha, p)/lamb
    
