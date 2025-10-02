
cpdef double normal_lpdf(double x, double mean, double sigma):
    if sigma <= 0:
        return math.NAN
    return -(x-mean)**2/(2*sigma*sigma) - .5*math.log(2*math.M_PI*sigma*sigma)
