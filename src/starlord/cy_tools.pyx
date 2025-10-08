import numpy as np

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
    
cdef class GridInterpolator:

    def __init__(self, axes, values, tol=1e-6):
        self.ndim = len(axes)
        assert self.ndim <= 5
        # Setup data array (axes, values)
        processed = []
        for i, ax in enumerate(axes):
            assert np.all(np.diff(ax) > 0.)
            lin = np.linspace(ax[0], ax[-1], len(ax))
            if np.all(np.absolute(ax - lin) <= tol + tol * np.absolute(lin)):
                processed.append(np.array([ax[0], (len(ax)-1.) / (ax[-1] - ax[0]), 0.]))
            else:
                processed.append(ax)
        processed.append(values.flatten())
        self._data = np.concatenate(processed)
        # Fill in additional data based on dimension
        self.y_len = 1
        self.z_len = 1
        self.u_len = 1
        self.v_len = 1
        start, stop = 0, len(processed[0])
        self.x_len = len(axes[0])
        self.x_axis = self._data[start:stop]
        if self.ndim > 1:
            start, stop = stop, stop+len(processed[1])
            self.y_len = len(axes[1])
            self.y_axis = self._data[start:stop]
        if self.ndim > 2:
            start, stop = stop, stop+len(processed[2])
            self.z_len = len(axes[2])
            self.z_axis = self._data[start:stop]
        if self.ndim > 3:
            start, stop = stop, stop+len(processed[3])
            self.u_len = len(axes[3])
            self.u_axis = self._data[start:stop]
        if self.ndim > 4:
            start, stop = stop, stop+len(processed[4])
            self.v_len = len(axes[4])
            self.v_axis = self._data[start:stop]
        self.v_stride = 1
        self.u_stride = self.v_stride * self.v_len
        self.z_stride = self.u_stride * self.u_len
        self.y_stride = self.z_stride * self.z_len
        self.x_stride = self.y_stride * self.y_len
        self.values = self._data[stop:]

    cpdef double interp(self, double x):
        return x

    cpdef double _interp1d(self, double point):
        cdef int xi
        cdef double xw
        # Locate on grid and bounds check
        xi = _locatePoint_(point, self.x_axis, self.x_, &xw)
        if(xi < 0):
            return math.NAN
        # Weighted sum over bounding points
        return self.values[xi]*(1.-xw) + xw * self.values[xi+1]

    cpdef double _interp2d(self, double x, double y):
        cdef int xi, yi
        cdef double xw, yw
        # Locate on grid and bounds check
        xi = _locatePoint_(x, self.x_axis, self.x_len, &xw)
        yi = _locatePoint_(y, self.y_axis, self.y_len, &yw)
        if (xi < 0) or (yi < 0):
            return math.NAN
        # Weighted sum over bounding points
        cdef int s = xi*self.x_stride + yi*self.y_stride
        cdef double out = ((
                self.values[s]*(1.-yw) +
                self.values[s+self.y_stride]*yw
            )*(1.-xw) + (
                self.values[s+self.x_stride]*(1.-yw) +
                self.values[s+self.y_stride+self.x_stride]*yw
            )*xw
        )
        return out

cdef inline int _locatePoint_(double point, double[:] axis, int ax_, double* w):
    if not math.isfinite(point):
        return -1
    cdef int i = 0
    cdef int low = 0
    cdef int high = axis.shape[0]-1
    cdef double weight = 0.
    # Is this grid dimension non-uniform?
    if axis[2] > axis[1]:
        # Check that the point is in bounds
        if point == axis[-1]:
            w[0] = 1.
            return high
        if point < axis[0] or point > axis[-1]:
            return -1
        # Binary search for the correct indices
        i = (low+high) // 2
        while not (axis[i] <= point < axis[i+1]):
            i = (low+high) // 2
            if point > axis[i]:
                low = i
            else:
                high = i
        # Calculate the the index and weight
        weight = (point - axis[i]) / (axis[i+1] - axis[i])
    else:
        # Check that the point is in bounds
        if point == (axis[0] + (ax_-1)/axis[1]):
            w[0] = 1.
            return high
        if point < axis[0] or point >= (axis[0] + (ax_-1)/axis[1]):
            return -1
        # Calculate the the index and weight
        weight = (point-axis[0]) * axis[1]
        i = int(weight)
        weight -= i
    w[0] = weight
    return i
