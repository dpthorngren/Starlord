import numpy as np
import numba as nb

# Used as a simple check that the data was actually packed by packInterpolator
_magicNumber = -936936.813665


def pack_interpolator(grid, values):
    '''Creates a packed array containing data required for interpolation.'''
    assert len(grid) == values.ndim
    pgrid = [_process_axis_(g, values.shape[i]) for i, g in enumerate(grid)]
    return np.concatenate([
        [_magicNumber, values.ndim],
        [len(xi) for xi in grid],
        [len(xi) for xi in pgrid],
        *pgrid,
        values.flatten(),
    ])


@nb.njit(fastmath=False, cache=True)
def interp1d(data, point):
    '''1-d interpolator (regular or irregular), where data is the output of pack_interpolator'''
    if (data[0] != _magicNumber) or (data[1] != 1.):
        print("Bad input array for interp1d.")
        return np.nan

    # Unpack the data array
    xlen = int(data[2])
    x1len = int(data[3])
    xAxis = data[4:4 + x1len]
    y = data[4 + x1len:]

    i, weight = _locate_point_(point, xAxis, xlen)

    # Bounds check -- points outside bounds are assigned index -1
    if (i < 0):
        return np.nan

    # Sum over bounding points
    return y[i] * (1.-weight) + weight * y[i + 1]


@nb.njit(fastmath=False, cache=True)
def interp2d(data, point0, point1):
    '''2-d interpolator (regular or irregular), where data is the output of pack_interpolator'''
    if (data[0] != _magicNumber) or (data[1] != 2.):
        print("Bad input array for interp2d.")
        return np.nan

    # Unpack the data array
    xlen = int(data[2])
    ylen = int(data[3])
    x1len = int(data[4])
    y1len = int(data[5])

    xAxis = data[6:6 + x1len]
    yAxis = data[6 + x1len:6 + x1len + y1len]
    z = data[6 + x1len + y1len:]

    i, weight0 = _locate_point_(point0, xAxis, xlen)
    j, weight1 = _locate_point_(point1, yAxis, ylen)

    # Bounds check -- points outside bounds are assigned index -1
    if (i < 0 or j < 0):
        return np.nan

    # Sum over bounding points
    result = z[i*ylen + j] * (1.-weight0) * (1.-weight1)
    result += z[(i+1) * ylen + j] * weight0 * (1.-weight1)
    result += z[i*ylen + j + 1] * (1.-weight0) * weight1
    result += z[(i+1) * ylen + j + 1] * weight0 * weight1
    return result


@nb.njit(fastmath=False, cache=True)
def interp3d(data, point0, point1, point2):
    '''3-d interpolator (regular or irregular), where data is the output of pack_interpolator'''
    if (data[0] != _magicNumber) or (data[1] != 3.):
        print("Bad input array for interp3d.")
        return np.nan

    # Unpack the data array
    xlen = int(data[2])
    ylen = int(data[3])
    zlen = int(data[4])
    x1len = int(data[5])
    y1len = int(data[6])
    z1len = int(data[7])

    xAxis = data[8:8 + x1len]
    yAxis = data[8 + x1len:8 + x1len + y1len]
    zAxis = data[8 + x1len + y1len:8 + x1len + y1len + z1len]
    z = data[8 + x1len + y1len + z1len:]

    i, weight0 = _locate_point_(point0, xAxis, xlen)
    j, weight1 = _locate_point_(point1, yAxis, ylen)
    k, weight2 = _locate_point_(point2, zAxis, zlen)

    # Bounds check -- points outside bounds are assigned index -1
    if (i < 0 or j < 0 or k < 0):
        return np.nan

    # Sum over bounding points
    p = (i*ylen + j) * zlen + k
    result = z[p] * (1.-weight0) * (1.-weight1) * (1-weight2)
    result += z[p + 1] * (1.-weight0) * (1.-weight1) * weight2
    p += zlen
    result += z[p] * (1.-weight0) * weight1 * (1-weight2)
    result += z[p + 1] * (1.-weight0) * weight1 * weight2
    p += (ylen-1) * zlen
    result += z[p] * weight0 * (1.-weight1) * (1-weight2)
    result += z[p + 1] * weight0 * (1.-weight1) * weight2
    p += zlen
    result += z[p] * weight0 * weight1 * (1-weight2)
    result += z[p + 1] * weight0 * weight1 * weight2
    return result


@nb.njit(fastmath=False, cache=True)
def interp4d(data, point0, point1, point2, point3):
    '''4-d interpolator (regular or irregular), where data is the output of pack_interpolator'''
    if (data[0] != _magicNumber) or (data[1] != 4.):
        print("Bad input array for interp3d.")
        return np.nan

    # Unpack the data array
    xlen = int(data[2])
    ylen = int(data[3])
    zlen = int(data[4])
    ulen = int(data[5])
    x1len = int(data[6])
    y1len = int(data[7])
    z1len = int(data[8])
    u1len = int(data[9])

    xAxis = data[10:10 + x1len]
    yAxis = data[10 + x1len:10 + x1len + y1len]
    zAxis = data[10 + x1len + y1len:10 + x1len + y1len + z1len]
    uAxis = data[10 + x1len + y1len + z1len:10 + x1len + y1len + z1len + u1len]
    q = data[10 + x1len + y1len + z1len + u1len:]

    i, weight0 = _locate_point_(point0, xAxis, xlen)
    j, weight1 = _locate_point_(point1, yAxis, ylen)
    k, weight2 = _locate_point_(point2, zAxis, zlen)
    l, weight3 = _locate_point_(point3, uAxis, ulen)

    # Bounds check -- points outside bounds are assigned index -1
    if (i < 0 or j < 0 or k < 0 or l < 0):
        return np.nan

    # Sum over bounding points
    p = ((i*ylen + j) * zlen + k) * ulen + l
    result = q[p] * (1.-weight0) * (1.-weight1) * (1-weight2) * (1-weight3)
    result += q[p + 1] * (1.-weight0) * (1.-weight1) * (1-weight2) * weight3
    p += ulen
    result += q[p] * (1.-weight0) * (1.-weight1) * weight2 * (1-weight3)
    result += q[p + 1] * (1.-weight0) * (1.-weight1) * weight2 * weight3
    p += (zlen-1) * ulen
    result += q[p] * (1.-weight0) * weight1 * (1-weight2) * (1-weight3)
    result += q[p + 1] * (1.-weight0) * weight1 * (1-weight2) * weight3
    p += ulen
    result += q[p] * (1.-weight0) * weight1 * weight2 * (1-weight3)
    result += q[p + 1] * (1.-weight0) * weight1 * weight2 * weight3
    p += ((ylen-1) * zlen - 1) * ulen
    result += q[p] * weight0 * (1.-weight1) * (1-weight2) * (1-weight3)
    result += q[p + 1] * weight0 * (1.-weight1) * (1-weight2) * weight3
    p += ulen
    result += q[p] * weight0 * (1.-weight1) * weight2 * (1-weight3)
    result += q[p + 1] * weight0 * (1.-weight1) * weight2 * weight3
    p += (zlen-1) * ulen
    result += q[p] * weight0 * weight1 * (1-weight2) * (1-weight3)
    result += q[p + 1] * weight0 * weight1 * (1-weight2) * weight3
    p += ulen
    result += q[p] * weight0 * weight1 * weight2 * (1-weight3)
    result += q[p + 1] * weight0 * weight1 * weight2 * weight3
    return result


@nb.njit
def _process_axis_(x, length, tol=1e-6):
    '''Processes an interpolation axis into a regularized form for interpolation.

    Specifically, if the axis is non-uniformly spaced, it just verifies that it is sorted and the specified length,
    then returns a copy.  If the axis is uniformly-spaced, it instead returns the three element array
    ([grid start], [grid spacing], 0.). Other interpolation routines in this file will recognize uniform spacing by
    the fact that x[2] < x[1] and act accordingly.'''
    # Consistency checks
    assert x.shape[0] == length
    assert np.all(np.diff(x) > 0)

    # Check for uniformity
    lin = np.linspace(x[0], x[-1], length)
    if np.all(np.absolute(x - lin) <= tol + tol * np.absolute(lin)):
        return np.array([x[0], (length-1.) / (x[-1] - x[0]), 0.])
    return x.copy()


@nb.njit(fastmath=False, inline="always")
def _locate_point_(point, axis, axisLength):
    '''Quickly locates the linear interpolation index and weight across a given axis, which must be formatted
    according to _process_axis_. This allows it to detect if the axis is uniorm and directly calculate the
    index from that.'''
    if not np.isfinite(point):
        return -1, 0.
    weight = 0.
    i = 0
    # Is this grid dimension non-uniform?
    if axis[2] > axis[1]:
        # Check that the point is in bounds
        if point == axis[-1]:
            return axis.shape[0] - 2, 1.
        elif point < axis[0] or point >= axis[-1]:
            return -1, 0.
        # Binary search for the correct indices
        low = 0
        high = axis.shape[0] - 1
        i = (low+high) // 2
        while not (axis[i] <= point < axis[i + 1]):
            i = (low+high) // 2
            if point > axis[i]:
                low = i
            else:
                high = i
        # Calculate the the index and weight
        weight = (point - axis[i]) / (axis[i + 1] - axis[i])
    else:
        # Check that the point is in bounds
        xMax = (axis[0] + (axisLength-1) / axis[1])
        if point == xMax:
            return axisLength - 2, 1.
        if point < axis[0] or point >= xMax:
            return -1, 0.
        # Calculate the the index and weight
        weight = (point - axis[0]) * axis[1]
        i = int(weight)
        weight -= i
    return i, weight
