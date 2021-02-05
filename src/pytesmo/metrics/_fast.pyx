cimport numpy as np
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef bias(np.float_t [:] x, np.float_t [:] y):
    """
    Difference of the mean values.

    Sign of output depends on argument order. We calculate mean(x) - mean(y).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    bias : float
        Bias between x and y.
    """
    cdef np.float_t b = 0
    cdef int i, n = len(x)
    for i in range(n):
        b += x[i] - y[i]
    return b / n

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef RSS(np.float_t [:] x, np.float_t [:] y):
    """
    Residual sum of squares.

    Parameters
    ----------
    x : numpy.ndarray
        Observations.
    y : numpy.ndarray
        Predictions.

    Returns
    -------
    res : float
        Residual sum of squares.
    """
    cdef np.float_t sum = 0
    cdef int i
    cdef int n = len(x)
    for i in range(n):
        sum += (x[i] - y[i])**2
    return sum


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef msd_corr(np.float_t [:] x, np.float_t [:] y):
    r"""
    Correlation component of MSD.

    MSD can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSD &= MSD_{corr} + MSD_{var} + MSD_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSE_{corr} =
    2\sigma_x\sigma_y(1-r)`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    msd_corr : float
        Correlation component of MSE.
    """
    cdef np.float_t mx = 0, my = 0
    cdef np.float_t varx = 0, vary = 0, cov = 0
    cdef int i, n = len(x)
    
    # calculate means
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    
    # calculate variances and covariance
    for i in range(n):
        varx += (x[i] - mx)**2
        vary += (y[i] - my)**2
        cov += (x[i] - mx) * (y[i] - my)
    varx /= n
    vary /= n
    cov /= n
    return 2 * sqrt(varx) * sqrt(vary) - 2 * cov


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef msd_var(np.float_t [:] x, np.float_t [:] y):
    r"""
    Variance component of MSD.

    MSD can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSD &= MSD_{corr} + MSD_{var} + MSD_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSD_{var} = (\sigma_x -
    \sigma_y)^2`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    msd_var : float
        Variance component of MSD.
    """
    cdef np.float_t mx = 0, my = 0
    cdef np.float_t varx = 0, vary = 0, cov = 0
    cdef int i, n = len(x)
    
    # calculate means
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    
    # calculate variance
    for i in range(n):
        varx += (x[i] - mx)**2
        vary += (y[i] - my)**2
    varx /= n
    vary /= n
    return (sqrt(varx) - sqrt(vary)) ** 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef msd_bias(np.float_t [:] x, np.float_t [:] y):
    r"""
    Bias component of MSD.

    MSD can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSD &= MSD_{corr} + MSD_{var} + MSD_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSD_{bias} = (\mu_x - \mu_y)^2`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    msd_bias : float
        Bias component of MSD.
    """
    return bias(x, y) ** 2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef msd_decomposition(np.float_t [:] x, np.float_t [:] y):
    r"""
    Mean square deviation/mean square error.

    For validation, MSD (same as MSE) is defined as

    ..math::

        MSD = \frac{1}{n}\sum\limits_{i=1}^n (x_i - y_i)^2

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::

        MSD &= MSD_{corr} + MSD_{var} + MSD_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the all components as well as the sum.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector
    y : numpy.ndarray
        Second input vector

    Returns
    -------
    msd : float
        Mean square deviation
    msd_corr : float
        Correlation component of MSD.
    msd_bias : float
        Bias component of the MSD.
    msd_var : float
        Variance component of the MSD.
    """
    cdef np.float_t mx = 0, my = 0
    cdef np.float_t varx = 0, vary = 0, cov = 0
    cdef np.float_t msd, msd_corr, msd_var, msd_bias
    cdef int i, n = len(x)
    
    # calculate means
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    
    # calculate variances and covariance
    for i in range(n):
        varx += (x[i] - mx)**2
        vary += (y[i] - my)**2
        cov += (x[i] - mx) * (y[i] - my)
    varx /= n
    vary /= n
    cov /= n

    # decompositions
    msd_corr =  2 * sqrt(varx) * sqrt(vary) - 2 * cov
    msd_var = (sqrt(varx) - sqrt(vary)) ** 2
    msd_bias = (mx - my) ** 2
    msd = msd_corr + msd_var + msd_bias
    return msd, msd_corr, msd_bias, msd_var
