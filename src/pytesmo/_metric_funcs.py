# Copyright (c) 2021, Vienna University of Technology,
# Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the Vienna University of Technology,
#     Department of Geodesy and Geoinformation nor the
#     names of its contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL VIENNA UNIVERSITY OF TECHNOLOGY,
# DEPARTMENT OF GEODESY AND GEOINFORMATION BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Metric functions to be used in pytesmo.metrics

Formulas for analytical CIs are taken from Gilleland 2010, 10.5065/D6WD3XJM
https://opensky.ucar.edu/islandora/object/technotes:491
"""

from itertools import permutations, combinations
import numpy as np
from scipy import stats


def bias_func(x, y):
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
    return np.mean(x) - np.mean(y)


def bias_ci(x, y):
    """
    Confidence interval for bias.
    """
    n = len(x)
    delta = (
        np.std(x - y, ddof=1)
        / np.sqrt(n) * stats.t.ppf(1-0.025, n)
    )
    return -delta, delta


def aad_func(x, y):
    """
    Average (=mean) absolute deviation (AAD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    d : float
        Mean absolute deviation.
    """
    return np.mean(np.abs(x - y))


def mad_func(x, y):
    """
    Median absolute deviation (MAD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    d : float
        Median absolute deviation.
    """
    return np.median(np.abs(x - y))


def RSS(o, p):
    """
    Residual sum of squares.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    res : float
        Residual sum of squares.
    """
    return np.sum((o - p) ** 2)


def rmsd_func(x, y, ddof=0):
    """
    Root-mean-square deviation (RMSD).

    It is implemented for an unbiased estimator, which means the RMSD is the
    square root of the variance, also known as the standard error. The delta
    degree of freedom keyword (ddof) can be used to correct for the case the
    true variance is unknown and estimated from the population. Concretely, the
    naive sample variance estimator sums the squared deviations and divides by
    n, which is biased. Dividing instead by n -1 yields an unbiased estimator

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation.
    """
    return np.sqrt(RSS(x, y) / (len(x) - ddof))


def rmsd_ci(x, y, alpha=0.05, ddof=0):
    """
    Confidence interval for RMSD.

    This is calculated using standard results for CIs for the sample variance.
    """
    n = len(x) - ddof
    msd = RSS(x, y) / n
    lb_msd = n * msd / stats.chi2.ppf(alpha/2, n)
    ub_msd = n * msd / stats.chi2.ppf(1 - alpha/2, n)
    return np.sqrt(lb_msd,), np.sqrt(ub_msd)


def nrmsd_func(x, y, ddof=0):
    """
    Normalized root-mean-square deviation (nRMSD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    nrmsd : float
        Normalized root-mean-square deviation (nRMSD).
    """
    return rmsd_func(x, y, ddof) / (np.max([x, y]) - np.min([x, y]))


def nrmsd_ci(x, y, alpha=0.05, ddof=0):
    """
    Confidence interval for normalized RMSD.
    """
    c = np.max([x, y]) - np.min([x, y])
    lb, ub = rmsd_ci(x, y, alpha, ddof)
    return lb/c, ub/c


def ubrmsd_func(x, y, ddof=0):
    """
    Unbiased root-mean-square deviation (uRMSD).

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.
    ddof : int, optional
        Delta degree of freedom.The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.

    Returns
    -------
    urmsd : float
        Unbiased root-mean-square deviation (uRMSD).
    ci : 2-tuple of float
        ``(dl, du)`` so that ``(bias-dl, bias+du)`` is the 95% confidence
        interval.
    """
    return np.sqrt(np.sum(((x - np.mean(x)) -
                           (y - np.mean(y))) ** 2) / (len(x) - ddof))


def ubrmsd_ci(x, y, alpha=0.05, ddof=0):
    """
    Confidende interval for unbiased root-mean-square deviation (uRMSD).
    """
    ubMSD = ubrmsd_func(x, y, ddof)**2
    lb_ubMSD = n * ubMSD / stats.chi2.ppf(alpha/2, n)
    ub_ubMSD = n * ubMSD / stats.chi2.ppf(1 - alpha/2, n)
    return np.sqrt(lb_ubMSD,), np.sqrt(ub_ubMSD)
