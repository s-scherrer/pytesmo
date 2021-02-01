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
    bias = bias_func(x, y)
    delta = (
        np.std(x - y, ddof=1)
        / np.sqrt(n) * stats.t.ppf(1-0.025, n)
    )
    return bias - delta, bias + delta


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
    return np.sqrt(lb_msd), np.sqrt(ub_msd)


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
    n = len(x) - ddof
    ubMSD = ubrmsd_func(x, y, ddof)**2
    lb_ubMSD = n * ubMSD / stats.chi2.ppf(alpha/2, n)
    ub_ubMSD = n * ubMSD / stats.chi2.ppf(1 - alpha/2, n)
    return np.sqrt(lb_ubMSD), np.sqrt(ub_ubMSD)


def mse_corr_func(x, y):
    r"""
    Correlation component of MSE.

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::
        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
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
    mse_corr : float
        Correlation component of MSE.
    """
    return 2 * np.std(x) * np.std(y) * (1 - pearsonr_func(x, y))


def mse_var_func(x, y):
    r"""
    Variance component of MSE.

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::
        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSE_{var} = (\sigma_x -
    \sigma_y)^2`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    mse_var : float
        Variance component of MSE.
    """
    return (np.std(x) - np.std(y)) ** 2


def mse_bias_func(x, y):
    r"""
    Bias component of MSE.

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::
        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the term :math:`MSE_{bias} = (\mu_x - \mu_y)^2`.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    mse_bias : float
        Bias component of MSE.
    """
    return bias_func(x, y) ** 2


def mse_bias_ci(x, y, alpha=0.05):
    """
    Confidence interval for :math:`MSE_{bias}`

    See also :func:`pytesmo._metric_funcs.mse_bias_func`.
    """
    # we can get this by calculating the CI for
    lb_delta, ub_delta = bias_ci(x, y, alpha)
    return lb_delta ** 2, ub_delta ** 2


def mse_func(x, y):
    r"""
    Mean squared error.

    For validation, MSE is defined as

    ..math::
        MSE = \frac{1}{n}\sum\limits_{i=1}^n (x_i - y_i)^2

    MSE can be decomposed into a term describing the deviation of x and y
    attributable to non-perfect correlation (r < 1), a term depending on the
    difference in variances between x and y, and the difference in means
    between x and y (bias).

    ..math::
        MSE &= MSE_{corr} + MSE_{var} + MSE_{bias}\\
            &= 2\sigma_x\sigma_y (1-r) + (\sigma_x - \sigma_y)^2
               + (\mu_x - \mu_y)^2

    This function calculates the full MSE, the function `mse_corr`, `mse_var`,
    and `mse_bias` can be used to calculate the individual components.
    """
    return np.mean((x - y)**2)


def mse_ci(x, y, alpha=0.05):
    """
    Confidence interval for MSE.
    """
    n = len(x)
    mse = mse_func(x, y)
    lb_mse = n * mse / stats.chi2.ppf(alpha/2, n)
    ub_mse = n * mse / stats.chi2.ppf(1 - alpha/2, n)
    return np.sqrt(lb_mse), np.sqrt(ub_mse)


def pearsonr_func(x, y):
    """
    Wrapper for scipy.stats.pearsonr.

    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.

    Parameters
    ----------
    x : numpy.ndarray
        First input vector.
    y : numpy.ndarray
        Second input vector.

    Returns
    -------
    r : float
        Pearson's correlation coefficent.

    See Also
    --------
    scipy.stats.pearsonr
    """
    return stats.pearsonr(x, y)[0]


def pearsonr_ci(x, y, alpha=0.05):
    """
    Confidence interval for Pearson correlation coefficient.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    r = pearsonr_func(x, y)
    v = np.arctanh(r)
    z = stats.norm.ppf(alpha/2)
    cl = v - z/np.sqrt(n - 3)
    cu = v + z/np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)


def spearmanr_func(x, y):
    """
    Wrapper for scipy.stats.spearmanr. Calculates a Spearman
    rank-order correlation coefficient and the p-value to
    test for non-correlation.

    Parameters
    ----------
    x : numpy.array
        First input vector.
    y : numpy.array
        Second input vector.

    Returns
    -------
    rho : float
        Spearman correlation coefficient

    See Also
    --------
    scipy.stats.spearmenr
    """
    return stats.spearmanr(x, y)[0]


def spearmanr_ci(x, y, alpha=0.05):
    """
    Confidence interval for Spearman rank correlation coefficient.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    r = spearmanr_func(x, y)
    v = np.arctanh(r)
    z = stats.norm.ppf(alpha/2)
    # see reference for this formula
    cl = v - z * np.sqrt(1 + r ** 2 / 2) / np.sqrt(n - 3)
    cu = v - z * np.sqrt(1 + r ** 2 / 2) / np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)


def kendalltau_func(x, y):
    """
    Wrapper for scipy.stats.kendalltau

    Parameters
    ----------
    x : numpy.array
        First input vector.
    y : numpy.array
        Second input vector.

    Returns
    -------
    Kendall's tau : float
        The tau statistic

    See Also
    --------
    scipy.stats.kendalltau
    """
    return stats.kendalltau(x, y)[0]


def kendalltau_ci(x, y, alpha=0.05):
    r"""
    Confidence intervall for Kendall's rank coefficient.

    References
    ----------
    Bonett, D. G., & Wright, T. A. (2000). Sample size requirements for
    estimating Pearson, Kendall and Spearman correlations. Psychometrika,
    65(1), 23-28.
    """
    n = len(x)
    r = kendalltau_func(x, y)
    v = np.arctanh(r)
    z = stats.norm.ppf(alpha/2)
    # see reference for this formula
    cl = v - z * 0.431 / np.sqrt(n - 3)
    cu = v + z * 0.431 / np.sqrt(n - 3)
    return np.tanh(cl), np.tanh(cu)


def index_of_agreement_func(o, p):
    """
    Index of agreement was proposed by Willmot (1981), to overcome the
    insenstivity of Nash-Sutcliffe efficiency E and R^2 to differences in the
    observed and predicted means and variances (Legates and McCabe, 1999).
    The index of agreement represents the ratio of the mean square error and
    the potential error (Willmot, 1984). The potential error in the denominator
    represents the largest value that the squared difference of each pair can
    attain. The range of d is similar to that of R^2 and lies between
    0 (no correlation) and 1 (perfect fit).

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    d : float
        Index of agreement.
    """
    denom = np.sum((np.abs(p - np.mean(o)) + np.abs(o - np.mean(o)))**2)
    d = 1 - RSS(o, p) / denom

    return d


def nash_sutcliffe_func(o, p):
    """
    Nash Sutcliffe model efficiency coefficient E. The range of E lies between
    1.0 (perfect fit) and -inf.

    Parameters
    ----------
    o : numpy.ndarray
        Observations.
    p : numpy.ndarray
        Predictions.

    Returns
    -------
    E : float
        Nash Sutcliffe model efficiency coefficient E.
    """
    return 1 - (np.sum((o - p) ** 2)) / (np.sum((o - np.mean(o)) ** 2))
