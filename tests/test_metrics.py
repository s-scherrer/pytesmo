import numpy as np
import numpy.testing as nptest
import pytest

import pytesmo.metrics
from pytesmo.metrics import *


@pytest.fixture
def testdata():
    # the seed avoids random failure
    np.random.seed(0)

    # generate random data that is correlated with r = 0.8
    cov = np.array([[1, 0.8], [0.8, 1]])
    X = np.linalg.cholesky(cov) @ np.random.randn(2, 1000)
    x, y = X[0, :], X[1, :]
    y = 1.1 * y + 0.5
    return x, y


has_ci = [
    "bias",
    "msd",
    "rmsd",
    "nrmsd",
    "ubrmsd",
    "msd_bias",
    "pearson_r",
    "spearman_r",
    "kendall_tau",
]

no_ci = [
    "aad",
    "mad",
    "msd_corr",
    "msd_var",
    "nash_sutcliffe",
    "index_of_agreement",
]


def test_analytical_ci_availability(testdata):
    for funcname in has_ci:
        func = getattr(pytesmo.metrics, funcname)
        assert has_analytical_ci(func)

    for funcname in no_ci:
        func = getattr(pytesmo.metrics, funcname)
        assert not has_analytical_ci(func)


def test_analytical_cis(testdata):
    x, y = testdata
    for funcname in has_ci:
        func = getattr(pytesmo.metrics, funcname)
        m, lb, ub = with_analytical_ci(func, x, y)
        m10, lb10, ub10 = with_analytical_ci(func, x, y, alpha=0.1)
        assert m == m10
        assert lb < m
        assert m < ub
        assert lb < lb10
        assert ub > ub10


def test_bootstrapped_cis(testdata):
    x, y = testdata
    for funcname in has_ci:
        func = getattr(pytesmo.metrics, funcname)
        m, lb, ub = with_analytical_ci(func, x, y, alpha=0.1)
        m_bs, lb_bs, ub_bs = with_bootstrapped_ci(
            func, x, y, alpha=0.1, nsamples=1000
        )
        assert m == m_bs
        assert lb_bs < ub_bs
        if funcname != "nrmsd":
            # nrmsd is a bit unstable when bootstrapping, due to the data
            # dependent normalization that is applied
            assert abs(ub - ub_bs) < 1e-2
            assert abs(lb - lb_bs) < 1e-2
        else:
            assert abs(ub - ub_bs) < 1e-1
            assert abs(lb - lb_bs) < 1e-1
    for funcname in no_ci:
        m, lb, ub = with_bootstrapped_ci(func, x, y, alpha=0.1, nsamples=1000)
        assert lb < m
        assert m < ub


# expected values of the metrics, to test whether it works
def test_expected_values(testdata):
    x, y = testdata

    expected_msd_bias = 0.5 ** 2
    expected_msd_var = 0.1 ** 2
    expected_msd_corr = 2 * 1 * 1.1 * (1 - 0.8)
    expected_msd = expected_msd_bias + expected_msd_corr + expected_msd_var
    expected_values = {
        "bias": -0.5,
        "msd": expected_msd,
        "rmsd": np.sqrt(expected_msd),  # actually not totally true
        "ubrmsd": np.sqrt(expected_msd_corr + expected_msd_var),  # not quite
        "msd_corr": expected_msd_corr,
        "msd_bias": expected_msd_bias,
        "msd_var": expected_msd_var,
        "pearson_r": 0.8,
    }

    for metric in expected_values:
        func = getattr(pytesmo.metrics, metric)
        m, lb, ub = with_bootstrapped_ci(func, x, y)
        e = expected_values[metric]
        assert lb < e
        assert e < ub


def test_tcol_metrics():
    n = 1000000

    mean_signal = 0.3
    sig_signal = 0.2
    signal = np.random.normal(mean_signal, sig_signal, n)

    sig_err_x = 0.02
    sig_err_y = 0.07
    sig_err_z = 0.04
    err_x = np.random.normal(0, sig_err_x, n)
    err_y = np.random.normal(0, sig_err_y, n)
    err_z = np.random.normal(0, sig_err_z, n)

    alpha_y = 0.2
    alpha_z = 0.5

    beta_y = 0.9
    beta_z = 1.6

    x = signal + err_x
    y = alpha_y + beta_y * (signal + err_y)
    z = alpha_z + beta_z * (signal + err_z)

    beta_pred = 1.0 / np.array((1, beta_y, beta_z))
    err_pred = np.array((sig_err_x, sig_err_y, sig_err_z))
    snr_pred = np.array(
        (
            (sig_signal / sig_err_x),
            (sig_signal / sig_err_y),
            (sig_signal / sig_err_z),
        )
    )

    snr, err, beta = tcol_metrics(x, y, z, ref_ind=0)

    nptest.assert_almost_equal(beta, beta_pred, decimal=2)
    nptest.assert_almost_equal(err, err_pred, decimal=2)
    nptest.assert_almost_equal(
        np.sqrt(10 ** (snr / 10.0)), snr_pred, decimal=1
    )


def test_bias():
    """
    Test for bias
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2

    b_pred = -2
    b_obs = bias(x, y)

    nptest.assert_equal(b_obs, b_pred)

    # example 2
    x = np.arange(10)
    y = np.arange(20, 30)

    b_pred = 20.
    b_obs = bias(y, x)

    nptest.assert_equal(b_obs, b_pred)


def test_aad():
    """
    Test for average absolute deviation
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2
    dev_pred = 2.
    dev_obs = aad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 201.
    dev_pred = 21.
    dev_obs = aad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)


def test_mad():
    """
    Test for median absolute deviation
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2
    dev_pred = 2.
    dev_obs = mad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 201.
    dev_pred = 2.
    dev_obs = mad(x, y)

    nptest.assert_equal(dev_obs, dev_pred)


def test_rmsd():
    """
    Test for rmsd
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2

    rmsd_pred = 2.
    rmsd_obs = rmsd(x, y)

    nptest.assert_equal(rmsd_obs, rmsd_pred)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 100.

    rmsd_pred = np.sqrt(831.7)
    rmsd_obs = rmsd(x, y)

    nptest.assert_almost_equal(rmsd_obs, rmsd_pred, 6)


def test_ubrmsd():
    """
    Test for ubrmsd
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2

    ubrmsd_pred = 0
    ubrmsd_obs = ubrmsd(x, y)

    nptest.assert_equal(ubrmsd_obs, ubrmsd_pred)
    # aslo check consistency with direct formula
    ubrmsd_direct = np.sqrt(rmsd(x, y) ** 2 - bias(x, y)**2)
    nptest.assert_equal(ubrmsd_obs, ubrmsd_direct)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 100.

    ubrmsd_pred = 26.7
    ubrmsd_obs = ubrmsd(x, y)

    nptest.assert_almost_equal(ubrmsd_obs, ubrmsd_pred, 6)
    # aslo check consistency with direct formula
    ubrmsd_direct = np.sqrt(rmsd(x, y) ** 2 - bias(x, y)**2)
    nptest.assert_almost_equal(ubrmsd_obs, ubrmsd_direct)


def test_msd():
    """
    Test for msd
    """
    # example 1
    x = np.arange(10)
    y = np.arange(10) + 2

    msd_pred = 4.
    msd_bias_pred = 2. ** 2
    msd_obs = msd(x, y)
    msd_bias_obs = msd_bias(x, y)

    nptest.assert_equal(msd_obs, msd_pred)
    nptest.assert_equal(msd_bias_obs, msd_bias_pred)

    # example 2, with outlier
    x = np.arange(10)
    y = np.arange(10) + 2
    y[-1] = 51.

    msd_pred = 180.
    msd_bias_pred = 36.
    msd_obs = msd(x, y)
    msd_bias_obs = msd_bias(x, y)

    nptest.assert_almost_equal(msd_obs, msd_pred, 6)
    nptest.assert_almost_equal(msd_bias_obs, msd_bias_pred, 6)
