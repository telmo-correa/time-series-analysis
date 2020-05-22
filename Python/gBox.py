import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf
from scipy.signal import lfilter, lfiltic
from scipy.stats import chi2

import matplotlib.pyplot as plt


def gBox(fitted_model, lags=None, x=None, method="squared", plot=True, **kwargs):
    """ Adapted from R's TSA library function gBox, but to be used with
    models returned from Python's arch library. """
    
    def zlag(x, d=1):
        return np.array(pd.Series(x).shift(d))

    def kurtosis(x, na_rm=False):
        if na_rm:
            x = x[~np.isnan(x)]
        return np.sum((x - np.mean(x))**4) / (len(x) * np.var(x)**2) - 3

    def gBox_test(fitted_model, lag, x):
        p = fitted_model.model.volatility.p
        q = fitted_model.model.volatility.q
        beta = np.array([fitted_model.params['beta[' + str(i + 1) +']'] for i in range(q)])
        #x = fitted_model.model.y
        epsilon = zlag(fitted_model.resid, d=max(p, q))
        M = np.empty((len(x), p+q))
        for i in range(q):
            M[:, i] = zlag(x**2, i+1)
        h = zlag(fitted_model.conditional_volatility, max(p, q))**2
        for i in range(p):
            M[:, q + i] = zlag(h, i+1)
        sigma2 = np.mean(x**2)
        np.nan_to_num(M, nan=sigma2, copy=False)

        if p > 0:
            a, b = np.r_[1, -beta], np.r_[1.0]
            zi = lfiltic(b=b, a=a, y=np.ones(len(beta)) * sigma2)
            def beta_filter(x):
                y, zf = lfilter(b=b, a=a, x=x, zi=zi)
                return y

            M_swap = np.apply_along_axis(func1d=beta_filter, axis=0, arr=M)
            M = np.empty((len(x), p+q+1))
            M[:, 0] = 1 / (1 - np.sum(beta))
            M[:, 1:] = M_swap
        else:
            M_swap = M.copy()
            M = np.empty((len(x), p+q+1))
            M[:, 0] = 1
            M[:, 1:] = M_swap

        M = np.apply_along_axis(func1d=lambda c : c / h, axis=0, arr=M)
        np.nan_to_num(M, nan=0, copy=False)

        E = np.empty((lag, len(x)))
        e2 = epsilon**2
        e2_mean = np.nanmean(e2)
        for i in range(lag):
            E[i, :] = zlag(e2, i+1) - e2_mean
        np.nan_to_num(E, nan=0, copy=False)

        J = (E @ M)/len(h)
        k = kurtosis(epsilon, na_rm=True)
        Lambda = 2 * np.linalg.inv(M.T @ M)
        H = -J @ Lambda @ J.T/(2 * k) * len(h)
        for i in range(lag):
            H[i, i] += 1
        acf1 = np.resize(acf(e2, nlags=lag, fft=False, missing='drop')[1:], (lag, 1))
        stat = (acf1.T @ np.linalg.inv(H) @ acf1 * len(h))[0, 0]
        p_value = 1 - chi2.cdf(x=stat, df=lag)

        return {
            'p_value': p_value,
            'H': H,
            'Lambda': Lambda,
            'J': J,
            'stat': stat,
            'acf': acf1,
            'n': len(h)
        }

    def gBox1_test(fitted_model, lag, x):
        p = fitted_model.model.volatility.p
        q = fitted_model.model.volatility.q
        beta = np.array([fitted_model.params['beta[' + str(i + 1) +']'] for i in range(q)])
        #x = model.model.y
        epsilon = zlag(fitted_model.resid, d=max(p, q))
        M = np.empty((len(x), p+q))
        for i in range(q):
            M[:, i] = zlag(x**2, i+1)
        h = zlag(fitted_model.conditional_volatility, max(p, q))**2
        for i in range(p):
            M[:, q + i] = zlag(h, i+1)
        sigma2 = np.mean(x**2)
        np.nan_to_num(M, nan=sigma2, copy=False)

        if p > 0:
            a, b = np.r_[1, -beta], np.r_[1.0]
            zi = lfiltic(b=b, a=a, y=np.ones(len(beta)) * sigma2)
            def beta_filter(x):
                y, zf = lfilter(b=b, a=a, x=x, zi=zi)
                return y

            M_swap = np.apply_along_axis(func1d=beta_filter, axis=0, arr=M)
            M = np.empty((len(x), p+q+1))
            M[:, 0] = 1 / (1 - np.sum(beta))
            M[:, 1:] = M_swap
        else:
            M_swap = M.copy()
            M = np.empty((len(x), p+q+1))
            M[:, 0] = 1
            M[:, 1:] = M_swap

        M = np.apply_along_axis(func1d=lambda c : c / h, axis=0, arr=M)
        np.nan_to_num(M, nan=0, copy=False)

        e_abs = np.abs(epsilon)
        tau = np.nanmean(e_abs)
        nu = np.nanmean(e_abs**3)

        E = np.empty((lag, len(x)))

        for i in range(lag):
            E[i, :] = zlag(e_abs, i+1) - tau
        np.nan_to_num(E, nan=0, copy=False)

        J = (E @ M)/len(h)
        k = kurtosis(epsilon, na_rm=True)
        Lambda = 2 * np.linalg.inv(M.T @ M)

        H = (J @ Lambda @ J.T) * len(h) * (k/2 * tau**2/4 - tau * (nu - tau)/2)/(1 - tau**2)**2

        for i in range(lag):
            H[i, i] += 1
        acf1 = np.resize(acf(e_abs, nlags=lag, fft=False, missing='drop')[1:], (lag, 1))
        stat = (acf1.T @ np.linalg.inv(H) @ acf1 * len(h))[0, 0]
        p_value = 1 - chi2.cdf(x=stat, df=lag)

        return {
            'p_value': p_value,
            'H': H,
            'Lambda': Lambda,
            'J': J,
            'stat': stat,
            'acf': acf1,
            'n': len(h)
        }
    
    
    if x is None:
        x = fitted_model.model.y
    if lags is None:
        lags = np.arange(1, 21)
    
    get_p_value = None
    if method == "squared":
        get_p_value = lambda lag : gBox_test(fitted_model, lag, x)['p_value']
    elif method == "absolute":
        get_p_value = lambda lag : gBox1_test(fitted_model, lag, x)['p_value']
    else:
        raise "Unknown method: " + str(method)
    
    results = np.empty(len(lags))
    for i, lag in enumerate(lags):
        results[i] = get_p_value(lag)
    
    if plot:
        ax = plt.gca()
        kwargs = {'marker': 'o', 'markersize': 5, 'linestyle': None}
        ax.margins(.05)
        ax.plot(lags, results, **kwargs)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.05, xmin=np.min(lags)-1, xmax=np.max(lags)+1, color='red', linestyle=':')
        ax.set_xlabel('Lag')
        ax.set_ylabel('P-value')
    
    return {
        'lags': lags,
        'results': results
    }