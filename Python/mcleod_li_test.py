import numpy as np
from scipy.stats import chi2
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt


def mcleod_li_test(y, gof_lag=None, plot=True):

    def BL(x, lag):
        cor = acf(x, fft=False, nlags=lag)
        n = len(x)
        obs = cor[1:]
        denominators = np.arange(n-1, n-1-lag, step=-1)
        statistic = n * (n + 2) * np.sum(obs**2 / denominators)
        p_value = 1 - chi2.cdf(statistic, df=lag)
        return p_value
    
    residuals = y
    n = len(residuals)
    if gof_lag is None:
        lag_max = int(np.floor(10 * np.log10(n)))
    else:
        lag_max = gof_lag
    lbv = np.empty(lag_max)
    residuals2 = residuals**2
    for i in range(lag_max):
        lbv[i] = BL(residuals2, lag=i+1)
        
    if plot:
        ax = plt.gca()
        kwargs = {'marker': 'o', 'markersize': 5, 'linestyle': None}
        ax.margins(.05)
        ax.plot(np.arange(1, lag_max + 1), lbv, **kwargs)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.05, xmin=0, xmax=lag_max, color='red', linestyle=':')
        ax.set_xlabel('Lag')
        ax.set_ylabel('P-value')
        
    return lbv