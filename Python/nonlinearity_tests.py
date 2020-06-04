import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.anova import anova_lm
from scipy import stats

import warnings


def _get_ar_order(x):
    N = len(x)
    order_max = min(N - 1, int(np.floor(10 * np.log10(N))))

    # Use Yule-Walker to find best AR model via AIC
    def aic(sigma2, df_model, nobs):
        return np.log(sigma2) + 2 * (1 + df_model) / nobs

    best_results = None

    for lag in range(order_max+1):
        ar, sigma = yule_walker(x, order=lag, method='mle')
        model_aic = aic(sigma2=sigma**2, df_model=lag, nobs=N-lag)
        if best_results is None or model_aic < best_results['aic']:
            best_results = {
                'aic': model_aic,
                'order': lag,
                'ar': ar,
                'sigma2': sigma**2
            }

    return best_results['order']


def Keenan_test(x, order=None):
    
    def get_lm(y, x, m):
        assert len(x) == len(x), "x and y must have same length"

        X = np.nan * np.ones((len(x), m+1))
        X[:, 0] = y
        for i in range(1, m+1):
            X[i:, i] = x[:-i]

        # Omit NaNs
        X = X[~np.any(np.isnan(X), axis=1)]

        return sm.OLS(X[:, 0], X[:, 1:]).fit(method='qr')
    
    if order is None:
        order = _get_ar_order(x)
        
    n = len(x)
    m = order
    
    lm1 = get_lm(y=x, x=x, m=m)
    fit1 = lm1.fittedvalues**2
    res1 = lm1.resid
    
    lm2 = get_lm(y=np.r_[np.nan * np.ones(m), fit1], x=x, m=m)
    res2 = lm2.resid
    
    lm3 = sm.OLS(res1, res2).fit(method='qr')
    
    test_stat = lm3.fvalue * (n - 2 * m - 2)/(n - m - 1)
    p_value = 1 - stats.f.cdf(test_stat, dfn=1, dfd=n - 2 * m - 2)
    
    return {
        'test_stat': test_stat,
        'p_value': p_value,
        'order': order
    }


def Tsay_test(x, order=None):
    
    if order is None:
        order = _get_ar_order(x)
    
    N = len(x)
    m = order
    
    X1 = np.nan * np.ones((N, 1 + m))
    X2 = np.nan * np.ones((N, 1 + m + int(m * (m + 1)/2)))
    
    X1[:, 0] = x
    for i in range(1, m+1):
        X1[i:, i] = x[:-i]
        
    X2[:, :m+1] = X1
    
    i = m+1
    for j in range(1, m+1):
        for k in range(1, j+1):
            X2[:, i] = X2[:, j] * X2[:, k]
            i += 1
    
    # Omit NaNs
    X1 = X1[~np.any(np.isnan(X1), axis=1)]
    X2 = X2[~np.any(np.isnan(X2), axis=1)]
    
    y = X1[:, 0]
    
    lm1 = sm.OLS(y, sm.add_constant(X1[:, 1:])).fit()
    lm2 = sm.OLS(y, sm.add_constant(X2[:, 1:])).fit()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a1 = anova_lm(lm1, lm2)
    
    return {
        'test_stat': a1['F'][1],
        'p_value': a1['Pr(>F)'][1],
        'order': order
    }