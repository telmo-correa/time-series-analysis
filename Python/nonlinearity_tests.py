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

def tlrt(y, p, d=1, transform=None, a=0.25, b=0.75):
    
    def cvar(x, df=1):
        return np.sum(x**2) / (len(x) - df)
    
    def findstart(x, nseries, indexid, p):
        m = x.shape[0]
        m_lookup = np.arange(1, m+1).astype('int')
        amax = 0
        for i in range(1, nseries+1):
            amax = max(np.r_[amax, m_lookup[np.cumsum(x[:, indexid - 1] == i) == p]])
        return amax
    
    def makexy(x, p, start, d, thd_by_phase=False):
        n = len(x)
        xy = np.empty((n - start + 1, p + 2))
        for i in range(p):
            xy[:, i] = x[(start - i - 2):(n - i - 1)]

        xy[:, p] = x[(start-1):n]

        if thd_by_phase:
            xy[:, p+1] = x[(start-2):(n-1)] - x[(start-3):(n-2)]
        else:
            xy[:, p+1] = x[(start-d-1):(n-d)]

        return xy
    
    def setxy(old_xy, p, p1, nseries, is_coefficient=True):
        assert nseries >= 1, "nseries must be at least 1"

        is_coefficient_index = 1 if is_coefficient else 0
        ns_prefix_unit = (is_coefficient_index + p1)

        new_xy = np.empty((old_xy.shape[0], (nseries - 1) * ns_prefix_unit + is_coefficient_index + p1 + 3))
        temp = np.empty((old_xy.shape[0], is_coefficient_index + p1))

        if p1 > 0:
            temp[:, is_coefficient_index:] = old_xy[:, :p1]
            new_xy[:, (-p1-3):-3] = old_xy[:, :p1]

        new_xy[:, -3:] = old_xy[:, p:(p+3)]

        if is_coefficient:
            temp[:, -(is_coefficient_index + p1)] = 1
            new_xy[:, -(is_coefficient_index + p1 + 3)] = 1

        for i in range(2, nseries + 1):
            select = old_xy[:, p + 2] == i
            zero = np.zeros_like(temp)
            zero[select, :] = temp[select, :]
            new_xy[:, ((i-2)*ns_prefix_unit):((i-1)*ns_prefix_unit)] = zero

        return new_xy
    
    def dna(x):
        """ If any element in a row is NaN, replace the whole row with NaN """
        return np.where(np.repeat(np.expand_dims(np.any(np.isnan(x), axis=1), axis=1), x.shape[1], axis=1), np.nan, x)
    
    def makedata(dataf, p1, p2, d, is_constant1=True, is_constant2=True, thd_by_phase=False):
        n, nseries = dataf.shape
        start = max(p1, p2, d) + 1
        p = max(p1, p2)

        makexy_nrows = (n - start + 1)

        xy = np.empty((nseries * makexy_nrows, p + 3))
        for i in range(nseries):
            xy[(i*makexy_nrows):((i+1)*makexy_nrows), -1] = i+1
            xy[(i*makexy_nrows):((i+1)*makexy_nrows), :-1] = makexy(dataf[:, i], p, start, d, thd_by_phase=thd_by_phase)

        xy = dna(xy)
        sort_list = np.argsort(xy[:, p + 1])
        xy = xy[sort_list]

        xy1 = setxy(xy, p, p1, nseries, is_coefficient=is_constant1)
        xy2 = setxy(xy, p, p2, nseries, is_coefficient=is_constant2)

        return {
            'xy1': xy1,
            'xy2': xy2,
            'sort_list': sort_list
        }
    
    def p_value_tlrt(y, a=0.25, b=0.75, p=0):
        def t1(x):
            F = stats.norm.cdf(0.5)
            return 0.5 * np.log(F / (1 - F))

        lower = stats.norm.ppf(a)
        upper = stats.norm.ppf(b)

        if p == 0:
            temp = t1(upper) - t1(lower)
        elif p > 0:
            def tp1(x):
                F, f = stats.norm.cdf(x), stats.norm.pdf(x)
                b = 2 * F - x * f
                c = F * (F - x * f) - f * f
                root = 0.5 * (b + np.sqrt(b * b - 4 * c))

                return 0.5 * np.log(root/(1 - root))

            def tp2(x):
                F, f = stats.norm.cdf(x), stats.norm.pdf(x)
                b = 2 * F - x * f
                c = F * (F - x * f) - f * f
                root = 0.5 * (b - np.sqrt(b * b - 4 * c))

                return 0.5 * np.log(root/(1 - root))

            temp = (p1 - 1) * (t1(upper) - t1(lower)) + tp1(upper) - tp1(lower) + tp2(upper) - tp2(lower)

        if p > 0:
            return 1 - np.exp(-2 * stats.chi2.pdf(y, df=p + 1) * (y/(p + 1) - 1) * temp)

        z = np.sqrt(y)
        return np.sqrt(2/np.pi) * np.exp(-y/2) * (temp * (z - 1/z) + 1/z)
    
    dataf = y
    if len(dataf.shape) == 1:
        dataf = np.expand_dims(dataf, axis=1)
    if transform is not None:
        dataf = transform(dataf)
        
    p1, p2 = p, p
    res = makedata(dataf, p1, p2, d)
    xy1, xy2, sort_l = res['xy1'], res['xy2'], res['sort_list']
    
    m = xy1.shape[0]
    q1, q2 = xy1.shape[1], xy2.shape[1]
    
    s = np.arange(q1 - p1 - 4, q1 - 3).astype('int')
    s_before = np.arange(0, q1 - p1 - 4).astype('int')
    s_after = np.arange(q1 - 3, q1).astype('int')
    xy1 = xy1[:, np.r_[s, s_before, s_after]]
    
    s = np.arange(q2 - p2 - 4, q2 - 3).astype('int')
    s_before = np.arange(0, q2 - p2 - 4).astype('int')
    s_after = np.arange(q2 - 3, q2).astype('int')
    xy2 = xy2[:, np.r_[s, s_before, s_after]]
    
    cleaned_series = dataf[~np.isnan(dataf)]
    lbound = np.sum(cleaned_series == min(cleaned_series))
    ubound = np.sum(cleaned_series == max(cleaned_series))
    
    i1 = max(q1 - 3, lbound + 1, p1 + 1, d, findstart(xy1, 1, q1, p1 + 2))
    i1 = max(i1, int(np.floor(a * m)))
    
    i2 = m - max(q1 - 3, ubound + 1, p2 + 1, d, findstart(xy1[::-1, :], 1, q1, p1 + 2)) - 1
    i2 = min(i2, int(np.ceil(b * m)))
    
    truea, trueb = i1/m, i2/m
    
    rss1 = np.repeat(np.inf, m)
    rss2 = np.repeat(np.inf, m)
    
    R = np.linalg.qr(xy1[:i1, :-2], mode='r')
    posy = q1 - 3
    rss1[i1 - 1] = R[posy, posy]**2
    for i in range(i1 + 1, i2 + 1):
        R = np.linalg.qr(np.concatenate((R, xy1[(i-1):i, :-2])), mode='r')
        rss1[i - 1] = R[posy, posy]**2
    
    R = np.linalg.qr(xy2[i2:, :-2], mode='r')
    posy = q2 - 3
    rss2[i2 - 1] = R[posy, posy]**2
    for i in range(i2 - 1, i1 - 1, -1):
        R = np.linalg.qr(np.concatenate((R, xy2[i:(i+1), :-2])), mode='r')
        rss2[i - 1] = R[posy, posy]**2
        
    rss = rss1 + rss2
    
    rss_H1 = min(rss)
    
    R = np.linalg.qr(xy1[:, :-2], mode='r')
    rss_H0 = (R[posy, posy])**2
    
    test_stat = m * (rss_H0 - rss_H1)/rss_H1
    p_value = p_value_tlrt(test_stat, a=truea, b=trueb, p=p)
    
    return {
        'percentiles': np.r_[truea, trueb],
        'test_statistic': test_stat,
        'p_value': p_value
    }