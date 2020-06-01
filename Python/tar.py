import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt

def tar_display(obj):
    p1, p2, d = obj['p1'], obj['p2'], obj['d']
    print('Model: SETAR(%i, %i, %i), with model delay %i' % (2, p1, p2, d))
    
    thd, method = obj['thd'], obj['method']
    print('Estimated threshold %.4g, from a %s fit with thresholds.' % (thd, method))
    
    i1, i2, m = obj['i1'], obj['i2'], obj['m']
    print('Searched from the %.1f percentile to the %.1f percentile of all data.' % (100 * i1/m, 100 * i2/m))
    
    thdindex = obj['thdindex']
    print('The estimated threshold is %.1f percentile of all data.' % (100 * thdindex/m))
    
    AIC = obj['AIC']
    print('Nominal AIC: %.4g' % AIC)
    print()
    
    print('''
==============================================================================
|                               Lower Regime                                 |
==============================================================================
''')
    
    qr1 = obj['qr1']
    print(qr1.summary())
    
    print('\n(Unbiased) RMS:')
    rms1 = obj['rms1']
    print('%.4g' % rms1)
    
    print('with number of data falling into the regime:')
    n1 = obj['n1']
    print('%i' % n1)
    
    print('(Max. likelihood) RMS for each series (denominator=sample size in the regime):')
    rss1 = obj['rss1']
    print('%.4g' % (rss1 / n1))
    
    print('''
==============================================================================
|                               Upper Regime                                 |
==============================================================================
''')

    qr2 = obj['qr2']
    print(qr2.summary())
    
    print('\n(Unbiased) RMS:')
    rms2 = obj['rms2']
    print('%.4g' % rms2)
    
    print('with number of data falling into the regime:')
    n2 = obj['n2']
    print('%i' % n2)
    
    print('(Max. likelihood) RMS for each series (denominator=sample size in the regime):')
    rss2 = obj['rss2']
    print('%.4g' % (rss2 / n2))
    
    
def tar_skeleton(obj, Phi1=None, Phi2=None, thd=None, d=None, 
                 ntransient=500, n=500, xstart=None, plot=True, n_skeleton=50):
    
    def tar1_skeleton(x, Phi1, Phi2, thd, d):
        if x[d - 1] <= thd:
            return np.r_[1, x] @ Phi1
        return np.r_[1, x] @ Phi2
    
    if obj is not None:
        constant1 = obj['qr1'].params[0]
        p1, p2 = obj['p1'], obj['p2']
        d = obj['d']
        p = max(p1, p2)
        mpd = max(p, d)
        Phi1 = np.zeros(mpd + 1)
        Phi2 = np.zeros(mpd + 1)
        
        if obj['is_constant1']:
            Phi1a = obj['qr1'].params
        else:
            Phi1a = np.r_[0, obj['qr1'].params]
            
        if obj['is_constant2']:
            Phi2a = obj['qr2'].params
        else:
            Phi2a = np.r_[0, obj['qr2'].params]
            
        Phi1[:len(Phi1a)] = Phi1a
        Phi2[:len(Phi2a)] = Phi2a
            
        thd = obj['thd']
        if xstart is None:
            xstart = obj['y'][::-1][:mpd][::-1]
            
    if obj is None and xstart is None:
        xstart = np.zeros(mpd)
        
    x = np.zeros(ntransient + n)
    x[:mpd] = xstart
    for i in range(mpd, ntransient + n):
        x[i] = tar1_skeleton(x[(i-mpd):i][::-1], Phi1=Phi1, Phi2=Phi2, thd=thd, d=d)
    
    tail = x[::-1][:n_skeleton][::-1]
    
    if np.abs(tail[-1]) > 10**7:
        print('Skeleton explodes to infinity')
        return
    
    nocycle = True
    i = 1
    m = len(tail)
    while nocycle and i < np.floor(m / 2):
        nocycle = np.any(np.abs(tail[i:] - tail[:-i]) > 1e-4)
        if not nocycle:
            break
        i += 1
        
    if nocycle:
        print('No limit cycle')
        print('Tail part of the skeleton:')
        print(tail)
    else:
        print('Limit cycle of length %i' % i)
        print('Cycle:')
        print(tail[:i])
        
    if plot:
        ax = plt.gca()
        ax.plot(np.arange(1, m+1), tail, marker='o')
        ax.set_xlabel('t')
        ax.set_ylabel('Skeleton')
    
    return tail

    
def tar(y, p1, p2, d, threshold=None, method="MAIC", is_constant1=True, is_constant2=True, 
        transform=None, center=False, standard=None, estimate_thd=True,
        a=0.05, b=0.95, order_select=True, display=False):
    
    def cvar(x, df=1):
        return np.sum(x**2) / (len(x) - df)
    
    def findstart(x, nseries, indexid, p):
        m = x.shape[0]
        m_lookup = np.arange(1, m+1).astype('int')
        amax = 0
        for i in range(1, nseries+1):
            amax = max(np.r_[amax, m_lookup[np.cumsum(x[:, indexid - 1] == i) == p]])
        return amax
    
    def fmaic(R, n, order_select=False):
        k = R.shape[1]
        v = np.cumsum((R[1:, -1]**2)[::-1])[::-1]
        AIC = n * np.log(v / n) + 2 * np.arange(1, k)
        order = np.argmin(AIC) + 1
        like = (-n * np.log(v/n))/2 - n/2 - n * np.log(2 * np.pi)/2

        if order_select:
            return {
                'MAIC': min(AIC),
                'order': np.argmin(AIC),
                'AIC': AIC,
                'like': like[order - 1]
            }

        return {
            'MAIC': AIC[-1],
            'order': k - 2,
            'AIC': AIC,
            'like': like[len(AIC) - 1]
        }
    
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
    
    MAIC = (method == "MAIC")
    dataf = y
    if len(dataf.shape) == 1:
        dataf = np.expand_dims(dataf, axis=1)
    if transform is not None:
        dataf = transform(dataf)
        
    means = np.nanmean(dataf, axis=0)
    stds = np.sqrt(np.nanvar(dataf, axis=0, ddof=1))
    
    if standard is not None:
        dataf = standard(dataf)
        
    nseries = dataf.shape[1]
    res = makedata(dataf, p1, p2, d, is_constant1=is_constant1, is_constant2=is_constant2, thd_by_phase=False)
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
    
    i1 = max(q1 - 3, lbound + 1, 2 * p1 + 1, d, findstart(xy1, nseries, q1, p1 + 2))
    i1 = max(i1, int(np.floor(a * m)))
    
    i2 = m - max(q2 - 3, ubound + 1, 2 * p2 + 1, d, findstart(xy2[::-1, :], nseries, q2, p2 + 2)) - 1
    i2 = min(i2, int(np.ceil(b * m)))
    
    aic1 = np.repeat(np.inf, m)
    aic2 = np.repeat(np.inf, m)
    rss1 = np.repeat(np.inf, m)
    rss2 = np.repeat(np.inf, m)
    
    like1 = np.repeat(-np.inf, m)
    like2 = np.repeat(-np.inf, m)
    
    R = np.linalg.qr(xy1[:i1, :-2], mode='r')
    
    aic_no_thd = None
    if estimate_thd:
        posy = q1 - 3
        rss1[i1 - 1] = R[posy, posy]**2
        res_fmaic = fmaic(R, i1, order_select=order_select)
        like1[i1 - 1] = res_fmaic['like']
        aic1[i1 - 1] = res_fmaic['MAIC']
        for i in range(i1 + 1, i2 + 1):
            R = np.linalg.qr(np.concatenate((R, xy1[(i-1):i, :-2])), mode='r')
            rss1[i - 1] = R[posy, posy]**2
            res_fmaic = fmaic(R, i, order_select=order_select)
            like1[i - 1] = res_fmaic['like']
            aic1[i - 1] = res_fmaic['MAIC']
        
        posy = q2 - 3
        R = np.linalg.qr(xy2[i2:, :-2], mode='r')
        rss2[i2 - 1] = R[posy, posy]**2
        res_fmaic = fmaic(R, m - i2, order_select=order_select)
        like2[i2 - 1] = res_fmaic['like']
        aic2[i2 - 1] = res_fmaic['MAIC']
        for i in range(i2 - 1, i1 - 1, -1):
            R = np.linalg.qr(np.concatenate((R, xy2[i:(i+1), :-2])), mode='r')
            rss2[i - 1] = R[posy, posy]**2
            res_fmaic = fmaic(R, m - i, order_select=order_select)
            like2[i - 1] = res_fmaic['like']
            aic2[i - 1] = res_fmaic['MAIC']
            
        rss = rss1 + rss2
        thdindex = np.argmax(rss == min(rss)) + 1
        aic = aic1 + aic2
        if MAIC:
            # Set AIC out of interval to infinite
            s_before = np.arange(i1).astype('int')
            s_after = np.arange(i2+1, m).astype('int')
            aic[np.r_[s_before, s_after]] = np.inf
            aic_no_thd = min(aic)
            thdindex = np.argmax(aic == min(aic)) + 1
        
        thd = xy1[thdindex - 1, -2]
    else:
        thd = threshold
        thdindex = np.argmax(xy1[:, -2] > thd)
        
    R1 = np.linalg.qr(xy1[:thdindex, :-2], mode='r')
    
    if MAIC:
        order1 = fmaic(R1, thdindex, order_select=order_select)['order'] + 1
        subset1 = np.arange(order1).astype('int')
        p1 = order1 - (1 if is_constant1 else 0)
    else:
        subset1 = np.arange(q1 - 2).astype('int')
        
    x_regime1 = xy1[:thdindex, subset1]
    y_regime1 = xy1[:thdindex, -3]
    qr1 = sm.OLS(y_regime1, x_regime1).fit()
    
    dxy1 = xy1[:, subset1]
    dxy1[(thdindex+1):] = 0
    
    R2 = np.linalg.qr(xy2[thdindex:m, :-2], mode='r')
    
    if MAIC:
        order2 = fmaic(R2, m - thdindex, order_select=order_select)['order'] + 1
        subset2 = np.arange(order2).astype('int')
        p2 = order2 - (1 if is_constant2 else 0)
    else:
        subset2 = np.arange(q2 - 2).astype('int')
        
    x_regime2 = xy2[thdindex:, subset2]
    y_regime2 = xy2[thdindex:, -3]
    qr2 = sm.OLS(y_regime2, x_regime2).fit()
    
    dxy2 = xy2[:, subset2]
    dxy2[:thdindex] = 0
    
    method = "Minimum AIC" if MAIC else "CLS"
      
    qr1_dataframe = pd.DataFrame({'value':qr1.resid, 'series':xy1[:thdindex, -1].astype('int')})
    rms1 = qr1_dataframe.pivot_table(index='series', aggfunc=lambda x : cvar(x, df=p1+is_constant1)).to_numpy().squeeze()
    n1 = qr1_dataframe.pivot_table(index='series', aggfunc=len).to_numpy().astype('int').squeeze()
    
    qr2_dataframe = pd.DataFrame({'value':qr2.resid, 'series':xy2[thdindex:, -1].astype('int')})
    rms2 = qr2_dataframe.pivot_table(index='series', aggfunc=lambda x : cvar(x, df=p2+is_constant2)).to_numpy().squeeze()
    n2 = qr2_dataframe.pivot_table(index='series', aggfunc=len).to_numpy().astype('int').squeeze()
    
    AIC = n1 * np.log((rms1 * (n1 - p1 - is_constant1))/n1) + \
        n2 * np.log((rms2 * (n2 - p2 - is_constant2))/n2) + \
        (n1 + n2) * (1 + np.log(2 * np.pi)) + \
        2 * (p1 + p2 + is_constant1 + is_constant2 + 1)
    
    def resorted(X, sort_index):
        """ Resort not in place by creating a temporary copy, ensuring consistent indexing """
        X_sorted = np.empty_like(X)
        X_sorted[sort_index] = X
        return X_sorted
    
    residuals = resorted(np.r_[qr1.resid, qr2.resid], sort_l)    
    std_res = resorted(np.concatenate((qr1.resid / np.sqrt(rms1), (qr2.resid / np.sqrt(rms2)))), sort_l)
    dxy1 = resorted(dxy1, sort_l)
    dxy2 = resorted(dxy2, sort_l)

    result = {
        'dxy1': dxy1, 'dxy2': dxy2, 'p1': p1, 'q1': q1, 'd': d,
        'qr1': qr1, 'qr2': qr2, 'x_regime1': x_regime1, 'y_regime1': y_regime1,
        'x_regime2': x_regime2, 'y_regime2': y_regime2, 'thd': thd,
        'thdindex': thdindex, 'i1': i1, 'i2': i2,
        'x': xy1[:, -2], 'm': m, 'rss1': rms1 * (n1 - p1 - 1), 'rss2': rms2 * (n2 - p2 - 1),
        'n1': n1, 'n2': n2, 'std_res': std_res,
        'p2': p2, 'rms1': rms1, 'rms2': rms2, 'is_constant1': is_constant1,
        'is_constant2': is_constant2, 'residuals': residuals, 'AIC': AIC,
        'aic_no_thd': aic_no_thd, 'y': y, 'like': max(like1 + like2), 'method': method
    }
    
    if display:
        tar_display(result)
    
    return result