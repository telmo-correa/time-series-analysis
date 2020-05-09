import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.ar_model import AutoReg

def eacf(z, ar_max=7, ma_max=13, display=True):
    """
    Translation of the eacf function from R into Python.
   
    Original documentation:
    
    #
    #  PROGRAMMED BY K.S. CHAN, DEPARTMENT OF STATISTICS AND ACTUARIAL SCIENCE,
    #  UNIVERSITY OF IOWA.
    #
    #  DATE: 4/2001
    #  Compute the extended sample acf (ESACF) for the time series stored in z.
    #  The matrix of ESACF with the AR order up to ar.max and the MA order
    #  up to ma.max is stored in the matrix EACFM.
    #  The default values for NAR and NMA are 7 and 13 respectively.
    #  Side effect of the eacf function:
    #  The function prints a coded ESACF table with
    #  significant values denoted by * and nosignificant values by 0, significance
    #  level being 5%.
    #
    #  Output:
    #   eacf=matrix of esacf
    #   symbol=matrix of coded esacf
    #
    """

    def lag1(z, lag=1):
        return pd.Series(z).shift(lag)
    
    def reupm(m, nrow, ncol):
        k = ncol - 1
        m2 = np.empty((m.shape[0], k))
        for i in range(k):
            i1 = i + 1
            work = lag1(m1[:, i])
            work[0] = -1
            temp = m1[:, i1] - work * m1[i1, i1]/m1[i, i]
            temp[i1] = 0
            m2[:, i] = temp
        return m2
    
    def ceascf(m, cov1, nar, ncol, count, ncov, z, zm):
        result = np.zeros(nar+1)
        result[0] = cov1[ncov + count - 2]
        for i in range(1, nar+1):
            A = np.empty((len(z) - i, i+1))
            A[:, 0] = z[i:]
            A[:, 1:] = zm[i:, :i]
            b = np.r_[1, -m[:i, i-1]]
            temp = A @ b
            result[i] = acf(temp, nlags=count, fft=False)[count]
        return result
    
    ar_max = ar_max + 1
    ma_max = ma_max + 1
    nar = ar_max - 1
    nma = ma_max
    ncov = nar + nma + 2
    nrow = nar + nma + 1
    ncol = nrow - 1
    z = np.array(z) - np.mean(z)
    zm = np.empty((len(z), nar))
    for i in range(nar):
        zm[:, i] = lag1(z, lag=i+1)
    cov1 = acf(z, nlags=ncov, fft=False)
    cov1 = np.r_[np.flip(cov1[1:]), cov1]
    ncov = ncov + 1
    m1 = np.zeros((nrow, ncol))
    for i in range(ncol):
        m1[:i+1, i] = AutoReg(z, lags=i+1, trend='c').fit().params[1:]
        
    eacfm = np.empty((ar_max, nma))
    for i in range(nma):
        m2 = reupm(m = m1, nrow = nrow, ncol = ncol)
        ncol = ncol - 1
        eacfm[:, i] = ceascf(m2, cov1, nar, ncol, i+1, ncov, z, zm)
        m1 = m2
    
    work = np.arange(1, nar+2)
    work = len(z) - work + 1
    symbol = np.empty(eacfm.shape, dtype=object)
    for i in range(nma):
        work = work - 1
        symbol[:, i] = np.where(np.abs(eacfm[:, i]) > 2/np.sqrt(work), 'x', 'o')
    symbol = pd.DataFrame(symbol)
    if display:
        print('AR / MA')
        print(symbol)
    
    return {
        'eacf': eacfm,
        'ar.max': ar_max,
        'ma.max': ma_max,
        'symbol': symbol
    }
    