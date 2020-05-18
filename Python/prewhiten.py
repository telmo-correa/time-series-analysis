import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.ar_model import AutoReg


def prewhiten(x, y):
    best_aic, best_lags, best_model = None, None, None
    max_lags = int(np.floor(10 * np.log10(len(x))))
    for k in range(1, max_lags + 1):
        res = AutoReg(x, lags=k).fit()
        if best_aic is None or res.aic > best_aic:
            best_aic, best_lags = res.aic, k
            best_model = res
    
    f = np.r_[1, -best_model.params[1:]]
    nf = len(f)
    x = lfilter(x=x, b=f, a=np.ones(1))[nf:]
    y = lfilter(x=y, b=f, a=np.ones(1))[nf:]
    
    return x, y