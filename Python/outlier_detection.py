import numpy as np
from scipy.stats import norm
from scipy.signal import lfilter

from statsmodels.tsa.arima_process import arma_impulse_response


def detectAO(model, alpha=0.05, robust=True):
    """ Adapted from R's TSA library detectAO. 
    
    This function serves to detect whether there is any AO.
    It implements the test statistic lambda_{1,t} proposed by Chang, 
    Chen and Tiao (1988).
    """
    
    d = np.maximum(model.loglikelihood_burn, model.nobs_diffuse)
    resid = model.filter_results.standardized_forecasts_error[0]
    
    n_resid = len(resid)
    
    pi_weights = arma_impulse_response(
        ar=-model.polynomial_ma, 
        ma=-model.polynomial_ar, 
        leads=n_resid
    )
    rho2 = 1 / np.cumsum(pi_weights**2)
    k = np.zeros(2*len(resid) - 1)
    k[-len(resid):] = resid[::-1]
    omega = lfilter(b=pi_weights, a=np.ones(1), x=k)[n_resid - 1:] * rho2
    
    if robust:
        sigma = np.sqrt(np.pi/2) * np.nanmean(np.abs(resid))
    else:
        sigma = np.sqrt(model.params['sigma2'])
        
    lambda2T = (omega / (sigma * np.sqrt(rho2)))[::-1]
    
    cutoff = norm.ppf(1 - alpha/(2*len(lambda2T)))
    out = (np.abs(lambda2T) > cutoff)
    ind = np.array(np.where(out)).squeeze()
    lambda2 = lambda2T[out].squeeze()

    return lambda2, ind
    

def detectIO(model, alpha=0.05, robust=True):
    """ Adapted from R's TSA library detectIO. 
    
    This function serves to detect whether there is any IO.
    It implements the test statistic lambda_{1,t} proposed by Chang, 
    Chen and Tiao (1988).
    """

    d = np.maximum(model.loglikelihood_burn, model.nobs_diffuse)
    resid = model.filter_results.standardized_forecasts_error[0]
    
    if robust:
        sigma = np.sqrt(np.pi/2) * np.nanmean(np.abs(resid))
    else:
        sigma = np.sqrt(model.params['sigma2'])

    lambda1T = resid / sigma
    
    cutoff = norm.ppf(1 - alpha/(2*len(lambda1T)))
    out = (np.abs(lambda1T) > cutoff)
    ind = np.array(np.where(out)).squeeze()
    lambda1 = lambda1T[out].squeeze()

    return lambda1, ind