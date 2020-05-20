import numpy as np

def garch_sim(alpha, beta, n=100, rnd=np.random.randn, ntrans=100, **kwargs):
    alpha = np.r_[alpha]
    beta = np.r_[beta]
    
    p = len(beta)
    q = len(alpha) - 1
    
    total_n = n + ntrans
    
    e = rnd(total_n)
    x = np.empty(total_n)
    sigt = np.empty(total_n)
    d = max(p, q)
    sigma2 = np.sum(alpha[1:])
    if p > 0: 
        sigma2 = sigma2 + np.sum(beta)
    assert sigma2 <= 1, "Check model: it does not have finite variance" 
    sigma2 = alpha[0]/(1 - sigma2)
    assert sigma2 > 0, "Check model: it does not have positive variance"
    
    x[:d] = np.sqrt(sigma2) * rnd(d)
    sigt[:d] = sigma2
    
    if p == 0:
        for i in range(d, total_n):
            sigt[i] = alpha @ np.r_[1, x[i - np.arange(q) - 1]**2]
            x[i] = e[i] * np.sqrt(sigt[i])
    
    else:
        for i in range(d, total_n):
            sigt[i] = (alpha @ np.r_[1, x[i - np.arange(q) - 1]**2]) + (beta @ sigt[i - np.arange(p) - 1])
            x[i] = e[i] * np.sqrt(sigt[i])
       
    return x[-n:]