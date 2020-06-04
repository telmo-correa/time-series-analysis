"""
Adaptation of R's spectrum.R spectral analysis functions, used in Chapters 14+.

Based on version at https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/spectrum.R .
""" 

import numpy as np
from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker

import matplotlib.pyplot as plt


def spec_taper(x, p=0.1):
    """
    Computes a tapered version of x, with tapering p.
    
    Adapted from R's stats::spec.taper.
    """
    
    p = np.r_[p]
    assert np.all((p >= 0) & (p < 0.5)), "'p' must be between 0 and 0.5"
    
    x = np.r_[x].astype('float64')
    original_shape = x.shape
    
    assert len(original_shape) <= 2, "'x' must have at most 2 dimensions"
    while len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    
    nr, nc = x.shape
    if len(p) == 1:
        p = p * np.ones(nc)
    else:
        assert len(p) == nc, "length of 'p' must be 1 or equal the number of columns of 'x'"
    
    for i in range(nc):
        m = int(np.floor(nr * p[i]))
        if m == 0:
            continue
        w = 0.5 * (1 - np.cos(np.pi * np.arange(1, 2 * m, step=2)/(2 * m)))
        x[:, i] = np.r_[w, np.ones(nr - 2 * m), w[::-1]] * x[:, i]
    
    x = np.reshape(x, original_shape)
    return x

def spec_ci(df, coverage=0.95):
    """
    Computes the confidence interval for a spectral fit, based on the number of degrees of freedom.
    
    Adapted from R's stats::plot.spec.
    """
    
    assert coverage >= 0 and coverage < 1, "coverage probability out of range [0, 1)"
    
    tail = 1 - coverage
    
    phi = stats.chi2.cdf(x=df, df=df)
    upper_quantile = 1 - tail * (1 - phi)
    lower_quantile = tail * phi
    
    return df / stats.chi2.ppf([upper_quantile, lower_quantile], df=df)

def spec_pgram(x, xfreq=1, spans=None, kernel=None, taper=0.1, pad=0, fast=True, demean=False, detrend=True, 
               plot=True, **kwargs):
    """
    Computes the spectral density estimate using a periodogram.  Optionally, it also:
    - Uses a provided kernel window, or a sequence of spans for convoluted modified Daniell kernels.
    - Tapers the start and end of the series to avoid end-of-signal effects.
    - Pads the provided series before computation, adding pad*(length of series) zeros at the end.
    - Pads the provided series before computation to speed up FFT calculation.
    - Performs demeaning or detrending on the series.
    - Plots results.
    
    Implemented to ensure compatibility with R's spectral functions, as opposed to reusing scipy's periodogram.
    
    Adapted from R's stats::spec.pgram.
    """
    def daniell_window_modified(m):
        """ Single-pass modified Daniell kernel window.
        
        Weight is normalized to add up to 1, and all values are the same, other than the first and the
        last, which are divided by 2.
        """
        def w(k):
            return np.where(np.abs(k) < m, 1 / (2*m), np.where(np.abs(k) == m, 1/(4*m), 0))

        return w(np.arange(-m, m+1))

    def daniell_window_convolve(v):
        """ Convolved version of multiple modified Daniell kernel windows.
        
        Parameter v should be an iterable of m values.
        """
        
        if len(v) == 0:
            return np.r_[1]

        if len(v) == 1:
            return daniell_window_modified(v[0])

        return signal.convolve(daniell_window_modified(v[0]), daniell_window_convolve(v[1:]))
    
    # Ensure we can store non-integers in x, and that it is a numpy object
    x = np.r_[x].astype('float64')
    original_shape = x.shape
    
    # Ensure correct dimensions
    assert len(original_shape) <= 2, "'x' must have at most 2 dimensions"
    while len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
        
    N, nser = x.shape
    N0 = N
    
    # Ensure only one of spans, kernel is provided, and build the kernel window if needed
    assert (spans is None) or (kernel is None), "must specify only one of 'spans' or 'kernel'"
    if spans is not None:
        kernel = daniell_window_convolve(np.floor_divide(np.r_[spans], 2))
        
    # Detrend or demean the series
    if detrend:
        t = np.arange(N) - (N - 1)/2
        sumt2 = N * (N**2 - 1)/12
        x -= (np.repeat(np.expand_dims(np.mean(x, axis=0), 0), N, axis=0) + np.outer(np.sum(x.T * t, axis=1), t/sumt2).T)
    elif demean:
        x -= np.mean(x, axis=0)
        
    # Compute taper and taper adjustment variables
    x = spec_taper(x, taper)
    u2 = (1 - (5/8) * taper * 2)
    u4 = (1 - (93/128) * taper * 2)
         
    # Pad the series with copies of the same shape, but filled with zeroes
    if pad > 0:
        x = np.r_[x, np.zeros((pad * x.shape[0], x.shape[1]))]
        N = x.shape[0]
        
    # Further pad the series to accelerate FFT computation
    if fast:
        newN = fft.next_fast_len(N, True)
        x = np.r_[x, np.zeros((newN - N, x.shape[1]))]
        N = newN
        
    # Compute the Fourier frequencies (R's spec.pgram convention style)
    Nspec = int(np.floor(N/2))
    freq = (np.arange(Nspec) + 1) * xfreq / N
    
    # Translations to keep same row / column convention as stats::mvfft
    xfft = fft.fft(x.T).T
    
    # Compute the periodogram for each i, j
    pgram = np.empty((N, nser, nser), dtype='complex')
    for i in range(nser):
        for j in range(nser):
            pgram[:, i, j] = xfft[:, i] * np.conj(xfft[:, j]) / (N0 * xfreq)
            pgram[0, i, j] = 0.5 * (pgram[1, i, j] + pgram[-1, i, j])
       
    if kernel is None:    
        # Values pre-adjustment
        df = 2
        bandwidth = np.sqrt(1 / 12)
    else:
        def conv_circular(signal, kernel):
            """
            Performs 1D circular convolution, in the same style as R::kernapply,
            assuming the kernel window is centered at 0.
            """
            pad = len(signal) - len(kernel)
            half_window = int((len(kernel) + 1) / 2)
            indexes = range(-half_window, len(signal) - half_window)
            orig_conv = np.real(fft.ifft(fft.fft(signal) * fft.fft(np.r_[np.zeros(pad), kernel])))
            return orig_conv.take(indexes, mode='wrap')
                
        # Convolve pgram with kernel with circular conv
        for i in range(nser):
            for j in range(nser):
                pgram[:, i, j] = conv_circular(pgram[:, i, j], kernel)
        
        df = 2 / np.sum(kernel**2)
        m = (len(kernel) - 1)/2
        k = np.arange(-m, m+1)
        bandwidth = np.sqrt(np.sum((1/12 + k**2) * kernel))
    
    df = df/(u4/u2**2)*(N0/N)
    bandwidth = bandwidth * xfreq/N
    
    # Remove padded results
    pgram = pgram[1:(Nspec+1), :, :]
    
    spec = np.empty((Nspec, nser))
    for i in range(nser):
        spec[:, i] = np.real(pgram[:, i, i])
    
    if nser == 1:
        coh = None
        phase = None
    else:
        coh = np.empty((Nspec, int(nser * (nser - 1)/2)))
        phase = np.empty((Nspec, int(nser * (nser - 1)/2)))
        for i in range(nser):
            for j in range(i+1, nser):
                index = int(i + j*(j-1)/2)
                coh[:, index] = np.abs(pgram[:, i, j])**2 / (spec[:, i] * spec[:, j])
                phase[:, index] = np.angle(pgram[:, i, j])
            
    spec = spec / u2
    spec = spec.squeeze()
    
    results = {
        'freq': freq,
        'spec': spec,
        'coh': coh,
        'phase': phase,
        'kernel': kernel,
        'df': df,
        'bandwidth': bandwidth,
        'n.used': N,
        'orig.n': N0,
        'taper': taper,
        'pad': pad,
        'detrend': detrend,
        'demean': demean,
        'method': 'Raw Periodogram' if kernel is None else 'Smoothed Periodogram'
    }
    
    if plot:
        plot_spec(results, coverage=0.95, **kwargs)
    
    return results

def spec_ar(x, x_freq=1, n_freq=500, order_max=None, plot=True, **kwargs):
    x = np.r_[x]
    N = len(x)
    if order_max is None:
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
        
    order = best_results['order']
    freq = np.arange(0, n_freq) / (2 * (n_freq - 1))
      
    if order >= 1:
        ar, sigma2 = best_results['ar'], best_results['sigma2']
    
        outer_xy = np.outer(freq, np.arange(1, order+1))
        cs = np.cos(2 * np.pi * outer_xy) @ ar
        sn = np.sin(2 * np.pi * outer_xy) @ ar

        spec = sigma2 / (x_freq*((1 - cs)**2 + sn**2))
        
    else:
        sigma2 = best_results['sigma2']
        spec = (sigma2 / x_freq) * np.ones(len(freq))
    
    results = {
        'freq': freq,
        'spec': spec,
        'coh': None,
        'phase': None,
        'n.used': len(x),
        'method': 'AR(' + str(order) + ') spectrum'
    } 
    
    if plot:
        plot_spec(results, coverage=None, **kwargs)
    
    return results

def plot_spec(spec_res, coverage=None, ax=None, title=None):
    """Convenience plotting method, also includes confidence cross in the same style as R.
    
    Note that the location of the cross is irrelevant; only width and height matter."""
    f, Pxx = spec_res['freq'], spec_res['spec']
    
    if coverage is not None:
        ci = spec_ci(spec_res['df'], coverage=coverage)
        conf_x = (max(spec_res['freq']) - spec_res['bandwidth']) + np.r_[-0.5, 0.5] * spec_res['bandwidth']
        conf_y = max(spec_res['spec']) / ci[1]

    if ax is None:
        ax = plt.gca()
    
    ax.plot(f, Pxx, color='C0')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Log Spectrum')
    ax.set_yscale('log')
    if coverage is not None:
        ax.plot(np.mean(conf_x) * np.r_[1, 1], conf_y * ci, color='red')
        ax.plot(conf_x, np.mean(conf_y) * np.r_[1, 1], color='red')

    ax.set_title(spec_res['method'] if title is None else title)
        