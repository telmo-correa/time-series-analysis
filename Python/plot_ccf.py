""" 
Scripts from https://github.com/statsmodels/statsmodels/blob/master/statsmodels/graphics/tsaplots.py
alongside a new function to plot_ccf, in the same style as plot_acf and plot_pacf.
"""

import numpy as np
import scipy.signal as ss
from scipy.stats import norm

from statsmodels.graphics import utils


def ccf(x, y, lag_max=100):
    """ 
    CCF implementation reproducing R's ccf behavior.
    Adapted from https://stackoverflow.com/a/53994229
    """
    result = ss.correlate(y - np.mean(y), x - np.mean(x), method='direct') / (np.std(y) * np.std(x) * len(y))
    length = (len(result) - 1) // 2
    lo = length - lag_max
    hi = length + (lag_max + 1)

    return result[lo:hi]


def _prepare_data_corr_plot_ccf(x, lags):
    irregular = False
    if lags is None:
        # GH 4663 - use a sensible default value
        nobs = x.shape[0]
        lim = min(int(np.ceil(10 * np.log10(nobs))), nobs - 1)
        lags = np.arange(-lim, lim + 1)
    elif np.isscalar(lags):
        k = int(lags)
        lags = np.arange(-k, k + 1)  # +1 for zero lag
    else:
        irregular = True
        lags = np.asanyarray(lags).astype(np.int)
    nlags = lags.max(0)

    return lags, nlags, irregular

def _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines,
               vlines_kwargs, **kwargs):
    
    if irregular:
        acf_x = acf_x[lags]
        if confint is not None:
            confint = confint[lags]

    if use_vlines:
        ax.vlines(lags, [0], acf_x, **vlines_kwargs)
        ax.axhline(**kwargs)

    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    if 'ls' not in kwargs:
        # gh-2369
        kwargs.setdefault('linestyle', 'None')
    ax.margins(.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.set_title(title)

    if confint is not None:
        if lags[0] == 0:
            lags = lags[1:]
            confint = confint[1:]
            acf_x = acf_x[1:]
        lags = lags.astype(np.float)
        lags[0] -= 0.5
        lags[-1] += 0.5
        ax.fill_between(lags, confint[:, 0] - acf_x,
                        confint[:, 1] - acf_x, alpha=.25)


def plot_ccf(x, y, ax=None, lags=None, *, alpha=.05, use_vlines=True,
             unbiased=False, title='Cross-correlation', vlines_kwargs=None, **kwargs):
    """
    Plot the cross-correlation function
    Plots lags on the horizontal and the correlations on vertical axis.
    Parameters
    ----------
    x : array_like
        Array of time-series values
    y:  array_like
        Array of time-series values
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    lags : {int, array_like}, optional
        An int or array of lag values, used on horizontal axis. Uses
        np.arange(lags) when lags is an int.  If not provided,
        ``lags=np.arange(len(corr))`` is used.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        Bartlett's formula. If None, no confidence intervals are plotted.
    use_vlines : bool, optional
        If True, vertical lines and markers are plotted.
        If False, only markers are plotted.  The default marker is 'o'; it can
        be overridden with a ``marker`` kwarg.
    unbiased : bool
        If True, then denominators for autocovariance are n-k, otherwise n
    title : str, optional
        Title to place on plot.  Default is 'Cross-correlation'
    vlines_kwargs : dict, optional
        Optional dictionary of keyword arguments that are passed to vlines.
    **kwargs : kwargs, optional
        Optional keyword arguments that are directly passed on to the
        Matplotlib ``plot`` and ``axhline`` functions.
    Returns
    -------
    Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    See Also
    --------
    matplotlib.pyplot.xcorr
    matplotlib.pyplot.acorr
    Notes
    -----
    Adapted from 
    https://github.com/statsmodels/statsmodels/blob/master/statsmodels/graphics/tsaplots.py
    """
    fig, ax = utils.create_mpl_ax(ax)

    lags, nlags, irregular = _prepare_data_corr_plot_ccf(x, lags)
    vlines_kwargs = {} if vlines_kwargs is None else vlines_kwargs

    confint = None
    ccf_data = ccf(y, x, lag_max=nlags)
    
    if alpha is not None:
        z = norm.ppf(1 - alpha / 2)
        sl = z / np.sqrt(len(x))
        confint = np.empty((2*nlags+1, 2))
        confint[:, 0] = ccf_data - sl
        confint[:, 1] = ccf_data + sl

    _plot_corr(ax, title, ccf_data, confint, lags, irregular, use_vlines,
               vlines_kwargs, **kwargs)

    return fig