## Plot monthly values, with cyclic colors

from matplotlib import cm
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

def plot_monthly(dates, values, xlabel, ylabel):
    colormap = cm.get_cmap('twilight')
    plt.figure(figsize=(12, 4))
    plt.plot(dates, values, marker='o', markerfacecolor='none', linestyle='solid', ms=5, alpha=0.2, color='C0')
    for d, v in zip(dates, values):
        marker = r'$\rm{' + d.strftime("%b") + '}$'
        color = to_hex(colormap((d.month - 1)/12))
        plt.plot(d, v, marker=marker, markersize=12, color=color)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.show()


# Generate regression summary plots, mimicking R's plot(lm(...))
# Adapted from: https://towardsdatascience.com/going-from-r-to-python-linear-regression-diagnostic-plots-144d1c4aa5a

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.api as sm


def get_top_n(model, n):
    return np.argpartition(model.resid, -n)[:n]

def plot_residuals_vs_fitted(model, ax, annotate_index):
    x, y = model.fittedvalues, model.resid
    ax.plot(x, y, marker='o', markerfacecolor='none', ls='none')

    # Plot LOWESS regression results
    smoothed = lowess(y, x)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red')

    # Plot reference 0-residual dashed line
    ax.hlines(0, xmin=min(x), xmax=max(x), ls=':', color='C0')

    # Annotate indexed residuals
    if annotate_index is not None:
        for i in annotate_index:
            ax.annotate((i + 1), xy=(x[i], y[i]))

    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted')

def plot_normal_qq(model, ax):
    # Use StatsModels ProbPlot to compute quantiles
    probplot = sm.ProbPlot(model.resid_pearson)

    x, y = probplot.theoretical_quantiles, probplot.sample_quantiles
    ax.plot(x, y, marker='o', markerfacecolor='none', ls='none')

    # Draw 45 degree dotted line
    vmin, vmax = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
    ax.plot([vmin, vmax], [vmin, vmax], linestyle=':', color='C0')

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Standardized Residuals')
    ax.set_title('Normal Q-Q')
    
def plot_scale_location(model, ax, annotate_index):
    x, y = model.fittedvalues, np.sqrt(np.abs(model.resid_pearson))
    ax.plot(x, y, marker='o', markerfacecolor='none', ls='none')

    # Plot LOWESS regression results
    smoothed = lowess(y, x)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red')

    # Annotate indexed residuals
    if annotate_index is not None:
        for i in annotate_index:
            ax.annotate((i + 1), xy=(x[i], y[i]))

    ax.set_xlabel('Fitted values')
    ax.set_ylabel(r'$\sqrt{|\rm{Standardized\;residuals}|}$')
    ax.set_title('Scale-Location')
    
def plot_residuals_vs_leverage(model, ax, annotate_index):  
    x, y = model.get_influence().hat_matrix_diag, model.resid_pearson
    ax.plot(x, y, marker='o', markerfacecolor='none', ls='none')

    # Plot LOWESS regression results
    smoothed = lowess(y, x)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color='red')

    # Explicitly control the limits of the plot to ensure that x = 0 is included
    # (and to not be affected by Cook's distance)
    xmin, xmax = min(0, min(x)), max(x)
    ymin, ymax = min(y), max(y)
    pxmin, pxmax = xmin - 0.05 * (xmax - xmin), xmax + 0.05 * (xmax - xmin)
    pymin, pymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)

    # Add dotted lines on x = 0 and y = 0
    ax.hlines(0, xmin=pxmin, xmax=pxmax, ls=':', color='C0')
    ax.vlines(0, ymin=pymin, ymax=pymax, ls=':', color='C0')

    # Annotate indexed residuals
    if annotate_index is not None:
        for i in annotate_index:
            ax.annotate((i + 1), xy=(x[i], y[i]))

    # Calculate Cook's distance
    xpos = max(x) * 1.05
    cooksx = np.linspace(min(x), xpos, 50)
    p = len(model.params)
    
    # Clip quantities to avoid graphical bug
    y_epsilon = 1e-2 * (ymax - ymin)
    poscooks1y = np.minimum(np.sqrt((p*(1-cooksx))/cooksx), pymax + y_epsilon)
    poscooks05y = np.minimum(np.sqrt(0.5*(p*(1-cooksx))/cooksx), pymax + y_epsilon)
    negcooks1y = np.maximum(-np.sqrt((p*(1-cooksx))/cooksx), pymin - y_epsilon)
    negcooks05y = np.maximum(-np.sqrt(0.5*(p*(1-cooksx))/cooksx), pymin - y_epsilon)

    ax.plot(cooksx, poscooks1y, label = "Cook's Distance", ls=':', color='r')
    ax.plot(cooksx, poscooks05y, ls = ':', color = 'r')
    ax.plot(cooksx, negcooks1y, ls = ':', color = 'r')
    ax.plot(cooksx, negcooks05y, ls = ':', color = 'r')

    ax.annotate('1.0', xy = (xpos, poscooks1y[-1]), color = 'r')
    ax.annotate('0.5', xy = (xpos, poscooks05y[-1]), color = 'r')
    ax.annotate('1.0', xy = (xpos, negcooks1y[-1]), color = 'r')
    ax.annotate('0.5', xy = (xpos, negcooks05y[-1]), color = 'r')

    ax.legend()

    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')
    ax.set_xlim([pxmin, pxmax])
    ax.set_ylim([pymin, pymax])

    ax.set_title('Residuals vs Leverage')
    
def plot_summary(model):
    plt.figure(figsize=(12, 12))

    annotate_index = get_top_n(model, 3)

    plot_residuals_vs_fitted(model, plt.subplot(2, 2, 1), annotate_index)
    plot_normal_qq(model, plt.subplot(2, 2, 2))
    plot_scale_location(model, plt.subplot(2, 2, 3), annotate_index)
    plot_residuals_vs_leverage(model, plt.subplot(2, 2, 4), annotate_index)

    plt.tight_layout()
    plt.show()