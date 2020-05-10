"""
Implements armasubsets in R, using naive feature subset search (exhaustive, 
beam forward or beam backward) rather than R's regsubsets.
"""

import numpy as np
import pandas as pd
from itertools import chain, combinations
import heapq

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import matplotlib.pyplot as plt


def _get_armasubsets_features(y, nar, nma, y_name = "Y"):
    """ Generates X and y for subset feature selection"""
    
    labels = []
    for i in range(1, nar + 1):
        labels.append(y_name + '-lag' + str(i))
    for i in range(1, nma + 1):
        labels.append('error-lag' + str(i))
    
    def select_ar_lag(y):
        # Search for best AR lag based on AIC
        best_lag = 0
        best_score = None
        current_lag = 0
        len_y = len(y)

        while True:
            current_score = AutoReg(y, lags=current_lag, trend='c').fit().aic
            if best_score is None or current_score < best_score:
                best_score = current_score
                best_lag = current_lag
            elif current_lag - best_lag > 10 or current_lag > len_y:
                break
            current_lag += 1

        return best_lag
    
    k = select_ar_lag(y)
    model = AutoReg(y, lags=k, trend='c').fit()
    
    # Fill starting values with NaN
    resid = np.empty(len(y))
    resid[:] = np.nan
    resid[k:] = model.resid
    
    X = np.empty((len(y), nar + nma + 1))
    X[:, 0] = y
    
    Ys = pd.Series(y)
    for i in range(nar):
        X[:, i + 1] = Ys.shift(i+1)
    
    resids = pd.Series(resid)
    for j in range(nma):
        X[:, j + nar + 1] = resids.shift(j+1)
    
    # Remove NA rows
    X = X[~np.isnan(X).any(axis=1)]

    # Reselect y and X from the filtered rows
    y = X[:, 0]
    X = X[:, 1:]
    
    # Add labels
    X = pd.DataFrame(X, columns=labels)
    
    return X, y


def _exhaustive_best_subset(X, y, nbest=8, score="bic"):
    """Returns the n best scoring feature subsets, using an exhaustive search."""
    
    assert score in ["bic", "aic"], "Unknown score"
    assert nbest > 0, "nbest must be positive"
    
    def score_iterable():
        def get_bic(feature_subset):
            return OLS(y, add_constant(X[list(feature_subset)])).fit().bic
        
        def get_aic(feature_subset):
            return OLS(y, add_constant(X[list(feature_subset)])).fit().aic
        
        get_score = get_bic if score == "bic" else get_aic

        # Recipe from https://docs.python.org/3/library/itertools.html#itertools-recipes
        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        for k in powerset(X.columns):
            yield get_score(k), ('(Intercept)', *k)
            
    return heapq.nsmallest(nbest, score_iterable())


def _forward_best_subset(X, y, nbest=8, beamwidth=40, score="bic"):
    """Returns the n best scoring feature subsets, using forward search."""
    
    assert nbest > 0, "nbest must be positive"
    beamwidth = max(beamwidth, nbest)
    
    def get_bic(feature_subset):
        return -OLS(y, add_constant(X[list(feature_subset)])).fit().bic

    def get_aic(feature_subset):
        return -OLS(y, add_constant(X[list(feature_subset)])).fit().aic

    get_score = get_bic if score == "bic" else get_aic
    
    features = X.columns
    
    heap = []
    visited = set()
    
    def get_pair(k):
        return get_score(k), k
    
    k = ()
    heapq.heappush(heap, get_pair(k))
    
    while True:
        modified = False
        min_score = heap[0][0]
        for _, k in heap:
            for f in features:
                if f in k:
                    continue
                candidate_features = tuple(sorted([*k, f]))
                if candidate_features in visited:
                    continue
                visited.add(candidate_features)
                new_pair = get_pair(candidate_features)
                if new_pair[0] > min_score:
                    modified = True
                    heapq.heappush(heap, get_pair(candidate_features))
                    if len(heap) > beamwidth:
                        heapq.heappop(heap)
                        min_score = heap[0][0]
        if not modified:
            break
            
    return heapq.nsmallest(nbest, [(-x, ('(Intercept)', *y)) for x, y in heap])


def _backward_best_subset(X, y, nbest=8, beamwidth=40, score="bic"):
    """Returns the n best scoring feature subsets, using forward search."""
    
    assert nbest > 0, "nbest must be positive"
    beamwidth = max(beamwidth, nbest)
    
    def get_bic(feature_subset):
        return -OLS(y, add_constant(X[list(feature_subset)])).fit().bic

    def get_aic(feature_subset):
        return -OLS(y, add_constant(X[list(feature_subset)])).fit().aic

    get_score = get_bic if score == "bic" else get_aic
    
    features = X.columns
    
    heap = []
    visited = set()
    
    def get_pair(k):
        return get_score(k), k
    
    k = tuple(features)
    heapq.heappush(heap, get_pair(k))
    
    while True:
        modified = False
        min_score = heap[0][0]
        for _, k in heap:
            for f in features:
                if f not in k:
                    continue
                candidate_features = tuple([x for x in k if x != f])
                if candidate_features in visited:
                    continue
                visited.add(candidate_features)
                new_pair = get_pair(candidate_features)
                if new_pair[0] > min_score:
                    modified = True
                    heapq.heappush(heap, get_pair(candidate_features))
                    if len(heap) > beamwidth:
                        heapq.heappop(heap)
                        min_score = heap[0][0]
        if not modified:
            break
            
    return heapq.nsmallest(nbest, [(-x, ('(Intercept)', *y)) for x, y in heap])


def _plot_results(results, labels, score):
    """ Plot the best selected results in a colormap. """
    
    labels_dict = {k: i for i, k in enumerate(labels)}
    rmin, rmax = min(results)[0], max(results)[0]
    to_color = lambda x : 0.8 * (x - rmax) / (rmin - rmax) + 0.5

    result_matrix = np.zeros((8, len(labels_dict)))
    labels_dict = {k: i for i, k in enumerate(labels)}
    for i, (v, r_labels) in enumerate(results):
        for k in r_labels:
            result_matrix[i, labels_dict[k]] = to_color(v)
    
    plt.figure(figsize=(11, 6))
    plt.imshow(result_matrix, cmap='Greys', aspect='auto')
    plt.xticks(np.arange(0, len(labels)), labels, rotation=90)
    plt.yticks(np.arange(0, len(results)), [round(x[0]) for x in results])
    plt.ylabel(score)
    plt.show()
    

def armasubsets(y, nar, nma, y_name = "Y", score="bic", exploration="auto", nbest=8, beamwidth=100, plot=True):
    """ Performs a search through subsets of ARMA models, as in R's TSA library armasubsets """
    
    assert score in ["bic", "aic"], "Unknown score method"
    assert exploration in ["auto", "exhaustive", "forward", "backward"], "Unknown exploration method"
    
    if exploration == "auto":
        exploration = "exhaustive" if nar + nma <= 10 else "forward"
    
    X, y = _get_armasubsets_features(y, nar, nma, y_name)
    
    if exploration == "exhaustive":
        results = _exhaustive_best_subset(X, y, score=score, nbest=nbest)
    elif exploration == "forward":
        results = _forward_best_subset(X, y, score=score, nbest=nbest, beamwidth=beamwidth)
    else:
        results = _backward_best_subset(X, y, score=score, nbest=nbest, beamwidth=beamwidth)
    if plot:
        labels = np.r_[["(Intercept)"], X.columns]
        _plot_results(results, labels, score)
    return results
