import numpy as np
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg

import matplotlib.pyplot as plt


def lagplot(x, n_lags=6, method="lowess"):
    assert method in ["lowess", "glm", "both"], "method must be lowess, glm, or both"
    
    def get_glm_fit(x_endog, x_exog, n_steps=501):
        x_min, x_max = min(x_exog), max(x_exog)
        model_fit = sm.GLM(x_endog, sm.add_constant(x_exog)).fit()

        x_pred = (np.arange(n_steps) / n_steps) * (x_max - x_min) + x_min
        y_pred = model_fit.predict(sm.add_constant(x_pred))

        return x_pred, y_pred

    def get_loc_fit(x_endog, x_exog):
        res = sm.nonparametric.lowess(x_endog, x_exog, return_sorted=True)
        return res[:, 0], res[:, 1]
    
    n_half = int(np.ceil(n_lags/2))
    
    plt.figure(figsize=(8, 4 * n_half))
    for k in range(1, n_lags+1):
        x_endog, x_exog = x[k:], x[:-k]
        
        ax = plt.subplot(n_half, 2, k)
        ax.set_title('lag-' + str(k) + ' regression plot')
        ax.plot(x_exog, x_endog, linestyle='none', marker='.')
        
        if method in ["glm", "both"]:
            x_pred, y_pred = get_glm_fit(x_endog, x_exog)
            ax.plot(x_pred, y_pred, color='darkgreen')
        
        if method in ["lowess", "both"]:
            x_pred, y_pred = get_loc_fit(x_endog, x_exog)
            ax.plot(x_pred, y_pred, color='red')

    plt.tight_layout()
    plt.show()