prewhiten <- function (x, y, x.model = ar.res, ylab = "CCF", ...) {
    filter.mod = function(x, model) {
        if (length(model$Delta) >= 1) 
            x = stats::filter(x, filter = c(1, -model$Delta), method = "convolution", sides = 1)
        if (length(model$theta) >= 1 && any(model$theta != 0)) 
            x = stats::filter(x, filter = -model$theta, method = "recursive", sides = 1)
        if (length(model$phi) >= 1 && any(model$phi != 0)) 
            x = stats::filter(x, filter = c(1, -model$phi), method = "convolution", sides = 1)
        x
    }
    if (!missing(x.model)) {
        x = filter.mod(x, model = x.model$model)
        y = filter.mod(y, model = x.model$model)
    }
    else {
        ar.res = ar.ols(x, ...)
        x = stats::filter(x, filter = c(1, -ar.res$ar), method = "convolution", sides = 1)
        y = stats::filter(y, filter = c(1, -ar.res$ar), method = "convolution", sides = 1)
    }
    ccf.xy = ccf(x = x, y = y, na.action = na.omit, ylab = ylab, ...)
    invisible(list(ccf = ccf.xy, model = x.model))
}