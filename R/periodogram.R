spec <- function (x, taper = 0, detrend = FALSE, demean = TRUE, method = c("pgram", 
    "ar"), ci.plot = FALSE, ylim = range(c(lower.conf.band, upper.conf.band)), 
    ...) {
    method <- match.arg(method)
    x = ts(x, frequency = 1)
    if (!ci.plot) {
        if (missing(ylim)) 
            sp = switch(match.arg(method), pgram = spec.pgram(x, 
                taper = taper, detrend = detrend, demean = demean, 
                ...), ar = spec.ar(x, ...))
        else sp = switch(match.arg(method), pgram = spec.pgram(x, 
            taper = taper, detrend = detrend, demean = demean, 
            ylim = ylim, ...), ar = spec.ar(x, ylim = ylim, ...))
        return(sp)
    }
    if (ci.plot == TRUE && method == "ar") 
        stop("Option ci.plot==TRUE is not implemented for the ar method")
    if (ci.plot == TRUE && method == "pgram") {
        sp = spec.pgram(x, taper = taper, detrend = detrend, 
            demean = demean, ylim = ylim, plot = FALSE, ...)
        v = df.kernel(sp$kernel)
        lower.conf.band = sp$spec * v/qchisq(0.025, v)
        upper.conf.band = sp$spec * v/qchisq(0.975, v)
        sp = spec.pgram(x, taper = taper, detrend = detrend, 
            demean = demean, ylim = ylim, ...)
        lines(sp$freq, sp$spec * v/qchisq(0.025, v), lty = "dotted")
        lines(sp$freq, sp$spec * v/qchisq(0.975, v), lty = "dotted")
    }
    sp
}

periodogram <- function (y, log = "no", plot = TRUE, ylab = "Periodogram", xlab = "Frequency", lwd = 2, ...)  {
    if (is.matrix(y) && (dim(y)[2] > 1)) 
        stop("y must be a univariate time series")
    sp = spec(y, log = log, plot = FALSE)
    sp$spec = 2 * sp$spec
    temp=sp$spec[sp$freq==.5]
    sp$spec[sp$freq==.5]=temp/2
    if (plot == TRUE) 
        plot(y = sp$spec, x = sp$freq, type = "h", ylab = ylab, 
            xlab = xlab, lwd = lwd, ...)
    return(invisible(sp))
}