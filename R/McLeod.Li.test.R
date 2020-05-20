McLeod.Li.test <- function (object, y, gof.lag, col = "red", omit.initial = TRUE,plot=TRUE, 
    ...) {
    BL=function(x,lag){
    cor <- acf(x, lag.max = lag, plot = FALSE, na.action = na.pass)
    n <- sum(!is.na(x))
    obs <- cor$acf[1:lag]
        STATISTIC <- n * (n + 2) * sum(1/seq.int(n - 1, n - lag) * 
            obs^2)
        PVAL <- 1 - pchisq(STATISTIC, lag)
    PVAL}

    if(!missing(object)){
    residuals = residuals(object)
    Delta = c(1, -object$mod$Delta)
    d1 = length(Delta)
    if (omit.initial) 
        residuals = window(residuals, start = time(residuals)[d1 + 
            1])
     } else {
if(missing(y)) stop('supply the time series thru the y-argument') 
residuals=y}
    n=length(residuals)
    if (missing(gof.lag)) 
        lag.max = 10 * log10(n)
    else lag.max = gof.lag
    lbv = rep(NA, lag.max)
    for(i in 1:lag.max) {lbv[i]=BL(residuals^2,lag=i)}
    if(plot) {plot(y = lbv, x = 1:lag.max, ylim = c(0, 1), pch = 21, ylab = "P-value", 
        xlab = "Lag", ...)
    abline(h = 0.05, lty = 2, col = col)}
     invisible(list(p.values=lbv))
}