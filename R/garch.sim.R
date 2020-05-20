garch.sim <- function (alpha, beta, n = 100, rnd = rnorm, ntrans = 100, ...) {
    #
    # simulate a GARCH process
    # input:
    # alpha is the vector of ARCH coefficients including omega as the
    # first element.
    # beta is the vector GARCH coefficients
    # n=sample size
    # ntrans=number of transient data
    # rnd=random number generator function. It first two arguments should be
    # sample size and sd (standard deviation) (default=1). Default=normal random
    # generator.
    # 
    # output:
    # the garch time series of size n
    #
    # 
    if (!missing(beta)) 
        p = length(beta)
    else p = 0
    if (!missing(alpha)) 
        q = length(alpha) - 1
    else stop("beta is missing!")
    if (q == 0) 
        stop("Check model: q=0!")
    total.n = n + ntrans
    e = rnd(total.n, ...)
    x = double(total.n)
    sigt = x
    d = max(p, q)
    sigma2 = sum(alpha[-1])
    if (p > 0) 
        sigma2 = sigma2 + sum(beta)
    if (sigma2 > 1) 
        stop("Check model: it does not have finite variance")
    sigma2 = alpha[1]/(1 - sigma2)
    if (sigma2 <= 0) 
        stop("Check model: it does not have positive variance")
    x[1:d] = rnd(d, sd = sqrt(sigma2))
    sigt[1:d] = sigma2
    if (p == 0) {
        for (i in (d + 1):total.n) {
            sigt[i] = sum(alpha * c(1, x[i - (1:q)]^2))
            x[i] = e[i] * sqrt(sigt[i])
        }
    }
    else {
        for (i in (d + 1):total.n) {
            sigt[i] = sum(alpha * c(1, x[i - (1:q)]^2)) + sum(beta * 
                sigt[i - (1:p)])
            x[i] = e[i] * sqrt(sigt[i])
        }
    }
    return(invisible(x[(ntrans + 1):total.n]))
}