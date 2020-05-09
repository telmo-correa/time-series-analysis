eacf <- function (z,ar.max=7,ma.max=13) {
    #
    #  PROGRAMMED BY K.S. CHAN, DEPARTMENT OF STATISTICS AND ACTUARIAL SCIENCE,
    #  UNIVERSITY OF IOWA.
    #
    #  DATE: 4/2001
    #  Compute the extended sample acf (ESACF) for the time series stored in z.
    #  The matrix of ESACF with the AR order up to ar.max and the MA order
    #  up to ma.max is stored in the matrix EACFM.
    #  The default values for NAR and NMA are 7 and 13 respectively.
    #  Side effect of the eacf function:
    #  The function prints a coded ESACF table with
    #  significant values denoted by * and nosignificant values by 0, significance
    #  level being 5%.
    #
    #  Output:
    #   eacf=matrix of esacf
    #   symbol=matrix of coded esacf
    #
    
    lag1 <- function(z, lag = 1) {
        c(rep(NA, lag), z[1:(length(z) - lag)])
    }
    reupm <- function(m1, nrow, ncol) {
        k <- ncol - 1
        m2 <- NULL
        for (i in 1:k) {
            i1 <- i + 1
            work <- lag1(m1[, i])
            work[1] <- -1
            temp <- m1[, i1] - work * m1[i1, i1]/m1[i, i]
            temp[i1] <- 0
            m2 <- cbind(m2, temp)
        }
        m2
    }
    ceascf <- function(m, cov1, nar, ncol, count, ncov, z, zm) {
        result <- 0 * seq(1, nar + 1)
        result[1] <- cov1[ncov + count]
        for (i in 1:nar) {
            temp <- cbind(z[-(1:i)], zm[-(1:i), 1:i]) %*% c(1, 
                -m[1:i, i])
            result[i + 1] <- acf(temp, plot = FALSE, lag.max = count, 
                drop.lag.0 = FALSE)$acf[count + 1]
        }
        result
    }
    ar.max <- ar.max + 1
    ma.max <- ma.max + 1
    nar <- ar.max - 1
    nma <- ma.max
    ncov <- nar + nma + 2
    nrow <- nar + nma + 1
    ncol <- nrow - 1
    z <- z - mean(z)
    zm <- NULL
    for (i in 1:nar) zm <- cbind(zm, lag1(z, lag = i))
    cov1 <- acf(z, lag.max = ncov, plot = FALSE, drop.lag.0 = FALSE)$acf
    cov1 <- c(rev(cov1[-1]), cov1)
    ncov <- ncov + 1
    m1 <- matrix(0, ncol = ncol, nrow = nrow)
    for (i in 1:ncol) m1[1:i, i] <- ar.ols(z, order.max = i, 
        aic = FALSE, demean = FALSE, intercept = FALSE)$ar
    eacfm <- NULL
    for (i in 1:nma) {
        m2 <- reupm(m1 = m1, nrow = nrow, ncol = ncol)
        ncol <- ncol - 1
        eacfm <- cbind(eacfm, ceascf(m2, cov1, nar, ncol, i, 
            ncov, z, zm))
        m1 <- m2
    }
    work <- 1:(nar + 1)
    work <- length(z) - work + 1
    symbol <- NULL
    for (i in 1:nma) {
        work <- work - 1
        symbol <- cbind(symbol, ifelse(abs(eacfm[, i]) > 2/work^0.5, 
            "x", "o"))
    }
    rownames(symbol) <- 0:(ar.max - 1)
    colnames(symbol) <- 0:(ma.max - 1)
    cat("AR/MA\n")
    print(symbol, quote = FALSE)
    invisible(list(eacf = eacfm, ar.max = ar.max, ma.ma = ma.max, 
        symbol = symbol))
}