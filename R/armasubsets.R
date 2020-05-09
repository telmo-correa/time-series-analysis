library("leaps")

armasubsets <- function (y, nar, nma, y.name = "Y", ar.method = "ols", ...) {
    zlag <- function(x, d=1) {
        if (d != as.integer(d) || d < 0) 
            stop("d must be a non-negative integer")
        if (d == 0) 
            return(x)
        else return(c(rep(NA, d), rev(rev(x)[-(1:d)])))
    }
    
    lab = NULL
    if (nar > 1) 
        lab = c(lab, paste(y.name, 1:nar, sep = "-lag"))
    if (nma > 1) 
        lab = c(lab, paste("error", 1:nma, sep = "-lag"))
    res.ar = ar(y, method = ar.method)
    resid = res.ar$resid
    x = NULL
    if (nar > 1) {
        for (i in 1:nar) {
            x = cbind(x, zlag(y, d = i))
        }
    }
    if (nma > 1) {
        for (j in 1:nma) {
            x = cbind(x, zlag(resid, d = j))
        }
    }
    x = na.omit(cbind(y, x))
    y = x[, 1]
    x = x[, -1]
    x = data.frame(x)
    colnames(x) = lab
    regobj = regsubsets(y ~ ., data = x, ...)
    class(regobj) = c("armasubsets", "regsubsets")
    invisible(regobj)
}