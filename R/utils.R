require(ggplot2)
require(zoo)
require(tseries)

palette = c('#e2d9e2', '#b3c6ce', '#7ba1c2', '#6276ba',
 '#5e43a5', '#4e186f', '#2f1436', '#581647',
 '#8e2c50', '#b25652', '#c68a6d', '#d4bcac')

plot_monthly <- function(x, y, xlabel, ylabel) {
  ggplot() +
    geom_line(aes(x=x, y=y), color='blue', alpha=0.2) +
    geom_text(aes(x=x, y=y, label=format(data[['date']], "%b")), color=palette[as.numeric(format(data[['date']], "%m"))]) +
    xlab(xlabel) + ylab(ylabel) +
    theme_bw() + theme(text = element_text(size=16))
}
