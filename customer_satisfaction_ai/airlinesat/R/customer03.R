#!/usr/bin/Rscript

#install.packages(c('ggplot2','tidyr','devtools'))
library("PCAmixdata")
library("readr")
data1<- read_csv("airlineR.csv",show_col_types= FALSE)
head(data1,5)

rows=rownames(data1)
rows=colnames(data1)

split <- splitmix(data1)
X1 <- split$X.quanti
X2 <- split$X.quali
res.pcamix <- PCAmix(X.quanti = X1, X.quali = X2, rename.level = TRUE, graph = FALSE)
res.pcamix$eig

#reso<- 1200
#length <-3.25*reso/72
length <-480

jpeg("eigs_map001.jpeg",width=length,height=length)
plot(res.pcamix, choice = "ind", coloring.ind = NULL, label = FALSE,
posleg = "bottomright", main = "(a) Observations")
dev.off()

jpeg("eigs002.jpeg",width=length,height=length)
plot(res.pcamix, choice = "levels", xlim = c(-1.5,2.5), main = "(b) Levels")
dev.off()

jpeg("eigs003.jpeg",width=length,height=length)
plot(res.pcamix,choice = "cor", main = "(c) Numerical variables")
dev.off()

jpeg("eigs004.jpeg",width=length,height=length)
plot(res.pcamix, choice = "sqload", coloring.var = T, leg = TRUE,
posleg = "topright", main = "(d) All variables")
dev.off()
