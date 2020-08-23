setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

source("Titanic.ETL.R")

# library(liquidSVM), configure.args="native /usr/local/cuda")

library(liquidSVM)



model <- liquidSVM::svm(myFormula, training, GPUs=1)

prediction <- liquidSVM::test(model, testing)
str(prediction)
summary(prediction)

errors(prediction)

plot(training$X1, training$Y,pch='.', ylim=c(-.2,.8), ylab='', xlab='', axes=F)
curve(predict(model, x),add=T,col='red')


# Multi Class

banana <- liquidData('banana-mc')
banana
#> LiquidData "banana-mc" with 4000 train samples and 4000 test samples
#>   having 3 columns named Y,X1,X2
#>   target "Y" factor with 4 levels: 1 (1200 samples) 2 (1200 samples) 3 (800 samples) ...

model <- liquidSVM::svm(Y~., banana$train, GPUs=1 )

plot(banana$train$X1, banana$train$X2,pch='o', col=banana$train$Y, ylab='', xlab='', axes=F)
x <- seq(-1,1,.05)
z <- matrix(predict(model,expand.grid(x,x)),length(x))
contour(x,x,z, add=T, levels=1:4,col=1,lwd=4)

errors(test(model,banana$test))
