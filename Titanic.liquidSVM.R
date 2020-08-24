setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

source("Titanic.ETL.R")

# library(liquidSVM), configure.args="native /usr/local/cuda")

library(liquidSVM)



liquidSVM.model <- liquidSVM::svm(myFormula, training, max_gamma=25)

str(liquidSVM.model)
summary(liquidSVM.model)

#outcomePrediction <- liquidSVM::test(liquidSVM.model, testing)
outcomePrediction <- predict(liquidSVM.model, testing)

errors(outcomePrediction)
plotROC(liquidSVM.model ,testing)


# Accuracy   ----

if(length(outcomePrediction) != length(outcomeTest)) print("different lenghts")
if(!identical(levels(outcomePrediction),levels(outcomeTest))) print("not identical levels")

(cm <- confusionMatrix ( data = outcomePrediction, reference = outcomeTest ))

(cmf <- as.list( c(cm[c("overall","byClass")], recursive = TRUE) ))


# variable importance plot

varImp(liquidSVM.model)
plot(liquidSVM.model)

