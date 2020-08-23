setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

source("Titanic.ETL.R")

# library(liquidSVM), configure.args="native /usr/local/cuda")

library(liquidSVM)



liquidSVM.model <- liquidSVM::svm(myFormula, training, max_gamma=25)

#outcomePrediction <- liquidSVM::test(liquidSVM.model, testing)
outcomePrediction <- predict(liquidSVM.model, testing)
str(outcomePrediction)
summary(outcomePrediction)

errors(outcomePrediction)
plotROC(liquidSVM.model ,testing)


# Accuracy   ----

if(length(outcomePrediction) != length(outcomeTest)) print("different lenghts")
if(!identical(levels(outcomePrediction),levels(outcomeTest))) print("not identical levels")

(cm <- confusionMatrix ( data = outcomePrediction, reference = outcomeTest ))

(cmf <- as.list( c(cm[c("overall","byClass")], recursive = TRUE) ))




#view variable importance plot
( mat <- xgb.importance (model = liquidSVM))
xgb.plot.importance (importance_matrix = mat[1:20])

