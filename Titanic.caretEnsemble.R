setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

source("Titanic.ETL.R")

setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)


nrow(training)
nrow(submitData)

seed <- 222

# Stacking algorithms

control <- trainControl(
  method="repeatedcv"
  , number=10
  , repeats=20
  , savePredictions='final'
  , twoClassSummary
  , classProbs=TRUE
  , index=createResample(training$Survived, 20)
  , allowParallel = TRUE)

algorithmList <- c('xgbLinear','xgbTree','lda', 'rpart', 'glm', 'knn', 'svmRadial', 'nnet', 'glmStepAIC')
set.seed(seed)
modelsList <- caretList(myFormula, data=training, trControl=control, methodList=algorithmList, continue_on_fail = T)
results <- resamples(modelsList)
summary(results)
dotplot(results)

stackModel <- caretEnsemble(modelsList)
summary(stackModel)
varImp(greedy_ensemble)

# correlation between results
modelCor(results)
splom(results)

# stack using glm
stackControl <- trainControl(method="repeatedcv", number=10, repeats=10, savePredictions=TRUE, classProbs=TRUE)

set.seed(seed)
stackModel <- caretStack(modelsList, method="glm", metric="Accuracy", trControl=stackControl)
print(stackModel)

# stack using random forest
set.seed(seed)
stackModel <- caretStack(modelsList, method="rf", metric="Accuracy", trControl=stackControl)
print(stackModel)


# stack using xgbTree
set.seed(seed)
stackModel <- caretStack(modelsList, method="xgbTree", metric="Accuracy", trControl=stackControl)
print(stackModel)
summary(stackModel)
dotplot(stackModel)


# stack using xgbDART
set.seed(seed)
stackModel <- caretStack(modelsList, method="xgbDART", metric="Accuracy", trControl=stackControl)
print(stackModel)
summary(stackModel)
plot(stackModel)



pred = predict(stackModel, testing)
(cm = confusionMatrix(testing$Survived, pred))

# prepare submition

(submitData$predictedSurvived <- predict(stackModel, submitData))


(submitData$PassengerId <- as.numeric(rownames(submitData)))
(submitData$predictedSurvived <- ifelse( submitData$predictedSurvived == 'Y', 0,1))


print(data.frame( submitData$PassengerId, submitData$predictedSurvived ))

write.csv( data.frame(PassengerId = submitData$PassengerId, Survived = submitData$predictedSurvived)
           , file = "TitanicSubmission-2020-09-13-21.csv"
           , row.names=FALSE, quote=FALSE)



source("D:\\Dropbox\\Eclipse\\R\\caret\\stopCluster.R")
