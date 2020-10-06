setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

source("Titanic.ETL.R")

setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

# Load libraries
library(mlbench)
library(caret)
library(caretEnsemble)


nrow(training)
nrow(submitData)

numberPartitions <- 4
crossValidationRepeats <- 500

# Stacking algorithms

control <- trainControl(
  method = "repeatedcv",
  number = numberPartitions,
  repeats = crossValidationRepeats,
  savePredictions = 'final',
  twoClassSummary,
  classProbs = TRUE,
  index = createMultiFolds(training$Survived, numberPartitions, crossValidationRepeats),
  # index = createResample(y = training$Survived, times = 3),
  allowParallel = TRUE
)

algorithmList <-
  c('ranger',
    'xgbTree',
    'rpart',
    #"rf",
    #'glm',
    #'knn',
    'svmRadial',
    'nnet',
    'glmStepAIC')
set.seed(seed)

runTime <- system.time(
  modelsList <-
    caretList(
      myFormula,
      data = training,
      trControl = control,
      methodList = algorithmList,
      continue_on_fail = T
    )
)

library(lubridate)
lapply(runTime, seconds_to_period)$elapsed

# summary(modelsList)
# str(modelsList)

(results <- resamples(modelsList))
summary(results)
dotplot(results)

bwplot(results)

set.seed(seed)
system.time(stackModel <- caretEnsemble(modelsList))
summary(stackModel)
bwplot(stackModel)

# correlation between results
modelCor(results)
splom(results)

# stackControl for caretStack
stackControl <-
  trainControl(
    method = "repeatedcv",
    number = numberPartitions,
    repeats = crossValidationRepeats,
    index = createMultiFolds(training$Survived, numberPartitions, crossValidationRepeats),
    savePredictions = TRUE,
    classProbs = TRUE,
    allowParallel = TRUE
  )


set.seed(seed)
system.time(
  stackModel <-
    caretStack(
      modelsList,
      method = "glmStepAIC",
      metric = "Accuracy",
      trControl = stackControl
    )
)
print(stackModel)

set.seed(seed)
system.time(
  stackModel <-
    caretStack(
      modelsList,
      method = "nnet",
      metric = "Accuracy",
      trControl = stackControl
    )
)
print(stackModel)

# stack using random forest
set.seed(seed)
system.time(
  stackModel <-
    caretStack(
      modelsList,
      method = "rf",
      metric = "Accuracy",
      trControl = stackControl
    )
)
print(stackModel)


# stack using xgbTree
set.seed(seed)
stackModel <-
  caretStack(modelsList,
             method = "xgbTree",
             metric = "Accuracy",
             trControl = stackControl)
print(stackModel)
summary(stackModel)
plot(stackModel)


# stack using xgbDART
set.seed(seed)
system.time(
  stackModel <-
    caretStack(
      modelsList,
      method = "xgbDART",
      metric = "Accuracy",
      trControl = stackControl
    )
)
print(stackModel)
summary(stackModel)
plot(stackModel)



pred = predict(stackModel, testing)
(cm = confusionMatrix(testing$Survived, pred))

# prepare submission

(submitData$predictedSurvived <- predict(stackModel, submitData))


(submitData$PassengerId <- as.numeric(rownames(submitData)))
(submitData$predictedSurvived <-
    ifelse(submitData$predictedSurvived == 'Y', 1, 0))


print(data.frame(submitData$PassengerId, submitData$predictedSurvived))

write.csv(
  data.frame(
    PassengerId = submitData$PassengerId,
    Survived = submitData$predictedSurvived
  ),
  file = "TitanicSubmission-2020-11-06-22.csv",
  row.names = FALSE,
  quote = FALSE
)



source("D:\\Dropbox\\Eclipse\\R\\caret\\stopCluster.R")
