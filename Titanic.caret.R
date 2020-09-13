setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

source("Titanic.ETL.R")

library(caret)

str(training)

# caret train

fitControl <- caret::trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,   # partitions
  repeats = 15, ## repeat ten times
  index=createResample(training$Survived, 10),
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  verbose = FALSE,
  allowParallel = TRUE
  , savePredictions = "final")


tune.grid <- expand.grid(
                         nrounds = c(50, 75, 100),
                         max_depth = 6:15,
                         eta = c(0.05, 0.075, 0.9),
                         gamma = 0,
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         min_child_weight = c(2.0, 2.25, 2.5),
                         subsample = c( 0.5, 0.7, 0.9)
                         )


set.seed(1111)
system.time(
  gbmFit1 <- caret::train(
      myFormula,
      data = training,
      method = "xgbTree",
      # method = "glmStepAIC",
      metric='ROC',
      trControl = fitControl,
      # preProcess = "bagImpute",
      verbose = FALSE
     , tuneGrid = tune.grid
  )
)

summary(gbmFit1)

varImp(gbmFit1)
plot(varImp(gbmFit1, scale = F))


predValues <- predict(gbmFit1, testing)

(cm <- confusionMatrix( as.factor(predValues), testing$Survived ))
$overall[1]


library(pROC)
auc <- roc(as.numeric(testing$SurvivedFactor), as.numeric(predValues))
print(auc$auc)

(testing$predictedSurvived <- predValues)

confusionMatrix( predValues, testing$Survived ) # $overall[1]


