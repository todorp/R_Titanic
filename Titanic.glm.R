## Load cleaned data
source("D:\\Dropbox\\Eclipse\\R\\Titanic\\titanicETL01.R")

library(reshape)
library(caret)


system.time(fit.8 <-
              glm(myFormula, data = training, family = binomial(("logit"))))
summary(fit.8)
(p <- predict(fit.8, testing))
p[which(p > 0.5)] <- 1
p[which(p <= 0.5)] <- 0

class(testing$Survived)
class(p)

confusionMatrix(as.factor(p), testing$Survived)$overall[1]

summary(fit.8)

varImp(fit.8)
plot(varImp(fit.8, scale = F))

(submitData$predictedSurvived <- predict(fit.8, submitData))

print(data.frame(c(
  submitData$PassengerId,
  ifelse(submitData$predictedSurvived == 'Yes', 1, 0)
)))

write.csv(
  data.frame(
    submitData$PassengerId,
    ifelse(submitData$predictedSurvived == 'Yes', 1, 0)
  ) ,
  file = "D:\\Dropbox\\Eclipse\\R\\Titanic\\Submission05.csv",
  row.names = FALSE,
  quote = FALSE,
  col.names = c('PassengerId', 'SurvivedFactor')
)

# caret train

fitControl <- caret::trainControl(
  ## 10-fold CV
  method = "repeatedcv",
  number = 10,
  # partitions
  repeats = 5,
  ## repeate ten times
  index = createResample(training$SurvivedFactor, 10),
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  # preProcOptions = c(cutoff = 0.97, freqCut = 95/5, uniqueCut = 10),
  verbose = FALSE,
  allowParallel = TRUE
)


tune.grid <- expand.grid(
  eta = c(0.05, 0.075, 0.1),
  nrounds = c(50, 75, 100),
  max_depth = 6:8,
  min_child_weight = c(2.0, 2.25, 2.5),
  colsample_bytree = c(0.3, 0.4, 0.5),
  gamma = 0,
  subsample = 1
)


set.seed(825)
system.time(
  gbmFit1 <- caret::train(
    myFormula,
    data = training,
    preProcess = c("center", "scale"   , "pca"),
    method = "glmStepAIC",
    metric = 'ROC',
    trControl = fitControl,
    verbose = FALSE
    # , tuneGrid = tune.grid
  )
)

summary(gbmFit1)

varImp(gbmFit1)
plot(varImp(gbmFit1, scale = F))


predValues <- predict(gbmFit1, testing)
# p[which(p > 0.5)] <- 1
# p[which(p <= 0.5)] <- 0

# class(testing$SurvivedFactor)
# class(predValues)

confusionMatrix(as.factor(predValues), testing$SurvivedFactor)$overall[1]


library(pROC)
auc <-
  roc(as.numeric(testing$SurvivedFactor), as.numeric(predValues))
print(auc$auc)

(testing$predictedSurvived <- predValues)

confusionMatrix(testing$predictedSurvived, testing$SurvivedFactor)$overall[1]


(slimData$predictedSurvived <- predict(gbmFit1, slimData))

(submitData <- slimData[which(is.na(slimData$Survived)),])
s <- submitData$PassengerId
(p <- ifelse(submitData$predictedSurvived == 'Yes' , 1, 0))

print(data.frame(c(s, p)))

write.csv(
  data.frame(
    submitData$PassengerId,
    ifelse(submitData$predictedSurvived == 'Yes', 1, 0)
  ) ,
  file = "D:\\Dropbox\\Eclipse\\R\\Titanic\\Submission06.csv",
  row.names = FALSE,
  quote = FALSE,
  col.names = c('PassengerId', 'SurvivedFactor')
)



# ensembles


fitControl <- trainControl(
  method = 'repeatedcv',
  number = 3,
  repeats = 5,
  savePredictions = 'final',
  classProbs = TRUE,
  index = createResample(training$SurvivedFactor, 10),
  search = 'grid',
  summaryFunction = twoClassSummary,
  verbose = FALSE,
  allowParallel = TRUE
)

# (myMethods <- names(getModelInfo()))
myMethods <-
  c('glm',
    'glmnet',
    'rpart' ,
    'xgbLinear',
    'xgbTree',
    'rf',
    "nnet",
    "svmRadial")

myMethods <- c('lmStepAIC', 'glmStepAIC')

library('rpart')
library('caretEnsemble')
system.time(
  model_list <- caretList(
    myFormula,
    data = training,
    metric = 'ROC',
    trControl = fitControl,
    methodList = myMethods,
    preProcess = c("center", "scale"),
    continue_on_fail = TRUE,
    verbose = FALSE
  )
)

model_list


(predictionClass <- predict(model_list, newdata = testing))

modelCor(resamples(model_list))

output = resamples(model_list)
summary(output)
dotplot(output)

# Model Correlation

models_results <- resamples(model_list)

modelCor(models_results)

dotplot(models_results)

greedy_ensemble <- caretEnsemble(model_list)

summary(greedy_ensemble)
plot(varImp(greedy_ensemble, scale = F))

predValues <- predict(greedy_ensemble , newdata = testing)
ifelse(predValues > 0.5,  'Yes' , 'No')


confusionMatrix(as.factor(predValues), testing$SurvivedFactor)$overall[1]

###################


## Control for the train function
# fitControl <- trainControl(method="cv",
#                             number=10,
#                             savePredictions = "final",
#                             classProbs=TRUE,
#                             index=createResample(training$SurvivedFactor, times = 15),
#                             summaryFunction = twoClassSummary,
#                             verboseIter = FALSE)



library(pROC)
library(randomForest)
library(nnet)
library(kernlab)

set.seed(100)
# system.time(model_list <- caretList(myFormula,
#                         data = training,
#                         trControl = fitControl,
#                         metric = "ROC",
#                         tuneList = list(knnr = caretModelSpec(method = "knn", tuneLength = 3),
#                                         nnet = caretModelSpec(method = "nnet", tuneLength = 3, trace=FALSE),
#                                         svmR = caretModelSpec(method = "svmRadial", tuneLength = 3),
#                                         rdmf = caretModelSpec(method = "rf", tuneLength = 3))))
#
# # Model Correlation
#
# models_results <- resamples(model_list)
#
# modelCor( models_results )
#
# dotplot( models_results )
#
# greedy_ensemble <- caretEnsemble(model_list)
#
#
# summary(greedy_ensemble)
#
# predValues <- predict( greedy_ensemble , newdata=testing )
# # predict( greedy_ensemble, newdata =testing)
#
# summary(greedy_ensemble)
# varImp(greedy_ensemble)
# plot(varImp(greedy_ensemble, scale = F))

#   caretStack


stack = caretStack(model_list, method = "glm", trControl = fitControl)
summary(stack)
varImp(stack)

pred = predict(stack, testing)
cm = confusionMatrix(testing$SurvivedFactor, pred)
print(cm)
