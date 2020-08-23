

setwd("D:\\Dropbox\\Eclipse\\R\\Kagle\\Titanic")

source("Titanic.ETL.R")

param <-
  list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = 0.1,
    gamma = 0,
    max_depth = 15,
    min_child_weight = 1.05,
    subsample = 0.7,
    colsample_bytree = 0.5,
    nthread = 7,
    nrounds = 150,
    mySeed  = 4321,
    gpu_id  = 0,
    # tree_method = 'gpu_hist'
    tree_method = 'hist'
  )


set.seed(param$myParams$mySeed)

(xgbcv <- xgb.cv( data = dtrain
                  , params = param
                  , nrounds = param$nrounds
                  , nfold = 10
                  , missing = NA
                  , metrics = list( "auc") # , "auc", "error")
                  , showsd = T
                  , stratified = T
                  , print_every_n = 10
                  , early_stopping_rounds = param$nrounds
                  , maximize = F))


param$nrounds <- xgbcv$best_iteration
set.seed(param$myParams$mySeed)

system.time(xgbModel <- xgb.train(
  params    = param,
  data      = dtrain,
  label     = getinfo( wl$test, "label"),
  watchlist = wl,
  nrounds   = param$nrounds,
  metrics = list( "auc" ), #, "auc", "error"),
  verbose   = 0))

str(xgbModel)
summary(xgbModel)

outcomePrediction <- predict (xgbModel,wl$test)
outcomePrediction <- as.factor( as.integer(  ifelse (outcomePrediction > 0.5,1,0) ) )
outcomeTest <- as.factor(  getinfo( wl$test, "label") )


if(length(outcomePrediction) != length(outcomeTest)) print("different lenghts")
if(!identical(levels(outcomePrediction),levels(outcomeTest))) print("not identical levels")

(cm <- confusionMatrix ( data = outcomePrediction, reference = outcomeTest ))

(cmf <- as.list( c(cm[c("overall","byClass")], recursive = TRUE) ))


# Accuracy   ----


#view variable importance plot
( mat <- xgb.importance (model = xgbModel))
xgb.plot.importance (importance_matrix = mat[1:20])

# prepare submition

(submitData$predictedSurvived <- predict(xgbModel, dSubmit))


(submitData$PassengerId <- as.numeric(rownames(submitData)))
(submitData$predictedSurvived <- ifelse( submitData$predictedSurvived > 0.5, 1,0))


print(data.frame( submitData$PassengerId, submitData$predictedSurvived ))

write.csv( data.frame(PassengerId = submitData$PassengerId, Survived = submitData$predictedSurvived)
           , file = "TitanicSubmission.csv"
           , row.names=FALSE, quote=FALSE)
