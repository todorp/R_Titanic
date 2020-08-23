# Initialisation   ----

rm(list = ls())
gc(reset = TRUE)
setwd("D:\\Dropbox\\Eclipse\\R\\caret\\")
source("startCluster.R")

setwd("D:\\Dropbox\\Eclipse\\R\\Titanic\\")
library(reshape)
library(caret)
require( gtools)
library(xgboost)
# ETL  ----

trainData <- read.csv("train.csv", stringsAsFactors =  F)
submitData <- read.csv("test.csv", stringsAsFactors =  F)
submitData$Survived <- NA

rawData<-rbind(trainData, submitData) # Merge data to simplify preprocessing
(row.names(rawData) <- rawData$PassengerId)
lapply(rawData, class)

# Data Cleaning   ----

allData = rawData
allData[ allData == "" ] <- NA

allData$Fare <- ifelse( allData$Fare == 0, NA, allData$Fare)


#   Family   ----

allData$FamilyName<-regmatches(as.character(allData$Name),regexpr("[A-z]{1,20}\\,", as.character(allData$Name)))
allData$FamilyName<-unlist(lapply(allData$FamilyName,FUN=function(x) substr(x, 1, nchar(x)-1)))
table(allData$FamilyName)


allData$FamilyName<-regmatches(as.character(allData$Name),regexpr("[A-z]{1,20}\\,", as.character(allData$Name)))
allData$FamilyName<-unlist(lapply(allData$FamilyName,FUN=function(x) substr(x, 1, nchar(x)-1)))
table(allData$FamilyName)
allData$FamilyName<-as.factor(allData$FamilyName)

FamilyName.count <- aggregate(allData$FamilyName, by=list(allData$FamilyName), function(x) sum( !is.na(x) ))
allData$FamilyNameCount<-apply(allData, 1, function(x) FamilyName.count[which(FamilyName.count[, 1] == x["FamilyName"]), 2])


allData$FamilySize <- allData$SibSp + allData$Parch + 1

# Prices per person

(ticket.count <- aggregate(allData$Ticket, by=list(allData$Ticket), function(x) sum( !is.na(x) )))
( allData$PricePerPerson<-apply(allData, 1, function(x) as.numeric(x["Fare"]) / ticket.count[which(ticket.count[, 1] == x["Ticket"]), 2]) )


allData$TicketCount<-apply(allData, 1, function(x) ticket.count[which(ticket.count[, 1] == x["Ticket"]), 2])


#   Title ----

allData$Title<-regmatches(as.character(allData$Name),regexpr("\\,[A-z ]{1,20}\\.", as.character(allData$Name)))
allData$Title<-unlist(lapply(allData$Title,FUN=function(x) substr(x, 3, nchar(x)-1)))
table(allData$Title)


allData$Title[which(allData$Title %in% c("Mme", "Mlle"))] <- "Miss"
allData$Title[which(allData$Title %in% c("Lady", "Ms", "the Countess", "Dona"))] <- "Mrs"
allData$Title[which(allData$Title=="Dr" & allData$Sex=="female")] <- "Mrs"
allData$Title[which(allData$Title=="Dr" & allData$Sex=="male")] <- "Mr"
allData$Title[which(allData$Title %in% c("Capt", "Col", "Don", "Jonkheer", "Major", "Rev", "Sir"))] <- "Mr"

#   Deck and Cabins ----

allData$Cabin <- trimws( allData$Cabin, which = c("both"))

allData$naCabin <- as.numeric( is.na(allData$Cabin))

allData$naAge <- as.numeric( is.na(allData$Age))

(allData$Cabin <- ifelse( !is.na(allData$Cabin) & substring( allData$Cabin, 1, 2 ) == "F "
                         , substring( allData$Cabin, 3, 1111 )
                         , allData$Cabin))


(allData$Deck <- ifelse( !is.na(allData$Cabin)
                         , substring( allData$Cabin, 1, 1 )
                         , NA))

indexes <- !is.na(allData$Deck)

allData$deckNum[indexes] <- asc( allData$Deck[indexes] )  - asc('A') + 1



# number of cabins per ticket/family

(allData$numCabins <- ifelse( !is.na(allData$Cabin)
          , lengths( strsplit(allData$Cabin, " ") )
          ,NA) )


# unique tickets

# length(unique(allData$Ticket))
# length(allData$Ticket)


# Set factors

allData$Survived <- as.factor(allData$Survived)
allData$Pclass <- as.factor(allData$Pclass)
allData$Sex <- as.factor(allData$Sex)
allData$Embarked <- as.factor(allData$Embarked)
allData$Title <- as.factor(allData$Title)
names(allData)

# features to use

features <- c( "Survived", "Pclass", "Sex", "Age", "SibSp",
               "Parch", "Fare", "PricePerPerson", "TicketCount",
               "Embarked",
               "Title", "FamilyNameCount",
               "FamilySize", "deckNum", "numCabins",
               "naCabin" ,        "naAge")

myFormula <- as.formula( Survived ~ . )

slimData <- as.data.frame( allData[, features] )
# slimData <- slimData
str(slimData)

# dumming vars   ----

dummies <- dummyVars( myFormula, data = slimData)
dummyVarData <- as.data.frame( predict(dummies, newdata = slimData))
names(dummyVarData)
length( names(dummyVarData) )

lapply(dummyVarData,  class)

# automatic imputation   ----

#processImpute     <- preProcess( dummyVarData, method = c( "center",  "nzv",  "scale", "knnImpute"))
processImpute     <- preProcess( dummyVarData, method = c( "bagImpute"))
#processImpute     <- preProcess( dummyVarData, method = "medianImpute")

imputeData        <- as.data.frame( cbind( Survived = slimData$Survived, predict( processImpute, dummyVarData) ) )

lapply(imputeData,  class)

(modelData <- imputeData[which(!is.na(imputeData$Survived)), ])
(submitData <- imputeData[which(is.na(imputeData$Survived)), ])


set.seed(1234)
inTrain<-createDataPartition(modelData$Survived, p = 0.8)[[1]]


(training <- modelData[inTrain,])
(testing <- modelData[-inTrain,])

lapply(training, class)

# training[, - which( names(training) %in% c("Survived"))]


dtrain <- xgb.DMatrix( data = as.matrix( training[, - which( names(training) %in% c("Survived"))] ), label = as.matrix( training$Survived) )
dtest <- xgb.DMatrix( data = as.matrix( testing[, - which( names(testing) %in% c("Survived"))] ), label = as.matrix( testing$Survived) )
dSubmit <- xgb.DMatrix( data = as.matrix( submitData[, - which( names(submitData) %in% c("Survived"))] ), label = as.matrix( submitData$Survived) )


wl <- list(train = dtrain, test = dtest)
getinfo( wl$test, "label")


param <-
  list(
    booster = "gbtree",
    objective = "binary:logistic",
    eta = 0.1,
    gamma = 0,
    max_depth = 12,
    min_child_weight = 1.05,
    subsample = 0.7,
    colsample_bytree = 0.5,
    nthread = 7,
    nrounds = 150,
    mySeed  = 1234,
    gpu_id  = 0,
    # tree_method = 'gpu_hist'
    tree_method = 'hist'
  )

# param <- xdbGridRet$myParams
set.seed(param$myParams$mySeed)

(xgbcv <- xgb.cv( data = dtrain
                  , params = param
                  , nrounds = param$nrounds
                  , nfold = 10
                  , missing = NA
                  , metrics = list( "error") # , "auc", "error")
                  , showsd = T
                  , stratified = T
                  , print_every_n = 10
                  , early_stopping_rounds = param$nrounds
                  , maximize = F))


param$nrounds <- xgbcv$best_iteration
set.seed(param$myParams$mySeed)

system.time(xgb1 <- xgb.train(
  params    = param,
  data      = dtrain,
  label     = getinfo( wl$test, "label"),
  watchlist = wl,
  nrounds   = xgbcv$best_iteration,
  metrics = list( "error" ), #, "auc", "error"),
  verbose   = 0))

xgbpred <- predict (xgb1,wl$test)
xgbpred <- as.integer(  ifelse (xgbpred > 0.5,1,0) )
xgbpred <- as.factor(xgbpred)

xgblabel <- getinfo( wl$test, "label")
xgblabel <- as.factor( xgblabel)

if(length(xgbpred) != length(xgblabel)) print("different lenghts")
if(!identical(levels(xgbpred),levels(xgblabel))) print("different lenghts")

(cm <- confusionMatrix ( data = xgbpred, reference = xgblabel ))

class(cm)
(cmf <- as.list( c(cm[c("overall","byClass")], recursive = TRUE) ))


#Accuracy

xgb1 <- xdbGridRet$xgbModel

#view variable importance plot
( mat <- xgb.importance (model = xgb1))
xgb.plot.importance (importance_matrix = mat[1:20])

# prepare submition

(submitData$predictedSurvived <- predict(xgb1, dSubmit))


(submitData$PassengerId <- as.numeric(rownames(submitData)))
(submitData$predictedSurvived <- ifelse( submitData$predictedSurvived > 0.5, 1,0))


print(data.frame( submitData$PassengerId, submitData$predictedSurvived ))

write.csv( data.frame(PassengerId = submitData$PassengerId, Survived = submitData$predictedSurvived)
           , file = "TitanicSubmission.csv"
           , row.names=FALSE, quote=FALSE)
