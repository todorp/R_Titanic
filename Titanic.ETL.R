# Initialisation   ----

rm(list = ls())
gc(reset = TRUE)

source("D:\\Dropbox\\Eclipse\\R\\libTests\\doParallel\\startCluster.R")


library(reshape)
library(caret)
require( gtools)


# ETL  ----

trainData <- read.csv("D:\\Dropbox\\Eclipse\\R\\Titanic\\train.csv", stringsAsFactors =  F)
submitData <- read.csv("D:\\Dropbox\\Eclipse\\R\\Titanic\\test.csv", stringsAsFactors =  F)
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

str(slimData)

# dumming vars   ----

dummies <- dummyVars( myFormula, data = slimData)
dummyVarData <- as.data.frame( predict(dummies, newdata = slimData))
names(dummyVarData)
length( names(dummyVarData) )

lapply(dummyVarData,  class)

# automatic imputation   ----

processImpute     <- preProcess( dummyVarData, method = c( "center",  "nzv",  "scale", "knnImpute") )
#processImpute     <- preProcess( dummyVarData, method = c("center",  "nzv",  "scale",  "bagImpute"))
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

library(xgboost)

dtrain <- xgb.DMatrix( data = as.matrix( training[, - which( names(training) %in% c("Survived"))] ), label = as.matrix( training$Survived) )
dtest <- xgb.DMatrix( data = as.matrix( testing[, - which( names(testing) %in% c("Survived"))] ), label = as.matrix( testing$Survived) )
dSubmit <- xgb.DMatrix( data = as.matrix( submitData[, - which( names(submitData) %in% c("Survived"))] ), label = as.matrix( submitData$Survived) )


wl <- list(train = dtrain, test = dtest)
getinfo( wl$test, "label")


