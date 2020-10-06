# Initialisation   ----

rm(list = ls())
gc(reset = TRUE)

source("D:\\Dropbox\\Eclipse\\R\\libTests\\doParallel\\startCluster.R")


library(reshape)
library(caret)
require(gtools)


# ETL  ----

trainData <-
  read.csv("D:\\Dropbox\\Eclipse\\R\\Titanic\\train.csv",
           stringsAsFactors =  F)
submitData <-
  read.csv("D:\\Dropbox\\Eclipse\\R\\Titanic\\test.csv",
           stringsAsFactors =  F)
submitData$Survived <- NA

rawData <-
  rbind(trainData, submitData) # Merge data to simplify preprocessing
(row.names(rawData) <- rawData$PassengerId)
lapply(rawData, class)

# Data Cleaning   ----

allData = rawData
allData[allData == ""] <- NA

allData$Fare <- ifelse(allData$Fare < 1, NA, allData$Fare)


#   Family   ----

allData$FamilyName <-
  regmatches(as.character(allData$Name),
             regexpr("[A-z]{1,20}\\,", as.character(allData$Name)))
allData$FamilyName <-
  unlist(lapply(
    allData$FamilyName,
    FUN = function(x)
      substr(x, 1, nchar(x) - 1)
  ))
table(allData$FamilyName)


allData$FamilyName <-
  regmatches(as.character(allData$Name),
             regexpr("[A-z]{1,20}\\,", as.character(allData$Name)))
allData$FamilyName <-
  unlist(lapply(
    allData$FamilyName,
    FUN = function(x)
      substr(x, 1, nchar(x) - 1)
  ))
table(allData$FamilyName)
allData$FamilyName <- as.factor(allData$FamilyName)

FamilyName.count <-
  aggregate(allData$FamilyName, by = list(allData$FamilyName), function(x)
    sum(!is.na(x)))
allData$FamilyNameCount <-
  apply(allData, 1, function(x)
    FamilyName.count[which(FamilyName.count[, 1] == x["FamilyName"]), 2])


allData$FamilySize <- allData$SibSp + allData$Parch + 1

# Price per person

(ticket.count <-
    aggregate(allData$Ticket, by = list(allData$Ticket), function(x)
      sum(!is.na(x))))
(allData$PricePerPerson <-
    apply(allData, 1, function(x)
      as.numeric(x["Fare"]) / ticket.count[which(ticket.count[, 1] == x["Ticket"]), 2]))


allData$TicketCount <-
  apply(allData, 1, function(x)
    ticket.count[which(ticket.count[, 1] == x["Ticket"]), 2])


#   Title ----

allData$Title <-
  regmatches(as.character(allData$Name),
             regexpr("\\,[A-z ]{1,20}\\.", as.character(allData$Name)))
allData$Title <-
  unlist(lapply(
    allData$Title,
    FUN = function(x)
      substr(x, 3, nchar(x) - 1)
  ))
table(allData$Title)


allData$Title[which(allData$Title %in% c("Mme", "Mlle"))] <- "Miss"
allData$Title[which(allData$Title %in% c("Lady", "Ms", "the Countess", "Dona"))] <-
  "Mrs"
allData$Title[which(allData$Title == "Dr" &
                      allData$Sex == "female")] <- "Mrs"
allData$Title[which(allData$Title == "Dr" &
                      allData$Sex == "male")] <- "Mr"
allData$Title[which(allData$Title %in% c("Capt", "Col", "Don", "Jonkheer", "Major", "Rev", "Sir"))] <-
  "Mr"

#   Age

# title.age<-aggregate(allData$Age,by = list(allData$Title), FUN = function(x) median(x, na.rm = T))
# allData[is.na(allData$Age), "Age"] <- apply(allData[is.na(allData$Age), ] , 1, function(x) title.age[title.age[, 1]==x["Title"], 2])


#   Deck and Cabins ----

allData$Cabin <- trimws(allData$Cabin, which = c("both"))

allData$naCabin <- as.numeric(is.na(allData$Cabin))

allData$naAge <- as.numeric(is.na(allData$Age))

(allData$Cabin <-
    ifelse(
      !is.na(allData$Cabin) & substring(allData$Cabin, 1, 2) == "F "
      ,
      substring(allData$Cabin, 3, 1111)
      ,
      allData$Cabin
    ))


(allData$Deck <- ifelse(!is.na(allData$Cabin)
                        , substring(allData$Cabin, 1, 1)
                        , NA))

indexes <- !is.na(allData$Deck)

allData$deckNum[indexes] <-
  asc(allData$Deck[indexes])  - asc('A') + 1



# number of cabins per ticket/family

(allData$numCabins <- ifelse(!is.na(allData$Cabin)
                             , lengths(strsplit(allData$Cabin, " "))
                             , NA))

(allData$farePerCabin <- ifelse(!is.na(allData$Cabin)
                                , allData$Fare / allData$numCabins
                                , NA))

# unique tickets

# length(unique(allData$Ticket))
# length(allData$Ticket)


# dbWriteTable(con, "TitanicData", value = allData, overwrite  = T )


# Set factors

allData$Survived <- as.factor(allData$Survived)
allData$Pclass <- as.factor(allData$Pclass)
allData$Sex <- as.factor(allData$Sex)
allData$Embarked <- as.factor(allData$Embarked)
allData$Title <- as.factor(allData$Title)
names(allData)


# features to use

features <- c(
  "Survived",
  "Pclass",
  "Sex",
  "Age",
  # "SibSp",
  # "Parch",
  "Fare",
  "PricePerPerson",
  "TicketCount",
  # "Embarked",
  "Title",
  # "FamilyNameCount",
  "FamilySize",
  "Deck",
  # "deckNum",
  "numCabins",
  'farePerCabin',
  "naCabin",
  "naAge"
)



slimData <- as.data.frame(allData[, features])

str(slimData)

# dumming vars   ----

slimData$Survived <- as.factor(slimData$Survived)
myFormula <- as.formula(Survived ~ .)

dummies <- dummyVars(myFormula, data = slimData)
dummyVarData <- as.data.frame(predict(dummies, newdata = slimData))

names(dummyVarData)
length(names(dummyVarData))

lapply(dummyVarData,  class)

# automatic imputation   ----


str(dummyVarData)

summary(dummyVarData)

processImpute     <-
  preProcess(dummyVarData,
             method = c("nzv", "center", "corr", "scale", "knnImpute"))
# processImpute     <- preProcess( dummyVarData, method = c( "nzv", "center", "corr", "scale", "bagImpute" ) )
# processImpute     <- preProcess( dummyVarData, method = c( "zv", "center", "corr", "scale", "medianImpute" ) )


imputeData        <-
  as.data.frame(cbind(
    Survived = ifelse(slimData$Survived == 1, "Y", "N")
    ,
    predict(processImpute, dummyVarData)
  ))
imputeData$Survived <- as.factor(imputeData$Survived)

lapply(imputeData,  class)

(modelData <- imputeData[which(!is.na(imputeData$Survived)),])
(submitData <- imputeData[which(is.na(imputeData$Survived)),])

length(modelData[, 1]) / length(submitData[, 1])

seed <- 217

set.seed(seed)
inTrain <- createDataPartition(modelData$Survived, p = 2 / 3)[[1]]


(training <- modelData[inTrain, ])
# training <- modelData
(testing <- modelData[-inTrain, ])


# training <- modelData
summary(training)

# training[, - which( names(training) %in% c("Survived"))]

# outcomeTest <- testing$Survived

# library(xgboost)
#
# dtrain <- xgb.DMatrix( data = as.matrix( training[, - which( names(training) %in% c("Survived"))] ), label = as.matrix( training$Survived) )
# dtest <- xgb.DMatrix( data = as.matrix( testing[, - which( names(testing) %in% c("Survived"))] ), label = as.matrix( testing$Survived) )
# dSubmit <- xgb.DMatrix( data = as.matrix( submitData[, - which( names(submitData) %in% c("Survived"))] ), label = as.matrix( submitData$Survived) )
#
#
# wl <- list(train = dtrain, test = dtest)
# getinfo( wl$test, "label")
