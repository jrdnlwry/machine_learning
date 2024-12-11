rm(list=ls())

set.seed(10)

install.packages("caret")
install.packages("dplyr")

#library(ggcorrplot)
library(e1071)
library(dplyr)
library(tidyr)
library(caret)

help(svm)

################################################################################
# Read in and transform the data
################################################################################
readData = function(path){
  data = read.csv(path, stringsAsFactors = TRUE)

  # attempt to convert yes/no to numeric values
  for (i in 1:nrow(data)){
    # ordinal encoding with business travel
    if (data$BusinessTravel[i] == "Non-Travel"){
      data$BusTravel[i] = 1
    } else if(data$BusinessTravel[i] == "Travel_Rarely"){
      data$BusTravel[i] = 2
    } else {
      data$BusTravel[i] = 3 # indicates travel frequently
    }
    
    # encode gender: M or F
    if (data$Gender[i] == "Female"){
      data$GenderVal[i] = 1
    } else {
      data$GenderVal[i] = 2
    }
    # encode Department
    # Sales, R&D, & HR
    if (data$Department[i] == "Sales"){
      data$DepartmentVal[i] = 1
    } else if (data$DepartmentVal[i] == "Research & Development"){
      data$DepartmentVal[i] = 2
    } else {
      data$DepartmentVal[i] = 3
    }
    #print(data$Attrition[i])
    if (data$Attrition[i] == "Yes"){
      data$AttrValue[i] = 1 
    } else {
      data$AttrValue[i] = 0
    }    
    
  }
  
  # drop uneeded columns
  drops = c("StandardHours", "Over18", "EmployeeID",
            "BusinessTravel", "Department", "EducationField",
            "Gender", "JobRole", "MaritalStatus", "EmployeeCount",
            "Attrition")
  
  data = data[,!(names(data) %in% drops)] 
  # replace NAs with  a median value (numeric values only)
  data = data %>% mutate(across(where(is.numeric), ~replace_na(., median(., na.rm=TRUE))))
  
  
#  # test scale data to quadratic term
#  for (i in 1:length(data$Age)){
#    #print(data$Attrition[i])
#    data$AgeSqr[i] = data$Age[i] * data$Age[i]
#  }
  
  
  return (data)
}

df = readData("C:/Users/tatel/OneDrive/Documents/Rstudio/datasets/attrition.csv")

head(df)

################################################################################
# partition data out
# 80% training; 10% testing
################################################################################

help(sample)

split = function(d){
  df = d
  # shuffle the data: sample without replacement
  dataSplit = sample(nrow(df), round(nrow(df)*.80))
  # partition data into training and test
  training = df[dataSplit, ]
  testing = df[-dataSplit,]
  #remaining_data = df[-dataSplit, ]
  # split the remaining data 50/50: 50 val and 50 testing
  #dfSplit = sample(nrow(remaining_data), round(nrow(remaining_data)*.50))
  #testing = remaining_data[dfSplit,]
  #val = remaining_data[-dfSplit,]
  
  dataList = list("training" = training,
                  "testing" = testing)
}

df = split(df)
# we have our training and testing data set contained in the df variable
names(df)
# sanity check
head(df$training)
nrow(df$testing)
# separate the training response from the coefficients
# prep training data
xTrain = as.matrix(df$training[,-22])
yTrain = as.factor(df$training[,22])
datTrain = data.frame(x = xTrain, y = yTrain)
datTrain$x.MonthlyIncome = as.numeric(datTrain$x.MonthlyIncome)
datTrain$y <- as.factor(datTrain$y)

# prep the test data
xTest = as.matrix(df$testing[,-22])
yTest = as.factor(df$testing[,22])
datTest = data.frame(x = xTest, y = yTest)
datTest$x.MonthlyIncome = as.numeric(datTest$x.MonthlyIncome)
datTest$y <- as.factor(datTest$y)

# sanity check
nrow(xTrain)
length(yTrain)

# build dataframe to train model
dat = data.frame(x = xTrain, y = yTrain)

# df$training$AttrValue <- as.factor(df$training$AttrValue)


svmfit = svm(y~., data=dat,
             kernel="linear", cost=10, scale=TRUE)


head(datTest)
length(datTest)
# predict using the test data set
pred.svm = predict(svmfit,datTest[,1:21])

# calculate the model accuracy

accuracy = sum(pred.svm == datTest[,22]) / nrow(datTest)








kernels = c("linear", "radial")
# creates a list of parameters to test
tune.out = tune(svm, y ~ ., data = datTrain, kernel = "radial",
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 1000)))

# access the cross-validation errors for each of these models
summary(tune.out)

names(tune.out)
# view the best parameter
bestmod = tune.out$best.model

pred.svm = predict(bestmod,datTest[,1:21])

# calculate the model accuracy
accuracy = sum(pred.svm == datTest[,22]) / nrow(datTest)


