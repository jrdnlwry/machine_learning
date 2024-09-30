rm(list=ls())

set.seed(10)
install.packages("ggcorrplot")
#library(ggcorrplot)
library(e1071)
library(dplyr)
library(tidyr)
help(svm)

readData = function(path){
  data = read.csv(path, stringsAsFactors = TRUE)

  # attempt to convert yes/no to numeric values
  for (i in 1:length(data$Attrition)){
    #print(data$Attrition[i])
    if (data$Attrition[i] == "Yes"){
      data$AttrValue[i] = 1 
    } else {
      data$AttrValue[i] = 0
    }
  }
#  # test scale data to quadratic term
#  for (i in 1:length(data$Age)){
#    #print(data$Attrition[i])
#    data$AgeSqr[i] = data$Age[i] * data$Age[i]
#  }
  
  
  return (data)
}

df = readData("C:/Users/tatel/OneDrive/Documents/Rstudio/datasets/attrition.csv")

# drop not needed columns
drops = c("StandardHours", "Over18", "EmployeeID",
          "BusinessTravel", "Department", "EducationField",
          "Gender", "JobRole", "MaritalStatus", "EmployeeCount")
df = df[,!(names(df) %in% drops)]

head(df)
# replace NAs with median value for each column
df %>% mutate(across(where(is.numeric), ~replace_na(., median(., na.rm=TRUE))))

# SPLITTING DATA INTO TRAINING & TESTING
# select 80% of data for training
sample = sample.int(n = nrow(df), size = floor(0.80*nrow(df)), replace = FALSE)
train = df[sample, ]
# 20% of data for testing
test = df[-sample, ]
test = test[3:20]

head(train)
# value we are predicting 
y = train$Attrition
x = train[3:20]

# build the initial model with the training data
svmfit = svm(x, y)

print(svmfit)
summary(svmfit)