import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots, cm
import sklearn.model_selection as skm
from ISLP import load_data, confusion_table
from sklearn.svm import SVC
from ISLP.svm import plot as plot_svm
from sklearn.metrics import RocCurveDisplay

file = r"C:\Users\tatel\OneDrive\Documents\Rstudio\datasets\attrition.csv"

def readData(filePath):
  data = pd.read_csv(filePath)
  return data

df = readData(file)

def cleanData(df):
  """
  apply one hot encoding to categorical data
  """
  df1 = df.copy()
  df_encoded = pd.get_dummies(
      df,
      columns=["BusinessTravel","Gender","Department"],
      dtype=int
  )

  # map attrition values
  df_encoded['AttrValue'] = df_encoded['Attrition'].map({"Yes": 1, "No": 0})

  df_encoded.drop(columns=["EmployeeID", "Age", "Attrition","DistanceFromHome","Education",
                          "EducationField", "EmployeeCount", "JobLevel",
                          "JobRole", "MaritalStatus", "MonthlyIncome",
                          "NumCompaniesWorked", "Over18", "PercentSalaryHike",
                          "StandardHours", "StockOptionLevel", "TotalWorkingYears",
                          "TrainingTimesLastYear", "YearsAtCompany", "YearsSinceLastPromotion",
                          "YearsWithCurrManager", "EnvironmentSatisfaction", "JobSatisfaction",
                          "WorkLifeBalance", "JobInvolvement", "PerformanceRating", "Attrition"],
                  inplace=True)
  df1.drop(columns=["Attrition", "EmployeeID", "BusinessTravel", "Department",
                    "EducationField", "Gender", "JobRole", "MaritalStatus", "Over18"], inplace=True)
  df1C = pd.concat([df1, df_encoded], axis=1)
  # replace missing values with median value
  df1C["NumCompaniesWorked"] = df1C["NumCompaniesWorked"].fillna(df1C["NumCompaniesWorked"].median())
  df1C["EnvironmentSatisfaction"] = df1C["EnvironmentSatisfaction"].fillna(df1C["EnvironmentSatisfaction"].median())
  df1C["JobSatisfaction"] = df1C["JobSatisfaction"].fillna(df1C["JobSatisfaction"].median())
  df1C["WorkLifeBalance"] = df1C["WorkLifeBalance"].fillna(df1C["WorkLifeBalance"].median())
  df1C["TotalWorkingYears"] = df1C["TotalWorkingYears"].fillna(df1C["TotalWorkingYears"].median())

  return df1C

data = cleanData(df)

def splitData(data):
  """
  split the data:
    80% training
    10% testing
    10% validation
  """
  # randomize the dataframe
  data_random = data.sample(frac=1)
  train, test, validate = np.split(data_random, [int(.8 * len(data_random)), int(.9 * len(data_random))])

  return train, test, validate

dataSplit = splitData(data)

train = dataSplit[0]
test = dataSplit[1]
validate = dataSplit[2]

"""
Create X & Y labels for training + test data set
"""
# TRAINING data
# X values
Xtrain = train.iloc[:,:-1]
# Y values
Ytrain = train.iloc[:,-1]
# TEST data
Xtest = test.iloc[:,:-1]
Ytest = test.iloc[:,-1]

# 'linear','poly','rbf','sigmoid'

# SVM estimator
svmFit = SVC()
# define the parameter grid
param_grid = {
    'C':[0.001,0.01,0.1,1,5,10,100],
    'kernel':['linear','rbf']
}
# create the GridSearchCV object
grid_search = skm.GridSearchCV(svmFit, param_grid, cv=5)
# fit the model
grid_search.fit(Xtrain, Ytrain)
# print the best parameters
print(grid_search.best_params_)