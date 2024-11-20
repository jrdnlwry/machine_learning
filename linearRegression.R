rm(list=ls())

# Feature set

X = matrix(
  c(1,1,1, 1,2,3), 
  nrow=3, 
  ncol=2)


linear_regression_equation = function(x, y){

  # Write a Python function that performs linear regression using the normal equation. 
  # The function should take a matrix X (features) and a vector y (target) as input, 
  # and return the coefficients of the linear regression model.
  # Round your answer to four decimal places, -0.0 is a valid result 
  # for rounding a very small number.
    
    X = x
    Y = y
    # find the inverse of the above matrices
    invX = solve(t(X) %*% X)
    X2 = invX %*% t(X)
    theta = X2 %*% Y
    
    return (theta) 
  
}

coef = linear_regression_equation(X, c(1, 2, 3))

roundOutput = function(co1, co2){
  # Round the output from the above function
  coef1 = round(co1)
  coef2 = round(co2)  
  coef = c(coef1, coef2)
  
  return (coef)
}

coef = roundOutput(coef[1], coef[2])


print(paste0("The linear model is y = ", coef[1], " + ", coef[2], "*x"))
      
      