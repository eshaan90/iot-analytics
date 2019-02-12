# Multivariable Linear Regression

Built different linear multivariable regression models to establish a relation between the dependent variable Y and a 5-tuple of independent variables X1, X2, X3, X4, and X5. Tested the goodness of fit of each model using residual analysis.

It was a three step process:
  1. Identify patterns in the data using exploratory statistical analysis techniques (such as outlier detection, boxplots, histograms, and correlation heatmaps).
  2. Implement various linear and polynomial regression models on the dataset and test for goodness of fit for each. 
  3. Finally, find the best model for the dataset and justify the approach. 
  
For this analysis Python's Statsmodel statistical package was used. Matplotlib and Seaborn packages were used for visualizations.

To check for the goodness of fit of the model, the following tests were performed on the residuals:
-Normality tests( Chi-squared test, Q-Q plot analysis)
-Autocorrelation of the residuals was checked
-Scatter plot analysis of the residuals

Outliers within the data have been handled by eliminating them using the Interquartile range. 

## How To Run the Code:

Each section's code has been clearly demarcated within the linear_regression.py file. 

If you want to enter your own data, change the location of dataset within the main() function. 
You will then have to mention the columns that you want to take into consideration for building your model. 
Again to be done in the main() function.


## Brief Analysis:
We got a poor fit for the simple linear regression model with just X1 as input. To improve the fit, we used a polynomial regression
model wherein we added X1^2 as another variable. This improved our R-squared value significantly and was a much better fit. 

Then taking all variables into account, didn't do much good for fitting a linear model. So we eliminated a few variables on the 
basis of the correlation matrix and achieved a much better fitted regression line. 

For a more detailed analysis of the results, check out the attached report. 
