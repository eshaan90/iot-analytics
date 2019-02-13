#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:28:01 2018

@author: MyReservoir
"""


'''
Code for each task has been mentioned below with comments. 

If you run the whole code, all the plots for each task would be outputed in the console at once.
To have better readibility, it is advisable to run code for each task individually.

The outliers have been removed at the time of fitting each models.


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
from pandas.tools.plotting import autocorrelation_plot

    

def correlation_matrix(dataset):
    '''
        Plots the correlation matrix in several different ways.
    '''
    corr=dataset.corr()
    #corr=dataset.corr(method='spearman')
    print(corr)
    plt.figure(figsize=(10,10))
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,cmap='BuGn');
    plt.title('Correlation Heatmap of the dataset')
    plt.savefig('Correlation heatmap.png')
    sns.pairplot(dataset);  #Correlation Scatter Plot

    
def chi_squared_test(x):
#    k2,p=stats.chisquare(y_observed,f_exp=y_expected)
    k2, p = stats.normaltest(x)
    alpha = 0.05
    print("\n\np = {:g}".format(p))
    
    if p > alpha:  # null hypothesis: x comes from a normal distribution
        print("Not Significant Result; We fail to reject the null hypothesis")
    else:
        print("Significant Result; The null hypothesis can be rejected")  
    

def eliminate_outliers(dataset):
    '''
        Uses interquartile range to eliminate outliers. 
    '''
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    dataset = dataset[~((dataset < (Q1 - 1.5 * IQR)) |(dataset > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    return dataset
    
def computing_mean(X):
    return np.mean(X)

def computing_var(X):
    return np.var(X)


def plot_hist(X,xlabel, ylabel, title):
    '''
        Plots histograms
    '''
    plt.figure()
    sns.distplot(X);
    plt.grid(True)
    plt.ylabel(ylabel) 
    plt.xlabel(xlabel) 
    plt.title(title)
    plt.show()
    
def plot_boxplot(X,col):
    '''
        Function to plot boxplots.
    '''
    plt.figure()
    sns.boxplot(x=X);
    plt.title('Boxplot for {}'.format(col))
    plt.savefig('Boxplot plot: {}'.format(col))
    
def plot_qqplot(residuals,var,title):
    '''
        Function to plot Q-Q plots.
    '''
    plt.figure()
    sm.qqplot(residuals,loc=0,scale=var**0.5,line='q')
    plt.tight_layout()
    plt.title(title)
    plt.show()
   
def plot_model_output(X,Y,predictions, xlabel,ylabel,title):
    '''
        Function to plot the regression line produced by the model against the input data.
    '''
    plt.figure()
    plt.scatter(X, Y,color='black')
    plt.plot(X, predictions, 'r+', linewidth=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_scatterplot_residuals(predictions,residuals,xlabel,ylabel,title):
    '''
        Function to plot a scatter plot for the residuals.
    '''
    plt.figure()
    plt.scatter(predictions,residuals)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    

def linear_regression(X, Y):
    '''
        Input: the dataset X and the target Y
        Creates the Ordinary Least Squares model, then predictions on the data are made, and 
        the residuals are computed. 
    '''
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)

    print("\n",model.summary())
    residuals=Y-predictions

    return (residuals,predictions)    


def main():    
    ##################### Reading the dataset ####################
    
    dataset=pd.read_csv('evkirpal.csv',header=None,names=['X1','X2','X3','X4','X5','Y'])
    dataset.describe()
    
    
    ##################### Task 1: Basic Statistical Analysis ####################
    ''' 
        Computing mean and plotting histograms and boxplots for each column
    '''
    means={}
    var={}
    for i in range(1,6):
        col='X'+str(i)
        means[col]=computing_mean(dataset[col])
        var[col]=computing_var(dataset[col])
        plot_hist(dataset[col],col, 'Frequency', 'Histogram plot for {}'.format(col))
        plt.savefig('Histogram plot: {}'.format(col));
        plot_boxplot(dataset[col],col)    
        
        
    dataset_mean=pd.Series(means)
    dataset_var=pd.Series(var)
    print('\n\nMeans: \n', dataset_mean)
    print('\n\nVariance: \n', dataset_var)    
        
    
    
    ##################### Task 1: Computing Correlation Matrix ####################
    correlation_matrix(dataset)
    
    
    ##################### Task 2.1: Simple Linear Regression ####################
    col=['X1','Y']
    X=dataset[col]
    X=eliminate_outliers(X)
    Y=X['Y']
    X=X['X1']
    X = sm.add_constant(X)
    (residuals,predictions)=linear_regression(X,Y)
    
    plot_model_output(X['X1'], Y, predictions, 'X1', 'Y', 'Results of Simple Linear Regression')
    
    plot_hist(residuals, 'Residuals', 'Frequency', 'Histogram of residuals')
    plot_scatterplot_residuals(predictions, residuals, 'Predicted values', 'Residuals',\
                                   'Scatter Diagram of Residuals-Simple Linear Regression')
    
    chi_squared_test(residuals)


    var=computing_var(residuals)
    print('\n\n Variance of Residual- Simple Linear Regression: {}'.format(var))
    plot_qqplot(residuals,var,'Q-Q Plot of Residuals and the Normal Distribution')

    
    ###################### Task 2.2: Polynomial Regression #####################
    
    col=['X1','Y']
    X=dataset[col]
    X=eliminate_outliers(X)
    X['X1_squared']=X['X1']**2
    Y=X['Y']
    col=['X1','X1_squared']
    X=X[col]
    X = sm.add_constant(X)
    (residuals,predictions)=linear_regression(X,Y)
    
    plot_model_output(X['X1'], Y, predictions, 'X1', 'Y', 'Results of Polynomial Regression')
    
    plot_hist(residuals, 'Residuals', 'Frequency', 'Histogram of residuals')
    plot_scatterplot_residuals(predictions, residuals, 'Predicted values', 'Residuals',\
                                   'Scatter Diagram of Residuals-Polynomial Regression')
    
    var=computing_var(residuals)
    print('\n\n Variance of Residual- Polynomial Linear Regression: {}'.format(var))
    plot_qqplot(residuals,var,'Q-Q Plot of Residuals and the Normal Distribution')
    
    chi_squared_test(residuals)
    
    
    ###################### Task 3: Multivariable Linear Regression #####################
    col=['X1','X2','X3','X4','X5','Y']
    X=dataset[col]
    X=eliminate_outliers(X)
    Y=X['Y']
    X=X[col[:-1]]
    X=sm.add_constant(X)
    (residuals,predictions)=linear_regression(X,Y)
    
    plot_hist(residuals, 'Residuals', 'Frequency', 'Histogram of residuals')
    plot_scatterplot_residuals(predictions,residuals,'Predicted values','Residuals',\
                                   'Scatter Diagram of Residuals-Multivariable Linear Regression')
    
    var=computing_var(residuals)
    print('\n\n Variance of Residual- Multivariable Linear Regression: {}'.format(var))
    plot_qqplot(residuals,var,'Q-Q Plot of Residuals and the Normal Distribution')
    
    chi_squared_test(residuals)
    
    autocorrelation_plot(residuals)
    plt.show()
    
    
    ##################### Task 3.2: Fit the best regression model #####################
    '''
        Best fitted regression line computed here
    
    '''
    col=['X1','X5','Y']
    X=dataset[col]
    X=eliminate_outliers(X)
    Y=X['Y']
    X=X[col[:-1]]
    (residuals,predictions)=linear_regression(X,Y)    
    
    plot_hist(residuals, 'Residuals', 'Frequency', 'Histogram of residuals')
    plot_scatterplot_residuals(predictions,residuals,'Predicted values','Residuals',\
                                   'Scatter Diagram of Residuals-Multivariable Linear Regression')
    
    chi_squared_test(residuals)
    
    var=computing_var(residuals)
    print('\n\n Variance of Residual- Multivariable Linear Regression: {}'.format(var))
    plot_qqplot(residuals,var,'Q-Q Plot of Residuals and the Normal Distribution')
    
    autocorrelation_plot(residuals)
    plt.show()


if __name__=='__main__':
    main()