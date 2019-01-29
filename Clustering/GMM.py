#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:14:04 2018

@author: MyReservoir
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def plot_gmm_graphs(model, df, xlabel, ylabel, 
                    col1= None, col2= None, 
                    projection=None, zlabel=None, title=None):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = projection)
    labels = model.fit (df).predict (df)
    if projection=='3d':
        ax.scatter (df[:, 0], df[:, 1], df[:, 2], c=labels)
        ax.set_zlabel(zlabel)
    else:
        ax.scatter (col1, col2, c=labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

df=pd.read_csv('evkirpal.csv',header=None,names=['A','B','C'])
X=df.values


X_train = X[:700]
X_test = X[700:]
scores = []
n_components = np.arange(1, 21)
for i in n_components:
    gmm = GMM(n_components=i).fit(X_train)
    scores.append(gmm.bic(X))
plt.plot(n_components, scores)
plt.xlabel('No of Components (K)')
plt.ylabel('Maximum Likelihood')
plt.title('Maximum Likelihood vs No. of Components')
plt.show()

n_max=5
print('Optimum number of Gaussian Clusters= {}'.format(n_max))

gmm_final = GMM(n_components=n_max).fit(X_train);
probs = gmm_final.predict_proba(X_test)
print(probs[:5].round(3))
predictions = gmm_final.predict(np.array(X_test).astype(np.float))
print(predictions)


plot_gmm_graphs(gmm_final, X,  xlabel='Column 1', ylabel='Column 2', 
                projection='3d', zlabel='Column 3',title='Scatter Plot- Gaussian Clustered Dataset')

plot_gmm_graphs(gmm_final, X, 
                xlabel='A', ylabel='B', 
                col1=X[:, 0], col2=X[:, 1], 
                title='Projection of the Gaussian Mixture Model on x-y plane')

plot_gmm_graphs(gmm_final, X, 
                xlabel='B', ylabel='C',  
                col1=X[:, 1], col2=X[:, 2],
                title='Projection of the Gaussian Mixture Model on y-z plane')

plot_gmm_graphs(gmm_final, X, 
                xlabel='A', ylabel='C', 
                col1=X[:, 0], col2=X[:, 2],
                title='Projection of the Gaussian Mixture Model on x-z plane')