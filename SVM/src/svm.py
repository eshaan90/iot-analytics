#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 17:00:22 2018

@author: MyReservoir
"""
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
# use seaborn plotting defaults
import seaborn as sns; sns.set()
from sklearn.grid_search import GridSearchCV
from mpl_toolkits import mplot3d


def training_model(svc_c,svc_gamma,data,y):
    accuracies=[]
    c=[]
    gamma=[]
    for c in svc_c:
        for gamma in svc_gamma:
            acc=0
            model=SVC(C=c, kernel='rbf', gamma=gamma)
            model.fit(data, y)
            acc=model.score(data, y)        
            accuracies.append([c,gamma,acc])

    df2=pd.DataFrame(accuracies,columns=['C','gamma','accuracy'])
    return df2


def plotting3d(df, title,xlabel,ylabel,zlabel, colorbar_label):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p =ax.scatter3D(df.C, df.gamma, df.accuracy, c=df.accuracy, cmap='Spectral', s=df.accuracy*60)
    ax.view_init(30, 185)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    cbr=fig.colorbar(p)
    cbr.set_label(colorbar_label, labelpad=1)
    plt.show()     

def heatmap(score_dct,svc_gamma,svc_c):
    
    score=[x[1] for x in score_dct]
    scores=np.array(score).reshape(len(svc_c),len(svc_gamma))
    plt.figure(figsize=(8, 6))
    plt.imshow(scores, interpolation='nearest', cmap='Spectral')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(svc_gamma)), svc_gamma, rotation=45)
    plt.yticks(np.arange(len(svc_c)), svc_c)
    plt.title('Validation accuracy')
    plt.show()
    
def grid_search(data,y, param_grid,nfolds):
    grid = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=param_grid, cv=nfolds)
    grid.fit(data, y)

    print("The best parameters are {} with a score of {}"
          .format(grid.best_params_, grid.best_score_))
    return grid


################## Read the data and plot #################
nfolds=5
data=pd.read_csv('evkirpal.csv',header=None, names=['A','B','label'])

data.head()

y=data['label']
columns_to_keep=['A','B']
data=data[columns_to_keep]

plt.figure()
plt.scatter(data.A, data.B, c=y, s=50, cmap='autumn');
plt.xlabel('A')
plt.ylabel('B')
plt.title('Scatter Plot of the dataset')

################## Normalize and Plot #################

data=((data-data.min())/(data.max()-data.min()))
data.head()

plt.figure()
plt.scatter(data.A, data.B, c=y, s=50, cmap='autumn');
plt.xlabel('A')
plt.ylabel('B')
plt.title('Scatter Plot of the normalized data')


#################### Grid Search #################
svc_c=[]
svc_gamma=[]
for i in range(-5,16,2):
    svc_c.append(2**i)
for i in range(-15,4,2):
    svc_gamma.append(2**i)
param_grid = {'C': svc_c,
              'gamma': svc_gamma}



grid=grid_search(data, y, param_grid, nfolds)

df1=training_model(svc_c,svc_gamma,data,y)
plotting3d(df1, 'Dataset in 3D','C','Gamma Values', 'Accuracy', 'Accuracy(%)')

score_dct=grid.grid_scores_
heatmap(score_dct,svc_gamma,svc_c)



#################### Finer Grid Search ############################

#Fine Search 1
svc_gamma=np.linspace(0.5, 5, 19)
svc_c=[]
for i in range(-8, -3, 1):
    svc_c.append(2**i)
svc_c.append(0.046875)
svc_c.append(0.0234375)
param_grid = {'C': svc_c,
              'gamma': svc_gamma}

grid=grid_search(data, y, param_grid, nfolds)

df1=training_model(svc_c,svc_gamma,data,y)
plotting3d(df1, 'Dataset in 3D','C','Gamma Values', 'Accuracy', 'Accuracy(%)')

score_dct=grid.grid_scores_
heatmap(score_dct,svc_gamma,svc_c)

#
##Fine Search 2
#svc_gamma=np.linspace(5, 10, 20)
#svc_c=np.linspace(2**14,2**16,20)
#param_grid = {'C': svc_c,
#              'gamma': svc_gamma}
#
#grid=grid_search(data, y, param_grid, nfolds)
#
#df1=training_model(svc_c,svc_gamma,data,y)
#plotting3d(df1, 'Dataset in 3D','C','Gamma Values', 'Accuracy', 'Accuracy(%)')
#
#score_dct=grid.grid_scores_
#heatmap(score_dct,svc_gamma,svc_c)

#
##Fine Search 3
#svc_gamma=np.linspace(5, 10, 10)
#svc_c=np.linspace(2**-3,1.5,10)
#
#param_grid = {'C': svc_c,
#              'gamma': svc_gamma}
#
#grid=grid_search(data, y, param_grid, nfolds)
#
#df1=training_model(svc_c,svc_gamma,data,y)
#plotting3d(df1, 'Dataset in 3D','C','Gamma Values', 'Accuracy', 'Accuracy(%)')
#
#score_dct=grid.grid_scores_
#heatmap(score_dct,svc_gamma,svc_c)


