#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:46:45 2018

@author: MyReservoir
"""


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


#DBSCAN functions
def DBSCAN_SIM(e,mpt, values):
    #Run DBscan on the data set
    dbsc = DBSCAN(eps = e, min_samples = mpt).fit(values)
    labels = dbsc.labels_
    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[dbsc.core_sample_indices_] = True
    
    # print core_samples
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#     print n_clusters_
    
    fig = plt.figure()
    ax2 = fig.add_subplot(111, projection='3d')    
    
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
        ax2.set_title('Estimated number of clusters: %d for minPts = %d, Radius = %.2f' %( n_clusters_ , mpt , e),loc="left")
    #     ax1.set_title('Scatter Plot')
        ax2.set_xlabel('x axis')
        ax2.set_ylabel('y axis')
        ax2.set_zlabel('z axis')
        #Plotting core samples
        xy = values[class_member_mask & core_samples]
        ax2.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        #PLotting outliers
        xy = values[class_member_mask & ~core_samples]
        ax2.plot(xy[:, 0], xy[:, 1],xy[:,2], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)    
    plt.show()


# KNN function to perform the elbow method for 
def KNN(minPts,data_transformed):
    X=data_transformed
    nbrs = NearestNeighbors(n_neighbors=minPts).fit(X)
    distances, indices = nbrs.kneighbors(X)
    y = sorted(distances[:,minPts-1],reverse=True)
    fig=plt.figure()
    ax1=plt.subplot(211)
    ax1.plot(range(0,len(X)),y)
    ax1.set_xlabel("Data Points")
    ax1.set_ylabel("Radius")
    ax1.set_title("Neighborhood Radius plot (Elbow Method DBScan minPts = " + str(minPts) + " )")

    
    ax2 = fig.add_subplot(212)
    ax2.plot(range(0,10),y[0:10])
    ax2.set_xlabel("Data Points")
    ax2.set_ylabel("Radius")
    ax2.set_title("Zoomed-in Radius Plot")

    plt.tight_layout()
    plt.show()



def printVals(values):
    x = values[:,0]
    y = values[:,1]
    z = values[:,2]
    fig = plt.figure()
    for c,m in [('r','o')]:
        
        ax1 = fig.add_subplot(111, projection='3d')    
        ax1.scatter(x,y,z,c=c)
        ax1.set_title("Scatter Plot")
        ax1.set_xlabel('A')
        ax1.set_ylabel('B')
        ax1.set_zlabel('C')
    plt.show()


df=pd.read_csv('evkirpal.csv',header=None,names=['A','B','C'])
X=df.values

dbScan_array = [[3,100.84],[4,115.09],[5,161.797],[6,177.295],[7,197.87],[8,222.633],[9,226.962]]



for i in range(0,len(dbScan_array)):
    minPts = dbScan_array[i][0]
    eps = dbScan_array[i][1]
    
    #Runn KNN to find the neighborhood Radius
    KNN(minPts,X)
    
    #Run DBSCAN by identifying right neighbor hood radius values
    DBSCAN_SIM(eps, minPts, X)
    