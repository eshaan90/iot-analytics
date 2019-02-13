#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:25:47 2018

@author: MyReservoir
"""


from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import openpyxl
import pandas as pd


dataset=pd.read_csv('evkirpal.csv',header=None,names=['A','B','C'])
dataset.describe()

x = dataset.A
y = dataset.B
z = dataset.C



fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')    
ax1.scatter(x, y, z, c='r', marker='o')
ax1.set_title("Scatter Plot")
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')
plt.show()
fig.savefig('3d_Scatter_plot.png')



linked = linkage(dataset, 'single')

fig2=plt.figure(figsize=(80, 20))  
dendrogram(linked,  
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()
fig2.savefig('dendogram.png')


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(dataset) 



# print "Cophenetic Correlation (CC) Values"
#Create linkage using the ward minimization algorithms

'''
for method in ['single','complete','average','weighted']:
    for metric in ['euclidean','cityblock','cosine','seuclidean','minkowski']:
        Z = linkage(values,method,metric)
        c, coph_dists = cophenet(Z, pdist(values))
        print "Method : " + str(method) + " , Metric : " + str(metric) + " , CC : " + str(c)
'''     
#This is the major step to create the cluster linkages
Z = linkage(dataset,'ward','euclidean')
# print Z
c, coph_dists = cophenet(Z, pdist(dataset))
sqform = squareform(pdist(dataset))
# print coph_dists
# print len(coph_dists)
# print "Distances"
# print pdist(values)
# print "Method : euclidean, Metric : ward, CC : " + str(c)

#calculate full dendrogram
fig2=plt.figure(figsize=(60,30), dpi=120, facecolor='white', edgecolor='green')
plt.title(" Dendrogram (Hierarchical Clustering)")
plt.xlabel("Index")
plt.ylabel("distance")
dendrogram(Z,
           leaf_rotation=90.,
           leaf_font_size=1.,
           show_leaf_counts=True,
           color_threshold= 0.7*max(Z[:,2]),#0.7*max(Z[:,2]),#This is the default vale
           no_labels=False,
           )
plt.show()
fig2.savefig('dendogram.png')

#show truncated dendrogram
fig3=plt.figure(figsize=(60,30), dpi=120, facecolor='white', edgecolor='green')
plt.title(" Truncated Dendrogram (Hierarchical Clustering)")
plt.xlabel("Sample Index")
plt.ylabel("distance")
dendrogram(Z,
           truncate_mode='lastp', #show only last 12 merges
           p=12, #
           leaf_rotation=90.,
           leaf_font_size=12.,
           show_leaf_counts=True,
           color_threshold= 0.7*max(Z[:,2]),#0.7*max(Z[:,2]),#This is the default vale
           no_labels=False,
           show_contracted=True,
           )
plt.show()
fig3.savefig('truncated_dendogram.png')


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

#Annotated dendrogram
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()



#find the clusters based on distance
fig = plt.figure()
max_d = 1155 #(based on average of max jump in dendrogram)
clusters = fcluster(Z, max_d, criterion='distance')
#print clusters
# plt.figure(figsize=(10, 8))
# plt.scatter(x,y,z, c=clusters, cmap='prism')  # plot points with cluster dependent colors
ax1 = fig.add_subplot(111, projection='3d')    
ax1.scatter(x,y,z,c=clusters)
ax1.set_title("Scatter Plot")
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis') 

plt.show()

#dendrogram based on splitting the clusters
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()



'''
#This part is optional
#find the clusters based on visual observation. No. of clusters = 3
fig = plt.figure()
k=2
clusters = fcluster(Z, k, criterion='maxclust')
#print clusters
ax1 = fig.add_subplot(111, projection='3d')    
ax1.scatter(x,y,z,c=clusters)
ax1.set_title("Scatter Plot")
ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis') 
plt.show()
# print Z[:200]