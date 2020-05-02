# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:29:15 2020

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
#%% generate the swiss roll dataset
N = 2000
np.random.seed(2020)
t = np.sort(4 * np.pi * np.sqrt(np.random.rand(N)))
x = (t + 1) * np.cos(t)
y = (t + 1) * np.sin(t) 
z = 8 * np.pi * np.random.rand(N)
swiss_roll = np.array([x,y,z])
#%% plot the swiss roll data
cm = plt.cm.get_cmap('Spectral') 
plt.scatter(x, y, c=t, s=10, cmap=cm)
plt.title('Swiss Roll (2D)')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.show
#%% plot the 3D data
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x, y, z,c=t,s=10,cmap=cm)
#%% divide the dataset into two clusters
cluster1 =  range(800)
cluster2 = range(1800,2000)
plt.scatter(x[cluster1], y[cluster1], s=10)
plt.scatter(x[cluster2], y[cluster2], s=10)
plt.show
cluster = swiss_roll[:,cluster1]
cluster = np.hstack((cluster, swiss_roll[:,cluster2]))
cmp = t[cluster1]
cmp = np.hstack((cmp, t[cluster2]))
cluster = cluster.T
#%% plot the swiss roll cluster
plt.scatter(cluster[:,0], cluster[:,1], c=cmp, s=10, cmap=cm)
plt.title('Swiss Roll Clusters (2D)')
plt.xlabel('Dim 1')
plt.ylabel('Dim 2')
plt.show


#%% implement UMAP with different parameters
neighbor = np.array([20,30,50,80])
dist = np.array([0.05,0.1,0.2,0.5])

fig,axes = plt.subplots(4,4,figsize=(20,20),
                       sharex=False,
                       sharey=False)
for i in range(4):
    for j in range(4):
        model_umap = umap.UMAP(n_neighbors=int(neighbor[j]), 
                  min_dist= float(dist[i]))
        print(int(neighbor[j]),float(dist[i]))
        embedding = model_umap.fit_transform(cluster)
        axes[i,j].scatter(embedding[:,0], embedding[:,1],c=cmp, s=5, cmap=cm)
plt.subplots_adjust(wspace=0.1, hspace=0)
#%% implement UMAP for dimension reduction using different hyperparameters and plot 
model_umap = umap.UMAP(n_neighbors=50, 
                  min_dist= 0.2)
embedding = model_umap.fit_transform(cluster)
plt.scatter(embedding[:,0], embedding[:,1],c=cmp, s=10, cmap=cm)
plt.show
#%% implement k_means to cluster 
kmeans = KMeans(n_clusters=2)
kmeans.fit(embedding)
label_pred = kmeans.labels_
x0 = cluster[label_pred == 0]
x1 = cluster[label_pred == 1]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.show()


#%% implment t-SNE for dimension reduction using different values of perplexity
perplex = np.array([10,20,30,40,50,60])
fig,axes = plt.subplots(2,3,figsize=(15,20),
                       sharex=False,
                       sharey=False)
k = 0
for i in range(2):
    for j in range(3):
        model_tsne = TSNE(perplexity= int(perplex[k]))
        embedding = model_tsne.fit_transform(cluster)
        axes[i,j].scatter(embedding[:,0], embedding[:,1],c=cmp, s=10, cmap=cm)
        k = k + 1
plt.subplots_adjust(wspace=0, hspace=0.3)
#%% implement t-SNE 
model_tsne = TSNE(perplexity= 50)
embedding = model_tsne.fit_transform(cluster)
plt.scatter(embedding[:,0], embedding[:,1],c=cmp, s=10, cmap=cm)
plt.show
#%% implement k_means to cluster the result of t-sne
kmeans = KMeans(n_clusters=2)
kmeans.fit(embedding)
label_pred = kmeans.labels_
x0 = cluster[label_pred == 0]
x1 = cluster[label_pred == 1]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.show()


#%%  load the result of CPM
f = open("C:/UMAP_tSNE_CPM/data/swiss_cpm2.csv",encoding="utf-8")
cpm2 = pd.read_csv(f, header=None)
cpm2 = np.array(cpm2)
f = open("C:/UMAP_tSNE_CPM/data/swiss_cpm4.csv",encoding="utf-8")
cpm4 = pd.read_csv(f, header=None)
cpm4 = np.array(cpm4)
f = open("C:/UMAP_tSNE_CPM/data/swiss_cpm6.csv",encoding="utf-8")
cpm6 = pd.read_csv(f, header=None)
cpm6 = np.array(cpm6)
f = open("C:/UMAP_tSNE_CPM/data/swiss_cpm8.csv",encoding="utf-8")
cpm8 = pd.read_csv(f, header=None)
cpm8 = np.array(cpm8)
#%% show the result of CPM with different parameters
fig,axes = plt.subplots(2,2,figsize=(10,20),
                       sharex=False,
                       sharey=False)
axes[0,0].scatter(cpm2[:,0], cpm2[:,1],c=cmp, s=10, cmap=cm)
axes[0,1].scatter(cpm4[:,0], cpm4[:,1],c=cmp, s=10, cmap=cm)
axes[1,0].scatter(cpm6[:,0], cpm6[:,1],c=cmp, s=10, cmap=cm)
axes[1,1].scatter(cpm8[:,0], cpm8[:,1],c=cmp, s=10, cmap=cm)
plt.subplots_adjust(wspace=0, hspace=0.3)
#%% CPM
plt.scatter(cpm6[:,0], cpm6[:,1],c=cmp, s=10, cmap=cm)
plt.show
#%% implement k_means to cluster the result of CPM
kmeans = KMeans(n_clusters=2)
kmeans.fit(cpm6)
label_pred = kmeans.labels_
x0 = cluster[label_pred == 0]
x1 = cluster[label_pred == 1]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.show()











