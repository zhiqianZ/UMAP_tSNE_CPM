# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:57:34 2020

@author: DELL
"""
import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

#%% laod the dataset and assign the cluster label
f = open("C:/UMAP_tSNE_CPM/data/pancreatic_filtered.csv",encoding="utf-8")
pancreatic = pd.read_csv(f, header=None)
f = open("C:/UMAP_tSNE_CPM/data/pancreatic_index.txt",encoding="utf-8")
cluster = pd.read_table(f,header=None)
cluster = np.array(cluster)
cluster = cluster[:,0]
cluster1 = np.argwhere(cluster==1)
cluster2 = np.argwhere(cluster==2)
cluster3 = np.argwhere(cluster==3)
cluster4 = np.argwhere(cluster==4)
cluster5 = np.argwhere(cluster==5)
cluster6 = np.argwhere(cluster==6)
cluster7 = np.argwhere(cluster==7)
cluster8 = np.argwhere(cluster==8)
cluster9 = np.argwhere(cluster==9)

#%% UMAP
np.random.seed(2020)
model_umap = umap.UMAP(n_neighbors=15, 
                  min_dist= 0.2)
embed_umap = model_umap.fit_transform(pancreatic)
#%% result of UMAP
plt.scatter(embed_umap[cluster1,0], embed_umap[cluster1,1], s=5, label='acinar')
plt.scatter(embed_umap[cluster2,0], embed_umap[cluster2,1], s=5, label='activated_stellate')
plt.scatter(embed_umap[cluster3,0], embed_umap[cluster3,1], s=5, label='alpha')
plt.scatter(embed_umap[cluster4,0], embed_umap[cluster4,1], s=5, label='beta')
plt.scatter(embed_umap[cluster5,0], embed_umap[cluster5,1], s=5, label='delta')
plt.scatter(embed_umap[cluster6,0], embed_umap[cluster6,1], s=5, label='ductal')
plt.scatter(embed_umap[cluster7,0], embed_umap[cluster7,1], s=5, label='endothelial')
plt.scatter(embed_umap[cluster8,0], embed_umap[cluster8,1], s=5, label='gamma')
plt.scatter(embed_umap[cluster9,0], embed_umap[cluster9,1], s=5, label='quiescent_stellate')
plt.legend()
plt.show
#%% implement k_means to cluster the result of t-sne
kmeans = KMeans(n_clusters=9)
kmeans.fit(embed_umap)
label_pred = kmeans.labels_
# assign the labels
x0 = embed_umap[label_pred == 0]
x1 = embed_umap[label_pred == 1]
x2 = embed_umap[label_pred == 2]
x3 = embed_umap[label_pred == 3]
x4 = embed_umap[label_pred == 4]
x5 = embed_umap[label_pred == 5]
x6 = embed_umap[label_pred == 6]
x7 = embed_umap[label_pred == 7]
x8 = embed_umap[label_pred == 8]
# plot the clsutering result 
plt.scatter(x0[:, 0], x0[:, 1], s=5, label='label0')
plt.scatter(x1[:, 0], x1[:, 1], s=5, label='label1')
plt.scatter(x2[:, 0], x2[:, 1], s=5, label='label2')
plt.scatter(x3[:, 0], x3[:, 1], s=5, label='label3')
plt.scatter(x4[:, 0], x4[:, 1], s=5, label='label4')
plt.scatter(x5[:, 0], x5[:, 1], s=5, label='label5')
plt.scatter(x6[:, 0], x6[:, 1], s=5, label='label6')
plt.scatter(x7[:, 0], x7[:, 1], s=5, label='label7')
plt.scatter(x8[:, 0], x8[:, 1], s=5, label='label8')
plt.legend()
plt.show()


#%% tSNE
np.random.seed(202004)
model_tsne = TSNE(perplexity= 20)
embed_tsne = model_tsne.fit_transform(pancreatic)
#%% result of t-SNE
plt.scatter(embed_tsne[cluster1,0], embed_tsne[cluster1,1], s=5, label='acinar')
plt.scatter(embed_tsne[cluster2,0], embed_tsne[cluster2,1], s=5, label='activated_stellate')
plt.scatter(embed_tsne[cluster3,0], embed_tsne[cluster3,1], s=5, label='alpha')
plt.scatter(embed_tsne[cluster4,0], embed_tsne[cluster4,1], s=5, label='beta')
plt.scatter(embed_tsne[cluster5,0], embed_tsne[cluster5,1], s=5, label='delta')
plt.scatter(embed_tsne[cluster6,0], embed_tsne[cluster6,1], s=5, label='ductal')
plt.scatter(embed_tsne[cluster7,0], embed_tsne[cluster7,1], s=5, label='endothelial')
plt.scatter(embed_tsne[cluster8,0], embed_tsne[cluster8,1], s=5, label='gamma')
plt.scatter(embed_tsne[cluster9,0], embed_tsne[cluster9,1], s=5, label='quiescent_stellate')
plt.legend()
plt.show
#%% implement k_means to cluster the result of t-sne
kmeans = KMeans(n_clusters=9)
kmeans.fit(embed_tsne)
label_pred = kmeans.labels_
x0 = embed_tsne[label_pred == 0]
x1 = embed_tsne[label_pred == 1]
x2 = embed_tsne[label_pred == 2]
x3 = embed_tsne[label_pred == 3]
x4 = embed_tsne[label_pred == 4]
x5 = embed_tsne[label_pred == 5]
x6 = embed_tsne[label_pred == 6]
x7 = embed_tsne[label_pred == 7]
x8 = embed_tsne[label_pred == 8]
plt.scatter(x0[:, 0], x0[:, 1], s=5, label='label0')
plt.scatter(x1[:, 0], x1[:, 1], s=5, label='label1')
plt.scatter(x2[:, 0], x2[:, 1], s=5, label='label2')
plt.scatter(x3[:, 0], x3[:, 1], s=5, label='label3')
plt.scatter(x4[:, 0], x4[:, 1], s=5, label='label4')
plt.scatter(x5[:, 0], x5[:, 1], s=5, label='label5')
plt.scatter(x6[:, 0], x6[:, 1], s=5, label='label6')
plt.scatter(x7[:, 0], x7[:, 1], s=5, label='label7')
plt.scatter(x8[:, 0], x8[:, 1], s=5, label='label8')
plt.legend()
plt.show()


#%%   CPM 
f = open("C:/UMAP_tSNE_CPM/data/pancreatic_cpm.csv",encoding="utf-8")
embed_cpm = pd.read_csv(f, header=None)
embed_cpm = np.array(embed_cpm)
plt.scatter(embed_cpm[cluster1,0], embed_cpm[cluster1,1], s=5, label='acinar')
plt.scatter(embed_cpm[cluster2,0], embed_cpm[cluster2,1], s=5, label='activated_stellate')
plt.scatter(embed_cpm[cluster3,0], embed_cpm[cluster3,1], s=5, label='alpha')
plt.scatter(embed_cpm[cluster4,0], embed_cpm[cluster4,1], s=5, label='beta')
plt.scatter(embed_cpm[cluster5,0], embed_cpm[cluster5,1], s=5, label='delta')
plt.scatter(embed_cpm[cluster6,0], embed_cpm[cluster6,1], s=5, label='ductal')
plt.scatter(embed_cpm[cluster7,0], embed_cpm[cluster7,1], s=5, label='endothelial')
plt.scatter(embed_cpm[cluster8,0], embed_cpm[cluster8,1], s=5, label='gamma')
plt.scatter(embed_cpm[cluster9,0], embed_cpm[cluster9,1], s=5, label='quiescent_stellate')
plt.legend()
plt.show
#%%
kmeans = KMeans(n_clusters=9)
kmeans.fit(embed_cpm)
label_pred = kmeans.labels_
x0 = embed_cpm[label_pred == 0]
x1 = embed_cpm[label_pred == 1]
x2 = embed_cpm[label_pred == 2]
x3 = embed_cpm[label_pred == 3]
x4 = embed_cpm[label_pred == 4]
x5 = embed_cpm[label_pred == 5]
x6 = embed_cpm[label_pred == 6]
x7 = embed_cpm[label_pred == 7]
x8 = embed_cpm[label_pred == 8]
plt.scatter(x0[:, 0], x0[:, 1], s=5, label='label0')
plt.scatter(x1[:, 0], x1[:, 1], s=5, label='label1')
plt.scatter(x2[:, 0], x2[:, 1], s=5, label='label2')
plt.scatter(x3[:, 0], x3[:, 1], s=5, label='label3')
plt.scatter(x4[:, 0], x4[:, 1], s=5, label='label4')
plt.scatter(x5[:, 0], x5[:, 1], s=5, label='label5')
plt.scatter(x6[:, 0], x6[:, 1], s=5, label='label6')
plt.scatter(x7[:, 0], x7[:, 1], s=5, label='label7')
plt.scatter(x8[:, 0], x8[:, 1], s=5, label='label8')
plt.legend()
plt.show()
#%%
