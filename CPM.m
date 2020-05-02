%% 
clear all; close all;
%% Run CPM on the Swiss roll dataset
addpath('drtoolbox');
% there seems to be lack of function 'find_nn' 
% addpath('C:\Users\DELL\Documents\MATLAB\NRPCA');
swiss_roll = xlsread("C:/UMAP_tSNE_CPM/data/swiss_cluster.csv");
swiss_roll(:,1) = [];
swiss_roll(1,:) = [];

ydata = cpm(swiss_roll,2,1,10,800);
C = [ones(800,1); 2*ones(200,1)];
scatter(ydata(:,1),ydata(:,2),9,C,'filled');
csvwrite("C:/UMAP t-SNE CPM/data/swiss_cpm.csv",ydata);
%% Run CPM on the Panreatic islet dataset
pancreatic = xlsread("C:/UMAP_tSNE_CPM/data/pancreatic_filtered.csv");
file = fopen("C:/UMAP_tSNE_CPM/data/pancreatic_index.txt");
cluster = textscan(file,'%d'); 
C = cluster{1};
ydata = cpm(pancreatic,2,1,5,1000);
scatter(ydata(:,1),ydata(:,2),9,C,'filled');
csvwrite("C:/UMAP t-SNE CPM/pancreatic_cpm.csv",ydata);
