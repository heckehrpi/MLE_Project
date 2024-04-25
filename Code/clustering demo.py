# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:15:55 2024

@author: henhe
"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from acc_score import calc_accuracy
import seaborn as sns
from gen_plot import RNG_Plot



num_plots = 2

plt.style.use('fivethirtyeight')

seed = int(np.random.random()*1e6)
pltr = RNG_Plot(seed=seed)
pltr.gen_pts(num_plots)

dpi = np.random.randint(5, 30)*10
pltr.plot(fig_dpi=dpi, save_fig=True)
fig_name = pltr.fig_name
color_img = io.imread(fig_name)
# color_img = io.imread("exam_plot.png")

n_clusters = 15
while n_clusters >= 4:
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, init='k-means++')
    kmeans.fit(color_img.reshape(-1, 3))
    clusters = kmeans.predict(color_img.reshape(-1, 3))
    
    clusters_img = clusters.reshape(color_img.shape[0], color_img.shape[1])
    
    plt.style.use('default')
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.title(n_clusters)
    plt.imshow(clusters_img.astype(int))
    
    layered_clusters = np.zeros([clusters_img.shape[0], clusters_img.shape[1], n_clusters])
    for k in range(n_clusters):
        for i in range(layered_clusters.shape[0]):
            for j in range(layered_clusters.shape[1]):
                if clusters_img[i, j] == k:
                    layered_clusters[i, j, k] = 1.
    
    for i in range(n_clusters):
        plt.figure(figsize=(7., 5.25))
        plt.imshow(layered_clusters[:, :, i])
        plt.title(i)
    
    n_clusters -= 1
    

    
