# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:47:51 2024

@author: heckeh
"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import gen_plot_RNG

class PlotDigitizer:
    
    def __init__(self):
        self.plotter = gen_plot_RNG.RNG_plot(seed=0)
        self.plotter.plot(save_fig=True, fig_dpi=150)
        self.plot_name = self.plotter.plot_name

    def sep_img(self, num_plots=1, debug_plot=False):
        img = io.imread(self.plot_name)
        X = img.reshape(-1, 4)
        n_clusters = num_plots+2
        
        kmeans = KMeans(n_clusters, n_init=10, init='k-means++')
        kmeans.fit(X)
        clusters = kmeans.predict(X)
        
        seg_img = clusters.reshape(img.shape[0], img.shape[1])
        self._iso_img = np.zeros([seg_img.shape[0], seg_img.shape[1], n_clusters])
        for k in range(n_clusters):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if seg_img[i, j] == k:
                        self._iso_img[i, j, k] = 1.
        
        self._layer_iso = self._iso_img[:, :, 2]
        self._iso_inds = np.zeros([1,2])
        for i in range(self._iso_img.shape[0]):
            for j in range(self._iso_img.shape[1]):
                if self._layer_iso[i, j] == 1.:
                    self._iso_inds = np.append(self._iso_inds, [[i, j]], axis=0)
        self._iso_inds = np.delete(self._iso_inds, 0, axis=0)
        
        if debug_plot:
            plt.figure(figsize=(7., 5.25))
            io.imshow(seg_img)
            
            plt.figure(figsize=(7., 5.25))
            io.imshow(self._iso_img[:, :, 0])
            plt.title("1")
            
            plt.figure(figsize=(7., 5.25))
            io.imshow(self._iso_img[:, :, 1])
            plt.title("2")
            
            plt.figure(figsize=(7., 5.25))
            io.imshow(self._iso_img[:, :, 2])
            plt.title("3")
        return None
    
    def fit_centroids(self, max_iter=50, debug_plot=False, test_split=0.25, n_radius=10, radius_ratio=200):
        X_KNN_train, X_KNN_test = train_test_split(self._iso_inds, test_size=test_split)
        self._centroid_validate = NearestNeighbors()
        self.centroids = X_KNN_test

        print(self._layer_iso.shape)
        num_iter = 0
                    
        while num_iter <= max_iter:
            num_iter += 1
            print("iter:", num_iter)
            self._centroid_validate.fit(self.centroids)
            rad_dist, rad_inds = self._centroid_validate.radius_neighbors(self.centroids, radius=self._layer_iso.shape[0]/radius_ratio)
            for i in range(rad_inds.shape[0]):
                if rad_inds[i] != np.array([]):
                    centroids_temp = np.delete(self.centroids, rad_inds[i], axis=0)
                    break
            if np.array_equal(centroids_temp, self.centroids):
                print("centroid array unchanging")
                break
            self.centroids = centroids_temp        
        
        self._centroid_validate.fit(X_KNN_train)
        rads, inds = self._centroid_validate.radius_neighbors(self.centroids, radius=2*self._layer_iso.shape[0]/radius_ratio)
        centroids_fit = np.zeros_like(self.centroids)
        for i in range(inds.size):
            for j in range(inds[i].size):
                centroids_fit[i, 0] += X_KNN_train[inds[i][j], 0]/inds[i].size
                centroids_fit[i, 1] += X_KNN_train[inds[i][j], 1]/inds[i].size
            centroids_fit[i, 0] = int(centroids_fit[i, 0])
            centroids_fit[i, 1] = int(centroids_fit[i, 1])
        self.centroids = np.unique(centroids_fit, axis=0)
        
        centroid_plot = np.copy(self._layer_iso)
        for i in range(self.centroids.shape[0]):
            centroid_plot[int(self.centroids[i, 0]), int(self.centroids[i, 1])] = 10.
        plt.figure(figsize=(7., 5.25), dpi=200)
        io.imshow(centroid_plot, vmin=0, vmax=10)
        return None