# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:51:35 2024

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
    
    def __init__(self, num_plots=1, gen_seed=0):
        self._num_plots = num_plots
        self.plotter = gen_plot_RNG.RNG_plot(num_plots=self._num_plots, seed=gen_seed)
        self.plotter.plot(save_fig=True, fig_dpi=200)
        self.plot_name = self.plotter.plot_name

    def sep_img(self, debug_plot=False):
        img = io.imread(self.plot_name)
        print(img.shape)
        X = img.reshape(-1, 4)
        n_clusters = self._num_plots+2
        
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
    
    def fit_centroids(self, max_iter=50, debug_plot=False, test_split=0.25, radius_ratio=200, noise_delete=50):
        self.X_KNN_train, self.X_KNN_test = train_test_split(self._iso_inds, test_size=test_split)
        self._max_iter = max_iter
        self._centroid_validate = NearestNeighbors()
        self.centroids = self.X_KNN_test
        self._radius = self._layer_iso.shape[0]/radius_ratio
        self._noise_delete = noise_delete

        print(self._layer_iso.shape)
        num_iter = 0
        #kevin was here ;))))
        
        while num_iter <= self._max_iter:
            num_iter += 1
            print("iter: ",num_iter)
            self.fit_to_centers()
            centroids_temp = self.get_far_apart_centroids()      
            if np.array_equal(centroids_temp, self.centroids):
                print("centroids unchanging at iter", num_iter)
                break
            self.centroids = centroids_temp
        
        if debug_plot:
            centroid_plot = np.copy(self._layer_iso)
            for i in range(self.centroids.shape[0]):
                centroid_plot[int(self.centroids[i, 0]), int(self.centroids[i, 1])] = 10.
            plt.figure(figsize=(7., 5.25), dpi=200)
            io.imshow(centroid_plot, vmin=0, vmax=10)
        return None

    def get_far_apart_centroids(self):            
        self._centroid_validate.fit(self.centroids)
        rad_dist, rad_inds = self._centroid_validate.radius_neighbors(self.centroids, radius=self._radius)
        centroids_temp = np.copy(self.centroids)
        
        ind_delete = np.array([])
        for i in range(rad_inds.shape[0]):
            if num_iter == 1:
                if rad_inds[i].size <= self._noise_delete and rad_inds[i].size != 0:
                    ind_delete = np.append(ind_delete, rad_inds[i][0])
            if rad_inds[i] != np.array([]):
                rad_inds[i] = np.delete(rad_inds[i], 0)
                ind_delete = np.append(ind_delete, rad_inds[i])
                
        ind_delete = np.unique(ind_delete, axis=0)
        
        if ind_delete != np.array([]):
            centroids_temp = np.delete(centroids_temp, ind_delete, axis=0)
        return centroids_temp
    
    def fit_to_centers(self):
        self._centroid_validate.fit(self.X_KNN_train)
        rads, inds = self._centroid_validate.radius_neighbors(self.centroids, radius=3*self._radius)
        centroids_fit = np.zeros_like(self.centroids)
        for i in range(inds.size):
            for j in range(inds[i].size):
                centroids_fit[i, 0] += self.X_KNN_train[inds[i][j], 0]/inds[i].size
                centroids_fit[i, 1] += self.X_KNN_train[inds[i][j], 1]/inds[i].size
            centroids_fit[i, 0] = int(centroids_fit[i, 0])
            centroids_fit[i, 1] = int(centroids_fit[i, 1])
        self.centroids = np.unique(centroids_fit, axis=0)
        return None
    
    def get_centroid_coords(self, x_range=[0,1], y_range=[0,1]):
        width = self._iso_img.shape[1]
        height = self._iso_img.shape[0]
        
        self.centroid_coords = np.zeros_like(self.centroids)
        self.centroid_coords[:, 0] = -((y_range[1]-y_range[0])*self.centroids[:, 0]/height + y_range[0])
        self.centroid_coords[:, 1] = ((x_range[1]-x_range[0])*self.centroids[:, 1]/width + x_range[0])
        return None
    
    def plot_digitized(self):
        plt.figure(dpi = 200)
        plt.plot(self.centroid_coords[:, 1], self.centroid_coords[:, 0], 'r.')
        return None