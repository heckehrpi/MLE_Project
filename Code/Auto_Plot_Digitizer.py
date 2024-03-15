# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 03:57:04 2024

@author: heckeh
"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import gen_plot

class PlotDigitizer:
    
    def __init__(self, num_plots, x_lim, y_lim, fig_name="gen_plot.png"):
        self.img = io.imread(fig_name)
        self._num_plots = num_plots
        self._num_clusters = num_plots + 2
        self._y_lim = y_lim
        self._x_lim = x_lim
        return None
            
            
    def _separate_clusters(self, verbose=True, debug_plot=False):
        X = self.img.reshape(-1, 4)
        kmeans = KMeans(self._num_clusters, n_init=15, init='k-means++')
        kmeans.fit(X)
        self.clusters = kmeans.predict(X).reshape(self.img.shape[0], self.img.shape[1])
        self._layered_clusters = np.zeros([self.clusters.shape[0], self.clusters.shape[1], self._num_clusters])
        for k in range(self._num_clusters):
            for i in range(self._layered_clusters.shape[0]):
                for j in range(self._layered_clusters.shape[1]):
                    if self.clusters[i, j] == k:
                        self._layered_clusters[i, j, k] = 1.
        return None
        
    def _detect_clusters(self, background_tol=0.89, axis_tol=0.5):
        self._separate_clusters()
        self._points_layers = []
        
        for k in range(self._num_clusters):
            avg = np.average(self._layered_clusters[:, :, k])
            if avg >= background_tol:
                self._background_layer = k
                
        for k in range(self._num_clusters):
            if k != self._background_layer:
                v_avg = np.average(self._layered_clusters[:, :, k], axis=0)
                h_avg = np.average(self._layered_clusters[:, :, k], axis=1)
                
                x_axis = []
                y_axis = []
                
                for i in range(h_avg.size):
                    if h_avg[i] >= axis_tol:
                        x_axis.append(i)
                        
                for j in range(v_avg.size):
                    if v_avg[j] >= axis_tol:
                        y_axis.append(j)
                
                if len(x_axis) != 0 and len(y_axis) != 0:
                    self._axes_layer = k
                    self._x_axis = x_axis
                    self._y_axis = y_axis
                if len(x_axis) == 0 and len(y_axis) == 0:
                    self._points_layers.append(k)
        return None
    
    def _get_preprocessed_clusters(self, noise_tol=3):
        self._detect_clusters()
        for layer in self._points_layers:
            for i in range(self._layered_clusters[:, :, layer].shape[0]):
                for j in range(self._layered_clusters[:, :, layer].shape[1]):
                    if self._layered_clusters[i, j, layer] >= 0.5:
                        if i <= min(self._x_axis)+noise_tol or i >= max(self._x_axis)-noise_tol:
                            self._layered_clusters[i, j, layer] = 0
                        if j <= min(self._y_axis)+noise_tol or j >= max(self._y_axis)-noise_tol:
                            self._layered_clusters[i, j, layer] = 0
        return None
    
    def _get_cluster(self, layer):
        layer_iso = self._layered_clusters[:, :, layer]
        inds = np.zeros([1,2])
        for i in range(layer_iso.shape[0]):
            for j in range(layer_iso.shape[1]):
                if layer_iso[i, j] == 1.:
                    inds = np.append(inds, [[i, j]], axis=0)
        inds = np.delete(inds, 0, axis=0)
        return inds
    
    def _fit_centroids(self, plot=0, epochs=50, init_split=0.15, culling_radius_factor=0.005, debug_plot=False):
        self._get_preprocessed_clusters()
        inds = np.zeros(self._num_plots, dtype=list)
        X_train = np.zeros(self._num_plots, dtype=list)
        self.centroids = np.zeros(self._num_plots, dtype=list)
        
        for i in range(self._num_plots):
            inds[i] = self._get_cluster(self._points_layers[i])
            X_train[i], self.centroids[i] = train_test_split(inds[i], test_size=init_split)
        self._centroid_validate = NearestNeighbors()
        self._radius = culling_radius_factor*self._layered_clusters.shape[0]
        
        self._num_iter = 0
        for i in range(self._num_plots):
            while self._num_iter <= epochs:
                self._num_iter += 1
                centroids_temp = np.copy(self.centroids[i])
                centroids_temp = self._cull_centroids(centroids_temp)
                centroids_temp = self._fit_to_centers(centroids_temp, X_train[i])
                if np.array_equal(centroids_temp, self.centroids):
                    print("centroids unchanging at iter", self._num_iter)
                    break
                self.centroids[i] = centroids_temp
        return None
    
    def _cull_centroids(self, centroids):
        centroids_culled = np.copy(centroids)
        self._centroid_validate.fit(centroids_culled)
        rads, inds = self._centroid_validate.radius_neighbors(centroids, radius=self._radius)
        inds_delete = np.array([])
        for i in range(inds.shape[0]):
            if inds[i].size != 0:
                inds[i] = np.delete(inds[i], 0)
                inds_delete = np.append(inds_delete, inds[i])
        inds_delete = np.unique(inds_delete, axis=0)
        if inds_delete.size != 0:
            centroids_culled = np.delete(centroids, inds_delete.astype(int), axis=0)
        return centroids_culled
    
    def _fit_to_centers(self, centroids, X_train):
        self._centroid_validate.fit(X_train)
        rads, inds = self._centroid_validate.radius_neighbors(centroids, radius=2*self._radius)
        centroids_temp = np.zeros_like(centroids)
        for i in range(inds.size):
            for j in range(inds[i].size):
                centroids_temp[i, 0] += X_train[inds[i][j], 0]/inds[i].size
                centroids_temp[i, 1] += X_train[inds[i][j], 1]/inds[i].size
            centroids_temp[i, 0] = int(centroids_temp[i, 0])
            centroids_temp[i, 1] = int(centroids_temp[i, 1])
        centroids_fit = np.unique(centroids_temp, axis=0)
        return centroids_fit
    
    def get_pts(self):
        self._fit_centroids()
        
        width = max(self._y_axis) - min(self._y_axis)
        height = max(self._x_axis) - min(self._x_axis)
        
        centroid_coords = np.zeros_like(self.centroids)
        
        for i in range(self._num_plots):
            centroid_coords[i][:, 0] = ((self._y_lim[1]-self._y_lim[0])*(max(self._x_axis) - self.centroids[i][:, 0])/height + self._y_lim[0])
            centroid_coords[i][:, 1] = ((self._x_lim[1]-self._x_lim[0])*(self.centroids[i][:, 1] - min(self._y_axis))/width + self._x_lim[0])
        
        self.x = centroid_coords[:, 1]
        self.y = centroid_coords[:, 0]
        return None