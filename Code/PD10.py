# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:27:36 2024

@author: heckeh
"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from acc_score import calc_accuracy

class PlotDigitizer:
    
    def __init__(self, fig_name, xlim, ylim, num_plots, verbose = False, debug=False):
        self._debug = debug
        self._verbose = verbose
        self._n_clusters = num_plots + 2
        self.color_img = io.imread(fig_name)
        self._x_lim = xlim
        self._y_lim = ylim
        
        kmeans_X = self.color_img.reshape(-1, 3)
        self.kmeans = KMeans(self._n_clusters, n_init=10, init='k-means++')
        print("fitting image...")
        self.kmeans.fit(kmeans_X)
        print("creating clusters...")
        self.clusters = self.kmeans.predict(kmeans_X)
        self._separate_clusters()
        self._detect_clusters()
        self._get_preprocessed_clusters()
        return None
        
    def _separate_clusters(self):
        print("separating clusters...")
        self.clusters_img = self.clusters.reshape(self.color_img.shape[0], self.color_img.shape[1])
        self._layered_clusters = np.zeros([self.clusters_img.shape[0], self.clusters_img.shape[1], self._n_clusters])
        for k in range(self._n_clusters):
            for i in range(self._layered_clusters.shape[0]):
                for j in range(self._layered_clusters.shape[1]):
                    if self.clusters_img[i, j] == k:
                        self._layered_clusters[i, j, k] = 1.
        return None
    
    def _detect_clusters(self, background_tol=0.9, axis_tol=0.75):
        print("detecting clusters...")
        self._points = []
        
        for k in range(self._n_clusters):
            avg = np.average(self._layered_clusters[:, :, k])
            if avg >= background_tol:
                self._background = k
                
        for k in range(self._n_clusters):
            if k != self._background:
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
                    self._axes = k
                    self._x_axis = x_axis
                    self._y_axis = y_axis
                if len(x_axis) == 0 and len(y_axis) == 0:
                    self._points.append(k)
        assert len(self._points) == self._n_clusters-2, "Clusters likely not separated correctly"
        return None
    
    def _get_preprocessed_clusters(self, noise_tol=1):
        print("preprocessing clusters...")
        for layer in self._points:
            for i in range(self._layered_clusters[:, :, layer].shape[0]):
                for j in range(self._layered_clusters[:, :, layer].shape[1]):
                    if self._layered_clusters[i, j, layer] >= 0.5:
                        if i <= min(self._x_axis)+noise_tol or i >= max(self._x_axis)-noise_tol:
                            self._layered_clusters[i, j, layer] = 0
                        if j <= min(self._y_axis)+noise_tol or j >= max(self._y_axis)-noise_tol:
                            self._layered_clusters[i, j, layer] = 0
        return None
    
    def _get_cluster(self, layer):
        # print("retreiving cluster...")
        layer_iso = self._layered_clusters[:, :, layer]
        inds = np.zeros([1,2])
        for i in range(layer_iso.shape[0]):
            for j in range(layer_iso.shape[1]):
                if layer_iso[i, j] == 1.:
                    inds = np.append(inds, [[i, j]], axis=0)
        inds = np.delete(inds, 0, axis=0)
        return inds
    
    
    def _fit_centroids(self, epochs=50, init_split=0.15, radius_factor=0.0075, denoise_tol=20):
        # print("fitting centroids...")
        
        self._denoise_tol = denoise_tol
        self._centroid_validate = NearestNeighbors()
        self._centroid_denoise = NearestNeighbors()
        self._radius = radius_factor*self._layered_clusters.shape[0]
        self.acc_history = np.zeros(epochs)
        
        self._centroids = np.zeros(len(self._points), dtype=list)
        
        for plot in range(len(self._points)):
            inds = self._get_cluster(self._points[plot])
            self._X_train, self._centroids[plot] = train_test_split(inds, test_size=init_split)
        
            num_iter = 1
            with alive_bar(epochs, length=20) as bar:
                while num_iter <= epochs:
                    bar()
                    centroids_temp = np.copy(self._centroids[plot])
                    centroids_temp = self._cull_centroids(centroids_temp)
                    centroids_temp = self._fit_to_centers(centroids_temp)
                    if np.array_equal(centroids_temp, self._centroids):
                        print("centroids unchanging at iter", num_iter)
                        break
                    self._centroids[plot] = centroids_temp
                    num_iter += 1
        return None
    
    def _fit_to_centers(self, centroids):
        # print("    fitting centroids to centers...")
        self._centroid_validate.fit(self._X_train)
        rads, inds = self._centroid_validate.radius_neighbors(centroids, radius=2*self._radius)
        centroids_fit = np.zeros_like(centroids)
        for i in range(inds.size):
            for j in range(inds[i].size):
                centroids_fit[i, 0] += self._X_train[inds[i][j], 0]/inds[i].size
                centroids_fit[i, 1] += self._X_train[inds[i][j], 1]/inds[i].size
            centroids_fit[i, 0] = int(centroids_fit[i, 0])
            centroids_fit[i, 1] = int(centroids_fit[i, 1])
        centroids_fit = np.unique(centroids_fit, axis=0)
        return centroids_fit
    
    def _cull_centroids(self, centroids):
        # print("    culling centroids...")
        centroids_culled = np.copy(centroids)
        self._centroid_validate.fit(centroids_culled)
        self._centroid_denoise.fit(self._X_train)
        _, inds = self._centroid_validate.radius_neighbors(centroids_culled, radius=self._radius)
        _, inds_denoise = self._centroid_denoise.radius_neighbors(centroids_culled, radius=self._radius)
        inds_delete = np.array([])
        for i in range(inds.shape[0]):
            if inds_denoise[i].size <= self._denoise_tol:
                inds_delete = np.append(inds_delete, i)
            if inds[i] != np.array([]):
                inds[i] = np.delete(inds[i], 0)
                inds_delete = np.append(inds_delete, inds[i])
        inds_delete = np.unique(inds_delete, axis=0)
        if inds_delete != np.array([]):
            centroids_culled = np.delete(centroids_culled, inds_delete.astype(int), axis=0)
        return centroids_culled
    
    def get_centroid_coords(self):
        
        data = np.zeros(len(self._points), dtype=list)
        
        width = max(self._y_axis) - min(self._y_axis)
        height = max(self._x_axis) - min(self._x_axis)
        x_range = self._x_lim
        y_range = self._y_lim
        
        
        for i in range(len(self._points)):
            temp = np.zeros_like(self._centroids[i])
            temp[:, 1] = ((y_range[1]-y_range[0])*(max(self._x_axis) - self._centroids[i][:, 0])/height + y_range[0])
            temp[:, 0] = ((x_range[1]-x_range[0])*(self._centroids[i][:, 1] - min(self._y_axis))/width + x_range[0])
            
            data[i] = temp
        return data