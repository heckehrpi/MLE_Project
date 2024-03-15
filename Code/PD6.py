# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:19:16 2024

@author: heckeh
"""

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import gen_plot_RNG
import gen_plot

class PlotDigitizer:
    
    def __init__(self, num_plots=1, gen_seed=None):
        self._n_clusters = num_plots + 2
        # self._plotter = gen_plot_RNG.RNG_plot(num_plots=num_plots, seed=gen_seed)
        # self._plotter.plot(save_fig=True, fig_dpi=200)
        # self._plot_name = self._plotter.plot_name
        self._plotter = gen_plot.RNG_Plot()
        self._plotter.gen_pts()
        self._plotter.plot(save_fig=True)
        self._plot_name = self._plotter.fig_name
        return None
        
    def separate_clusters(self):
        color_img = io.imread(self._plot_name)
        kmeans_X = color_img.reshape(-1, 4)
        
        kmeans = KMeans(self._n_clusters, n_init=15, init='k-means++')
        kmeans.fit(kmeans_X)
        clusters = kmeans.predict(kmeans_X)
        
        kmeans_cluster_img = clusters.reshape(color_img.shape[0], color_img.shape[1])
        self._layered_clusters = np.zeros([kmeans_cluster_img.shape[0], kmeans_cluster_img.shape[1], self._n_clusters])
        for k in range(self._n_clusters):
            for i in range(self._layered_clusters.shape[0]):
                for j in range(self._layered_clusters.shape[1]):
                    if kmeans_cluster_img[i, j] == k:
                        self._layered_clusters[i, j, k] = 1.
        return None
    
    def detect_clusters(self, background_tol=0.9, axis_tol=0.5):
        self.separate_clusters()
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
        return None
    
    def get_preprocessed_clusters(self, noise_tol=1):
        self.detect_clusters()
        for layer in self._points:
            for i in range(self._layered_clusters[:, :, layer].shape[0]):
                for j in range(self._layered_clusters[:, :, layer].shape[1]):
                    if self._layered_clusters[i, j, layer] >= 0.5:
                        if i <= min(self._x_axis)+noise_tol or i >= max(self._x_axis)-noise_tol:
                            self._layered_clusters[i, j, layer] = 0
                        if j <= min(self._y_axis)+noise_tol or j >= max(self._y_axis)-noise_tol:
                            self._layered_clusters[i, j, layer] = 0
        return None
    
    def get_cluster(self, layer):
        layer_iso = self._layered_clusters[:, :, layer]
        inds = np.zeros([1,2])
        for i in range(layer_iso.shape[0]):
            for j in range(layer_iso.shape[1]):
                if layer_iso[i, j] == 1.:
                    inds = np.append(inds, [[i, j]], axis=0)
        inds = np.delete(inds, 0, axis=0)
        return inds
    
    
    def fit_centroids(self, plot=0, epochs=50, init_split=0.15, culling_radius_factor=0.005, debug_plot=False):
        assert plot <= self._n_clusters-2
        self.get_preprocessed_clusters()
        inds = self.get_cluster(self._points[plot])
        self._X_train, self.centroids = train_test_split(inds, test_size=init_split)
        self._centroid_validate = NearestNeighbors()
        self._radius = culling_radius_factor*self._layered_clusters.shape[0]
        
        
        self._num_iter = 0
        while self._num_iter <= epochs:
            self._num_iter += 1
            print("iter: ",self._num_iter)
            centroids_temp = np.copy(self.centroids)
            self.cull_centroids()
            self.fit_to_centers()
            if np.array_equal(centroids_temp, self.centroids):
                print("centroids unchanging at iter", self._num_iter)
                break
        
        if debug_plot:
            for i in range(self._n_clusters):
                plt.figure()
                io.imshow(self._layered_clusters[:, :, i])
                plt.title("Layer "+str(i))
        return None
    
    def fit_to_centers(self):
        self._centroid_validate.fit(self._X_train)
        rads, inds = self._centroid_validate.radius_neighbors(self.centroids, radius=3*self._radius)
        centroids_temp = np.zeros_like(self.centroids)
        for i in range(inds.size):
            for j in range(inds[i].size):
                centroids_temp[i, 0] += self._X_train[inds[i][j], 0]/inds[i].size
                centroids_temp[i, 1] += self._X_train[inds[i][j], 1]/inds[i].size
            centroids_temp[i, 0] = int(centroids_temp[i, 0])
            centroids_temp[i, 1] = int(centroids_temp[i, 1])
        self.centroids = np.unique(centroids_temp, axis=0)
        return None
    
    def cull_centroids(self):
        self._centroid_validate.fit(self.centroids)
        rads, inds = self._centroid_validate.radius_neighbors(self.centroids, radius=self._radius)
        inds_delete = np.array([])
        for i in range(inds.shape[0]):
            if inds[i] != np.array([]):
                inds[i] = np.delete(inds[i], 0)
                inds_delete = np.append(inds_delete, inds[i])
        inds_delete = np.unique(inds_delete, axis=0)
        if inds_delete != np.array([]):
            self.centroids = np.delete(self.centroids, inds_delete.astype(int), axis=0)
        return
    
    def get_centroid_coords(self):
        width = max(self._y_axis) - min(self._y_axis)
        height = max(self._x_axis) - min(self._x_axis)
        x_range = self._plotter._x_lim
        y_range = self._plotter._y_lim
        
        self.centroid_coords = np.zeros_like(self.centroids)
        self.centroid_coords[:, 0] = ((y_range[1]-y_range[0])*(max(self._x_axis) - self.centroids[:, 0])/height + y_range[0])
        self.centroid_coords[:, 1] = ((x_range[1]-x_range[0])*(self.centroids[:, 1] - min(self._y_axis))/width + x_range[0])
        
        self.x_centroids = self.centroid_coords[:, 1]
        self.y_centroids = self.centroid_coords[:, 0]
        return None
    
    def plot_digitized(self):
        plt.figure(dpi = 200)
        plt.plot(self.centroid_coords[:, 1], self.centroid_coords[:, 0], 'r.')
        return None