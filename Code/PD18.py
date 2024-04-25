# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:06:35 2024

@author: henhe
v 0.18 redoing code again to be more cohearent, trying out the method of cropping input image
"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import numpy as np
from alive_progress import alive_bar

class PlotDigitizer:
    def __init__(self, fig_name, xlim, ylim, num_plots, verbose = False, debug=False):
        self._debug = debug
        self._verbose = verbose
        self._n_plots = num_plots
        self._color_img = io.imread(fig_name)
        self._x_lim = xlim
        self._y_lim = ylim
        self._layer_attributes = {"background": [], "plot_background": [], "axes": [], "datasets_and_other": [], "is_plot_background": False}
        self._cropped_layer_attributes = {"background": [], "axes": [], "datasets_and_other": []}
        if fig_name.endswith(".jpg") or fig_name.endswith(".jpeg"):
            self._filetype = "jpg"
        elif fig_name.endswith(".png"):
            self._filetype = "png"
        else:
            assert self._filetype == "jpg" or self._filetype == "png", "Only .jpg and .png filetypes supported."
        return None
    
    def _create_kmeans_clusters(self, image, n_clusters):
        kmeans = KMeans(n_clusters, n_init=1, init='k-means++', max_iter=1000)
        reshape_params = {"jpg": (-1, 3), "png": (-1, 4)}
        km = image.reshape(reshape_params[self._filetype])
        kmeans.fit(km)
        clusters = kmeans.predict(km)
        return clusters.reshape(image.shape[0], image.shape[1])
    
    def _create_isolated_clusters(self, clusters, n_clusters):
        isolated_clusters = np.zeros([clusters.shape[0], clusters.shape[1], n_clusters])
        for k in range(n_clusters):
            isolated_clusters[:, :, k] = (clusters == k).astype(int)
        return isolated_clusters
    
    def _classify_cluster_features(self, isolated_clusters, background_tol=0.25, axis_tol = 0.75,):
        avg = np.average(isolated_clusters, axis=(0,1))
        v_avg = np.average(isolated_clusters, axis=0)
        h_avg = np.average(isolated_clusters, axis=1)
        layer_attributes = {"background": [], "plot_background": [], "axes": [], "datasets_and_other": [], "is_plot_background": False}
        layer_attributes["background"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] >= background_tol and 1.0 in [np.max(v_avg[:, k]), np.max(h_avg[:, k])]]
        layer_attributes["plot_background"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] >= background_tol and 1.0 not in [np.max(v_avg[:, k]), np.max(h_avg[:, k])]]
        layer_attributes["axes"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] <= background_tol and (np.max(v_avg[:, k]) >= axis_tol or np.max(h_avg[:, k]) >= axis_tol)]
        layer_attributes["datasets_and_other"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] <= background_tol and (np.max(v_avg[:, k]) <= axis_tol or np.max(h_avg[:, k]) <= axis_tol)]
        return layer_attributes
    
    def _detect_plot_bounds(self, isolated_clusters, layer_attributes, background_tol=0.25, plot_background_tol=0.45, axis_tol = 0.75, buffer=10):
        x_bounds = int(isolated_clusters.shape[1]/2)*np.ones(2)
        y_bounds = int(isolated_clusters.shape[0]/2)*np.ones(2)
        for k in range(isolated_clusters.shape[2]):
            v_avg = np.average(isolated_clusters[:, :, k], axis=0)
            h_avg = np.average(isolated_clusters[:, :, k], axis=1)
            x_range = np.array([])
            y_range = np.array([])
            if k in layer_attributes["plot_background"]:
                x_range = np.array([i for i in range(v_avg.size) if v_avg[i] >= plot_background_tol])
                y_range = np.array([i for i in range(h_avg.size) if h_avg[i] >= plot_background_tol])
            if k in layer_attributes["axes"]:
                x_range = np.array([j for j in range(buffer, v_avg.size-buffer) for i in range(h_avg.size) if h_avg[i] >= axis_tol and np.average(isolated_clusters[i, j-buffer:j+buffer, k]) > 0.75])
                y_range = np.array([i for i in range(buffer, h_avg.size-buffer) for j in range(v_avg.size) if v_avg[j] >= axis_tol and np.average(isolated_clusters[i-buffer:i+buffer, j, k]) > 0.75])
            if x_range.size >= 0.25*isolated_clusters.shape[1]:
                x_bounds = np.array([min(x_bounds[0], np.min(x_range)), max(x_bounds[1], np.max(x_range))]) 
            if y_range.size >= 0.25*isolated_clusters.shape[0]:
                y_bounds = np.array([min(y_bounds[0], np.min(y_range)), max(y_bounds[1], np.max(y_range))]) 
        return x_bounds, y_bounds
    
    def _crop_image(self, image, x_bounds, y_bounds, padding=-5):
        return image[x_bounds[0]+padding:x_bounds[1]-padding, y_bounds[0]+padding:y_bounds[1]-padding]
    
    def _get_cluster(self, isolated_cluster):
        return np.argwhere(isolated_cluster[:, :] == 1.)
    
    def _preprocess_clusters(self, isolated_clusters, layer_attributes, padding=0.01, denoise_tol=0.5):
        padding_px = round(isolated_clusters.shape[0]*padding)
        preprocessed_clusters = np.copy(isolated_clusters)
        for k in range(isolated_clusters.shape[2]):
            preprocessed_clusters[:, :padding_px+1, k] = np.zeros_like(preprocessed_clusters[:, :padding_px+1, k])
            preprocessed_clusters[:, -padding_px+1:, k] = np.zeros_like(preprocessed_clusters[:, -padding_px+1:, k])
            preprocessed_clusters[:padding_px+1, :, k] = np.zeros_like(preprocessed_clusters[:padding_px+1, :, k])
            preprocessed_clusters[-padding_px+1:, :, k] = np.zeros_like(preprocessed_clusters[-padding_px+1:, :, k])
            preprocessed_clusters[:, :, k] = self._denoise(preprocessed_clusters[:, :, k], denoise_tol)
        
        return preprocessed_clusters
    
    def _denoise(self, cluster, denoise_tol):
        cluster_inds = self._get_cluster(cluster)
        denoiser = NearestNeighbors(radius=5)
        if cluster_inds.size == 0:
            return cluster
        denoiser.fit(cluster_inds)
        dist, ind = denoiser.radius_neighbors(cluster_inds)
        ind_delete = np.unique(np.array([ind[i][0] for i in range(ind.size) if ind[i].size <= 80*denoise_tol]))
        if ind_delete.size != 0:
            cluster_inds = np.delete(cluster_inds, ind_delete, axis=0)
        denoised_clusters = np.zeros_like(cluster)
        for i in range(cluster_inds.shape[0]):
            denoised_clusters[cluster_inds[i, 0], cluster_inds[i, 1]] = 1
        return denoised_clusters
    
    def _fit_centroids(self, clusters, layer_attributes, epochs=50, init_split=0.15, radius_factor=0.0075, denoise_tol=20):
        culling_radius = radius_factor*clusters.shape[0]
        self._centroid_validate = NearestNeighbors(radius=culling_radius)
        self._centroid_fit = NearestNeighbors(radius=2*culling_radius)
        centroids = np.zeros(len(layer_attributes["datasets_and_other"]), dtype=list)
        for plot in range(centroids.size):
            inds = self._get_cluster(layer_attributes["datasets_and_other"][plot])
            points_out, centroids[plot] = train_test_split(inds, test_size=init_split)
            num_iter = 1
            while num_iter <= epochs:
                print(num_iter)
                centroids_temp = np.copy(centroids[plot])
                centroids_temp, points_out = self._cull_centroids(centroids_temp, points_out)
                centroids_temp = self._fit_to_centers(centroids_temp, points_out)
                if np.array_equal(centroids_temp, self._centroids):
                    break
                centroids[plot] = centroids_temp
                num_iter += 1
        return None
    
    def _cull_centroids(self, centroids, points_out):
        centroids_culled = np.copy(centroids)
        self._centroid_validate.fit(centroids_culled)
        _, inds = self._centroid_validate.radius_neighbors(centroids_culled)
        inds_delete = np.array([])
        for i in range(inds.shape[0]):
            if len(inds[i]) != 0:
                inds_delete = np.append(inds_delete, inds[i][1:])
        inds_delete = np.unique(inds_delete, axis=0)
        if len(inds_delete) != 0:
            centroids_culled = np.delete(centroids_culled, inds_delete.astype(int), axis=0)
            points_out = np.append(points_out, inds_delete, axis=0)
        return centroids_culled, points_out
    
    def _fit_to_centers(self, centroids, points_out):
        self._centroid_fit.fit(points_out)
        _, inds = self._centroid_fit.radius_neighbors(centroids, radius=2*self._radius)
        centroids_fit = np.zeros_like(centroids)
        for i in range(inds.size):
            for j in range(inds[i].size):
                centroids_fit[i, 0] += points_out[inds[i][j], 0]/inds[i].size
                centroids_fit[i, 1] += points_out[inds[i][j], 1]/inds[i].size
        return np.unique(centroids_fit.astype(int), axis=0)
    
    def ensemble(self, n_solves=3, radius=0.001, epochs=50, init_split=0.15, radius_factor=0.0075, denoise_tol=20):
        self.history = np.zeros(n_solves, dtype=list)
        for i in range(n_solves):
            self._fit_centroids(epochs=epochs, init_split=init_split, radius_factor=radius_factor, denoise_tol=denoise_tol)
            data = self.get_centroid_coords()
            self.history[i] = data[0]
        return self.merge_arrays(self.history, radius=radius)
    
    def merge_arrays(self, x, radius):
        merge = np.empty([0, 2])
        for i in range(x.shape[0]):
            merge = np.vstack((merge, x[i]))
        ind_so_far = []
        merged_pts = np.empty([0, 2])
        pts_validate = NearestNeighbors()
        pts_validate.fit(merge)
        dist, ind = pts_validate.radius_neighbors(merge, radius=radius)
        for i in range(merge.shape[0]):
            if ind[i][0] not in ind_so_far:
                merged_pts = np.vstack((merged_pts, self.avg_pts(merge[ind[i], :])))
            for j in range(ind[i].size):
                ind_so_far.append(ind[i][j])
        return merged_pts
    
    
    
    
    
    
    
    
    
    
    
    
    
    