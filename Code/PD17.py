# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:14:47 2024

@author: henhe

v0.17: Added docstrings, cleaned up code and methods.

"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import numpy as np
from alive_progress import alive_bar
from acc_score import calc_accuracy

class PlotDigitizer:
    def __init__(self, fig_name, num_plots=1, xlim=[0, 1], ylim=[0, 1], verbose = False):
        """
        Initializes the PlotDigitizer object.

        Parameters
        ----------
        fig_name : str
            File name of figure to be digitized. .png and .jpg supported.
        num_plots : int, optional
            The number of different datasets plotted in the figure, separated by marker color. The default is 1.
        xlim : array, optional
            The x range of the plot. Must be a strictly increasing two element array. The default is [0, 1].
        ylim : array, optional
            The y range of the plot. Must be a strictly increasing two element array. The default is [0, 1].
        verbose : boolean, optional
            Determines the terminal output of the object. The default is False.

        Returns
        -------
        None.

        """
        self._verbose = verbose
        
        if fig_name.endswith(".jpg"):
            self._filetype = "jpg"
        elif fig_name.endswith(".png"):
            self._filetype = "png"
        else:
            assert self._filetype == "jpg" or self._filetype == "png", "Only .jpg and .png files supported."
        self._color_img = io.imread(fig_name)
        self._x_lim = xlim
        assert xlim[1] > xlim[0] and len(xlim)==2, "x range must be strictly increasing two element array."
        self._y_lim = ylim
        assert ylim[1] > ylim[0] and len(ylim)==2, "y range must be strictly increasing two element array."
        
        self._n_clusters = num_plots + 2
        self._points = []
        self._background = 0
        self._axes = 0
        
        self._create_clusters()
        
        return None
        
    def _create_clusters(self):
        """
        Creates the clusters used to digitize the plot.

        Returns
        -------
        None.

        """
        if self._verbose:
            print("Attempting to fit clusters...")
        n_init = 1
        self._fit_clusters(n_init)
        self._separate_clusters()
        self._detect_clusters()
        
        trial_iter = 1
        while len(self._points) != self._n_clusters-2 and trial_iter <= 5:
            if self._verbose:
                print("Clusters likely not fit correctly.\nAttempting to fit clusters...")
            n_init += 1
            trial_iter += 1
            self._fit_clusters(n_init)
            self._separate_clusters()
            self._detect_clusters()
        assert len(self._points) == self._n_clusters-2, "Clusters likely not separated correctly after 5 iterations. Aborting function..."
        self._get_preprocessed_clusters()
        
        return None
    
    def _fit_clusters(self, n_init):
        """
        Fits the color image provided using KMeans clustering, by pixel color. 
        The image is ideally split into clusters containing the background, axes, and each set of data.

        Parameters
        ----------
        n_init : int
            The number of iterations used to refine the KMeans clustering method.
            n_init=1 is most likely sufficient for adequate clustering.

        Returns
        -------
        None.

        """
        if self._verbose:
            print("fitting image...")
        kmeans = KMeans(self._n_clusters, n_init=n_init, init='k-means++')
        if self._filetype == "jpg":
            km = self._color_img.reshape(-1, 3)
        if self._filetype == "png":
            km = self._color_img.reshape(-1, 4)
        kmeans.fit(km)
        
        if self._verbose:
            print("creating clusters...")
        self.clusters = kmeans.predict(km)
        return None
        
    def _separate_clusters(self):
        """
        Separates the clusters created by _fit_clusters.
        A 3D tensor of each isolated cluster is created, returning 1 for each pixel in the cluster,
        and 0 otherwise.

        Returns
        -------
        None.

        """
        
        if self._verbose:
            print("separating clusters...")
        self.clusters_img = self.clusters.reshape(self._color_img.shape[0], self._color_img.shape[1])
        self._layered_clusters = np.zeros([self.clusters_img.shape[0], self.clusters_img.shape[1], self._n_clusters])
        for k in range(self._n_clusters):
            for i in range(self._layered_clusters.shape[0]):
                for j in range(self._layered_clusters.shape[1]):
                    if self.clusters_img[i, j] == k:
                        self._layered_clusters[i, j, k] = 1.
        return None
    
    def _detect_clusters(self, background_tol=0.4, axis_tol=0.75):
        """
        Detects the likely contents of each cluster. The background, axes, and each dataset are detected.

        Parameters
        ----------
        background_tol : float, optional
            The tolerance used to determine the background layer. The default is 0.4.
        axis_tol : float, optional
            The tolerance used to detect the axis layer. The default is 0.75.

        Returns
        -------
        None.

        """
        if self._verbose:
            print("detecting clusters...")
        
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
    
    def _get_preprocessed_clusters(self, noise_tol=1):
        """
        Preprocesses the points clusters, removing erroneous data clustered incorrectly, outside the axes.

        Parameters
        ----------
        noise_tol : int, optional
            The tolerance used to determine the amount of noise allowed or deleted. The default is 1.
            Equal to the number of pixels to the left and right of the detected axes that "noise" is allowed.

        Returns
        -------
        None.

        """
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
        """
        Retrieves one preprocessed cluster, for use in _fit_centroids.
        Separates only the indeces of the pixels in the image that belong to the cluster.

        Parameters
        ----------
        layer : int
            The index of the cluster layer to be retrieved.

        Returns
        -------
        inds : numpy array
            The isolated indeces of the requested layer.

        """
        # print("retreiving cluster...")
        layer_iso = self._layered_clusters[:, :, layer]
        inds = np.empty([1,2])
        for i in range(layer_iso.shape[0]):
            for j in range(layer_iso.shape[1]):
                if layer_iso[i, j] == 1.:
                    inds = np.append(inds, [[i, j]], axis=0)
        return inds
    
    def _fit_to_centers(self, centroids):
        """
        Fits the centroids to the centers of the detected points.

        Parameters
        ----------
        centroids : TYPE
            DESCRIPTION.

        Returns
        -------
        centroids_fit : TYPE
            DESCRIPTION.

        """
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
    
    def _create_kmeans_clusters(self, array):
        if self._verbose:
            print("fitting image...")
        kmeans = KMeans(self._n_plots+4, n_init=1, init='k-means++')
        if self._filetype == "jpg":
            km = self._color_img.reshape(-1, 3)
        if self._filetype == "png":
            km = self._color_img.reshape(-1, 4)
        kmeans.fit(km)
        if self._verbose:
            print("creating clusters...")
        clusters = kmeans.predict(km)
        return clusters