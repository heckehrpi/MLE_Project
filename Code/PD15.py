# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:33:28 2024

@author: heckeh
tried to clean up the code again
attempt 3 to implement an "ensemble" to hopefully decrease # of multi/no pts assuming only 1 plot for now
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
        self._color_img = io.imread(fig_name)
        self._x_lim = xlim
        self._y_lim = ylim
        self._points = []
        self._background = 0
        self._axes = 0
        
        return None
    
    def _create_clusters(self):
        if self._verbose:
            print("Attempting to fit clusters...")
        n_init = 1
        self._fit_clusters(n_init)
        self._separate_clusters()
        self._detect_clusters()
        
        trial_iter = 0
        while len(self._points) != self._n_clusters-2 and trial_iter <= 15:
            if self._verbose:
                print("Clusters likely not fit correctly.\nAttempting to fit clusters...")
            n_init += 1
            self._fit_clusters(n_init)
            self._separate_clusters()
            self._detect_clusters()
            
        assert len(self._points) == self._n_clusters-2, "Clusters likely not separated correctly after 15 iterations. Aborting function..."
        self._get_preprocessed_clusters()
        return None
    
    def _fit_clusters(self, n_init):
        if self._verbose:
            print("fitting image...")
        kmeans = KMeans(self._n_clusters, n_init=n_init, init='k-means++')
        kmeans.fit(self._color_img.reshape(-1, 3))
        if self._verbose:
            print("creating clusters...")
        self.clusters = kmeans.predict(self._color_img.reshape(-1, 3))
        
    def _separate_clusters(self):
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
