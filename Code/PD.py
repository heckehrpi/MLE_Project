"""
Created on Mon Mar  4 09:41:51 2024

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
    
    def fit_centroids(self, max_iter=500, debug_plot=False, test_split=0.25, n_neighbors = 10, dist_tol=1.5):
        # assert max_iter >= 25
        self._max_iter = max_iter
        self._test_size = test_split
        self._n_neighbors = n_neighbors
        self._d_tol = dist_tol
        X_KNN_train, X_KNN_test = train_test_split(self._iso_inds, test_size=self._test_size)
        self._KNN = NearestNeighbors(n_neighbors=self._n_neighbors)
        self._centroid_validate = NearestNeighbors(n_neighbors=2)

        self.centroids = X_KNN_test
        self._KNN.fit(X_KNN_train)

        print(self._layer_iso.shape)
        num_iter = 0
        while num_iter <= max_iter:
            num_iter += 1
            print("iter:", num_iter, self.centroids.shape)
            _, inds = self._KNN.kneighbors(self.centroids)
            centroids_temp = np.zeros_like(self.centroids)
            
            for i in range(inds.shape[0]):
                for j in range(self._n_neighbors):
                    centroids_temp[i, 0] += X_KNN_train[inds[i, j], 0]/self._n_neighbors
                    centroids_temp[i, 1] += X_KNN_train[inds[i, j], 1]/self._n_neighbors
                centroids_temp[i, 0] = int(centroids_temp[i, 0])
                centroids_temp[i, 1] = int(centroids_temp[i, 1])
            if np.array_equal(self.centroids, centroids_temp):
                print("centroid arrays equal")
                break
            
            centroids_temp = np.unique(centroids_temp, axis=0)
            self.centroids = centroids_temp
            # self._centroid_validate.fit(self.centroids)
            # dist, inds = self._centroid_validate.kneighbors(self.centroids)
            # ind_del = []
            # if num_iter >= 25:
            #     for i in range(dist.shape[0]):
            #         if dist[i, 1] <= dist_tol:
            #             ind_del.append(inds[i, 1])
            #     if ind_del != []:
            #         ind_del = np.array(ind_del)
            #         ind_del = np.unique(ind_del, axis=0)
            #     self.centroids = np.delete(self.centroids, ind_del, axis=0)

        self._centroid_validate.fit(self.centroids)
        rad_inds, rad_dist = self._centroid_validate.radius_neighbors(self.centroids, radius=10)
        for i in range(rad_inds.shape[0]):
            self.centroids = np.delete(self.centroids, rad_inds[i], axis=0)

        centroid_plot = np.copy(self._layer_iso)
        for i in range(self.centroids.shape[0]):
            centroid_plot[int(self.centroids[i, 0]), int(self.centroids[i, 1])] = 2.
        plt.figure(figsize=(7., 5.25), dpi=200)
        io.imshow(centroid_plot)

    
# if debug_plot == True:
#     plt.figure(figsize=(7., 5.25))
#     io.imshow(seg_img)
    
#     plt.figure(figsize=(7., 5.25))
#     io.imshow(layers_iso[:, :, 0])
#     plt.title("1")
    
#     plt.figure(figsize=(7., 5.25))
#     io.imshow(layers_iso[:, :, 1])
#     plt.title("2")
    
#     plt.figure(figsize=(7., 5.25))
#     io.imshow(layers_iso[:, :, 2])
#     plt.title("3")

