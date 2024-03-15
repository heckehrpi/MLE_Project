from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import gen_plot_RNG

class PlotDigitizer:
    
    def __init__(self, num_plots=1, random_seed=None):
        self.plotter = gen_plot_RNG.RNG_plot(seed=random_seed)
        self.plot_name = self.plotter.plot_name
        self._num_plots = num_plots
        
    def separate_clusters(self, fig_dpi=200):
        self.plotter.plot(save_fig=True, fig_dpi=200)
    
        img = io.imread(self.plot_name)
        img_reshape = img.reshape(-1, 4)
        n_clusters = self._num_plots+2
        
        kmeans = KMeans(n_clusters, n_init=10, init='k-means++')
        kmeans.fit(img_reshape)
        clusters = kmeans.predict(img_reshape)
        
        img_clusters = clusters.reshape(img.shape[0], img.shape[1])
        self._clusters = np.zeros([img_clusters.shape[0], img_clusters.shape[1], n_clusters])
        for k in range(n_clusters):
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img_clusters[i, j] == k:
                        self._clusters[i, j, k] = 1.
                        
        self._clusters_iso = self._clusters[:, :, 1]
        self._inds_iso = np.array([])
        for i in range(self._clusters.shape[0]):
            for j in range(self._clusters.shape[1]):
                if self._clusters_iso[i, j] == 1.:
                    self._inds_iso = np.append(self._inds_iso, [[i, j]], axis=0)
        
    def get_far_apart_centroids(self):
        num_iter = 0
        
        while num_iter <= self._max_iter:
            num_iter += 1
            self._centroid_validate.fit(self.centroids)
            rad_dist, rad_inds = self._centroid_validate.radius_neighbors(self.centroids, radius=self._radius)
            centroids_temp = np.copy(self.centroids)
            for i in range(rad_inds.shape[0]):
                if rad_inds[i] != np.array([]):
                    rad_inds[i] = np.delete(rad_inds[i], 0)
                    centroids_temp = np.delete(self.centroids, rad_inds[i], axis=0)
                    break
            if np.array_equal(centroids_temp, self.centroids):
                break
        return centroids_temp