"""
Created on Wed Feb 28 14:24:09 2024

@author: heckeh
"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

x_min = 0
xmax = 5
y_min = -1.
y_max = 1.25

img = io.imread("fig1.png")
X = img.reshape(-1, 4)
n_clusters = 3
max_iter = 800
debug_plot = False

kmeans = KMeans(n_clusters, n_init=10, init='k-means++')
kmeans.fit(X)
clusters = kmeans.predict(X)
seg_img = clusters.reshape(img.shape[0], img.shape[1])

layers_iso = np.zeros([seg_img.shape[0], seg_img.shape[1], n_clusters])
for k in range(n_clusters):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if seg_img[i, j] == k:
                layers_iso[i, j, k] = 1.

pts_iso = layers_iso[:, :, 1]
pts_iso_inds = np.zeros([1,2])
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if pts_iso[i, j] == 1.:
            pts_iso_inds = np.append(pts_iso_inds, [[i, j]], axis=0)
pts_iso_inds = np.delete(pts_iso_inds, 0, axis=0)

X_KNN_train, X_KNN_test = train_test_split(pts_iso_inds, test_size=0.25)

n = 30
dist_tol = 1.5
KNN = NearestNeighbors(n_neighbors=n)
centroid_validate = NearestNeighbors(n_neighbors=2)

centroids = X_KNN_test
centroids_new = np.zeros_like(centroids)
KNN.fit(X_KNN_train)

num_iter = 0
while num_iter <= max_iter:
    # print("iter:", num_iter)
    _, inds = KNN.kneighbors(centroids)
    centroids_new = np.zeros_like(centroids)
    
    for i in range(inds.shape[0]):
        for j in range(n):
            centroids_new[i, 0] += X_KNN_train[inds[i, j], 0]/n
            centroids_new[i, 1] += X_KNN_train[inds[i, j], 1]/n
        centroids_new[i, 0] = int(centroids_new[i, 0])
        centroids_new[i, 1] = int(centroids_new[i, 1])
    if np.array_equal(centroids, centroids_new):
        break
    
    centroid_validate.fit(centroids_new)
    
    centroids = centroids_new
    num_iter += 1
print("centroids:", centroids.shape)
centroids_xy = np.unique(centroids, axis=0)
print("strictly unique centroids:", centroids_xy.shape)

centroid_validate.fit(centroids_xy)
dist, cent_inds = centroid_validate.kneighbors(centroids_xy)
ind_del = []
for i in range(dist.shape[0]):
    if dist[i, 1] <= dist_tol:
        ind_del.append(cent_inds[i, 1])

ind = np.array(ind_del)
ind = np.unique(ind, axis=0)

centroids_xy = np.delete(centroids_xy, ind, axis=0)
print("unique centroids:", centroids_xy.shape)



centroids_plot = np.copy(pts_iso)
for i in range(centroids_xy.shape[0]):
    centroids_plot[int(centroids_xy[i, 0]), int(centroids_xy[i, 1])] = 2.


plt.figure(figsize=(7., 5.25))
io.imshow(centroids_plot)



