# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:50:31 2024

@author: henhe
"""
from PD18 import PlotDigitizer
from gen_plot import RNG_Plot
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import acc_score

num_plots = 1
n_clusters = num_plots+4

plt.style.use("default")

dgtzr = PlotDigitizer("drawnplot.png", [0, 100], [0, 100], num_plots, verbose=True)
plt.figure(figsize=(7., 5.25), dpi=250)
plt.imshow(dgtzr._color_img)

uncropped_clusters = dgtzr._create_kmeans_clusters(dgtzr._color_img, n_clusters)
plt.figure(figsize=(7., 5.25), dpi=250)
plt.imshow(uncropped_clusters)

uncropped_isolated_clusters = dgtzr._create_isolated_clusters(uncropped_clusters, n_clusters)
for k in range(num_plots+4):
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.imshow(uncropped_isolated_clusters[:, :, k])
    plt.title("Layer "+str(k))
    
uncropped_layer_attributes = dgtzr._classify_cluster_features(uncropped_isolated_clusters)
print(uncropped_layer_attributes)

x_bounds, y_bounds = dgtzr._detect_plot_bounds(uncropped_isolated_clusters, uncropped_layer_attributes)
print("x_bounds:", x_bounds)
print("y_bounds:", y_bounds)

cropped_image= dgtzr._crop_image(dgtzr._color_img, y_bounds, x_bounds, padding=-3)
plt.figure(figsize=(7., 5.25), dpi=250)
plt.imshow(cropped_image)

cropped_clusters = dgtzr._create_kmeans_clusters(cropped_image, num_plots+1)
plt.figure(figsize=(7., 5.25), dpi=250)
plt.imshow(cropped_clusters)

cropped_isolated_clusters = dgtzr._create_isolated_clusters(cropped_clusters, num_plots+1)
cropped_layer_attributes = dgtzr._classify_cluster_features(cropped_isolated_clusters)
preprocessed_cropped_clusters = dgtzr._preprocess_clusters(cropped_isolated_clusters, cropped_layer_attributes, padding=0.01)
for k in range(num_plots+1):
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.imshow(cropped_isolated_clusters[:, :, k])
    plt.title("Layer "+str(k))
    
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.imshow(preprocessed_cropped_clusters[:, :, k])
    plt.title("Preprocessed Layer "+str(k))
        
print(cropped_layer_attributes)

