# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:23:37 2024

@author: henhe
"""

from PD18 import PlotDigitizer
from gen_plot import RNG_Plot
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import acc_score

num_plots = 3
n_clusters = num_plots+4

# plt.style.use("default")
# dgtzr = PlotDigitizer("exam_plot.png", [0, 25], [0, 10], 1, verbose=True)
# clusters = dgtzr._create_kmeans_clusters(dgtzr._color_img, n_clusters)

# # io.imshow(dgtzr._color_img)
# # plt.figure(figsize=(7., 5.25), dpi=250)
# # plt.imshow(clusters)



# x_bounds, y_bounds = dgtzr._detect_plot_bounds(isolated_clusters)
# cropped_image= dgtzr._crop_image(dgtzr._color_img, y_bounds, x_bounds, padding=-5)

# plt.figure(figsize=(7., 5.25), dpi=250)
# plt.imshow(dgtzr._color_img)
# plt.figure(figsize=(7., 5.25), dpi=250)
# plt.imshow(cropped_image)


for style in plt.style.available:
    print(style)
    plt.style.use(style)
    seed = int(np.random.random()*1e6)
    pltr = RNG_Plot(seed=seed)
    pltr.gen_pts(num_plots)
    dpi = np.random.randint(10, 20)*10
    pltr.plot(fig_dpi=dpi, save_fig=True)
    fig_name = pltr.fig_name
    dgtzr = PlotDigitizer(fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=True)
    clusters = dgtzr._create_kmeans_clusters(dgtzr._color_img, n_clusters)
    isolated_clusters = dgtzr._create_isolated_clusters(clusters, n_clusters)
    x_bounds, y_bounds = dgtzr._detect_plot_bounds(isolated_clusters)
    print("x_bounds:", x_bounds)
    print("y_bounds:", y_bounds)
    cropped_image= dgtzr._crop_image(dgtzr._color_img, y_bounds, x_bounds, padding=-8)
    cropped_clusters = dgtzr._create_kmeans_clusters(cropped_image, num_plots+1)
    cropped_isolated_clusters = dgtzr._create_isolated_clusters(cropped_clusters, num_plots+5)
    dgtzr._detect_clusters(cropped_isolated_clusters)
    
    plt.style.use("default")
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.imshow(dgtzr._color_img)
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.imshow(cropped_image)
    
    for i in range(num_plots+5):
        plt.figure(figsize=(7., 5.25))
        plt.imshow(cropped_isolated_clusters[:, :, i])
        plt.title(i)

# dgtzr = PlotDigitizer("exam_plot.png", [20,160], [2.5,8.5], 1)

