# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:43:38 2024

@author: henhe
"""
from PD21 import PlotDigitizer
import matplotlib.pyplot as plt
from gen_plot2 import RNG_Plot
import numpy as np

num_plots = 1

# plt.style.use("default")
# seed = int(np.random.random()*1e6)
# pltr = RNG_Plot(seed=seed)
# pltr.gen_pts(num_plots)
# dpi = np.random.randint(10, 11)*10
# pltr.plot(fig_dpi=dpi, save_fig=True)
# fig_name = pltr.fig_name

dgtzr = PlotDigitizer("drawnplot.png", [0, 8], [0, 30], num_plots, verbose=True)
# dgtzr = PlotDigitizer(fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=True)
# data = dgtzr._fit_centroids(cluster, epochs=10)

# for k in range(dgtzr.uncropped_isolated_clusters.shape[2]):
#     plt.figure(figsize=(7., 5.25), dpi=250)
#     plt.imshow(dgtzr.uncropped_isolated_clusters[:, :, k])
#     plt.title("Layer "+str(k))

# for k in dgtzr.cropped_layer_attributes["datasets_and_other"]:
for k in range(dgtzr.cropped_preprocessed_clusters.shape[2]):
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.imshow(dgtzr.cropped_preprocessed_clusters[:, :, k])
    avg = np.average(dgtzr.cropped_preprocessed_clusters[:, :, k])
    plt.title("Layer "+str(k)+"\n avg: "+str(round(avg, 5)))

plt.figure(figsize=(7., 5.25), dpi=250)
plt.title("fit data")
plt.xlim([0, 8])
plt.ylim([0, 30])
# # for k in range(num_plots):
# #     plt.plot(pltr.data[k][:, 0], pltr.data[k][:, 1], ".", markersize=20)

for k in range(len(dgtzr.cropped_layer_attributes["datasets_and_other"])):
    plot = dgtzr.cropped_layer_attributes["datasets_and_other"][k]
    cluster = dgtzr.cropped_preprocessed_clusters[:, :, dgtzr.cropped_layer_attributes["datasets_and_other"][k]]
    data = dgtzr.fit_data(cluster, radius_factor=0.02)
    plt.plot(data[:, 0], data[:, 1], "X", markersize=10)

# plt.figure(figsize=(7., 5.25), dpi=250)
# plt.imshow(dgtzr.uncropped_clusters)


    
# print(dgtzr.uncropped_layer_attributes)

# plt.figure(figsize=(7., 5.25), dpi=250)
# plt.imshow(dgtzr.cropped_image)

# plt.figure(figsize=(7., 5.25), dpi=250)
# plt.imshow(dgtzr.cropped_clusters)

    





