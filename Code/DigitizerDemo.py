# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:21:49 2024

@author: heckeh
"""
from PD7 import PlotDigitizer
from gen_plot import RNG_Plot
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

num_plots = 2
# seed = 268792
seed = int(np.random.random()*1e6)

pltr = RNG_Plot(seed=seed)
pltr.gen_pts(num_plots)
pltr.plot(fig_dpi=80, save_fig=True)
# fig_name = pltr.fig_name
# dgtzr = PlotDigitizer(fig_name, pltr._x_lim, pltr._y_lim, num_plots)
# # dgtzr = PlotDigitizer("exam_plot.png", [20,160], [2.5,8.5], 1)


# x, y = dgtzr.get_centroid_coords()

# # plt.figure(figsize=(7., 5.25), dpi=250)
# # io.imshow(dgtzr.color_img)
# # for i in range(dgtzr._n_clusters):
# #     plt.figure(figsize=(7., 5.25))
# #     io.imshow(dgtzr._layered_clusters[:, :, i])
# #     plt.title(i)

# plt.figure(dpi=250)
# plt.title
# for plot in range(num_plots):
#     plt.plot(pltr.data[plot][:, 0], pltr.data[plot][:, 1], '.', markersize=10)
# for plot in range(num_plots):
#     plt.plot(x[plot], y[plot], 'r.', markersize=3)
#     print(x[plot].size, "point plot")
