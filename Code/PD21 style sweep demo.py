# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 23:11:27 2024

@author: henhe
"""

from PD21 import PlotDigitizer
import matplotlib.pyplot as plt
from gen_plot import RNG_Plot
import numpy as np

num_plots = 2

for style in plt.style.available:
    print(style)
    seed = int(np.random.random()*1e6)
    pltr = RNG_Plot(seed=seed)
    pltr.gen_pts(num_plots)
    dpi = np.random.randint(10, 11)*10
    pltr.plot(plot_style=style, fig_dpi=dpi, save_fig=True)
    fig_name = pltr.fig_name
    
    dgtzr = PlotDigitizer(fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=False)
    
    plt.figure(figsize=(7., 5.25), dpi=250)
    plt.title(f"Fit Data of {style} Plot")
    for k in range(num_plots):
        plt.plot(pltr.data[k][:, 0], pltr.data[k][:, 1], ".", markersize=20)
    for k in range(len(dgtzr.cropped_layer_attributes["datasets_and_other"])):
        plot = dgtzr.cropped_layer_attributes["datasets_and_other"][k]
        cluster = dgtzr.cropped_preprocessed_clusters[:, :, dgtzr.cropped_layer_attributes["datasets_and_other"][k]]
        data = dgtzr.fit_data(cluster, radius_factor=0.01)
        plt.plot(data[:, 0], data[:, 1], "x", markersize=5)