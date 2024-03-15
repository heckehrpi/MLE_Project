# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:43:27 2024

@author: heckeh
"""

import PD6 as PD
import matplotlib.pyplot as plt
from skimage import io

n_plots = 1

plt.figure(dpi=200)
digitizer = PD.PlotDigitizer(num_plots=n_plots, gen_seed = None)
for i in range(n_plots):
    digitizer.fit_centroids(plot = i, epochs=20, init_split=0.33, culling_radius_factor=0.004, debug_plot=False)
    digitizer.get_centroid_coords()
    plt.plot(digitizer.x_centroids, digitizer.y_centroids, '.')