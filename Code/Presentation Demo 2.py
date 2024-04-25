# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:56:26 2024

@author: heckeh
Demo 2 for MANE-4962 presentation.
"""

from AutoPD import PlotDigitizer
from gen_plot3 import RNG_Plot
import matplotlib.pyplot as plt
import numpy as np

dpis = [50, 80, 100, 120, 150, 180, 200, 250, 300]
num_trials = 1

# Iterate for desired number of datasets.
for dpi in dpis:
    print("dpi:", dpi)
    
    # Iterate for number of trials.
    for trial in range(num_trials):
        
        # Set seed for demo.
        seed = int(np.random.random()*1e6)
        
        # Initialize plot generator and random number generator for plot style and dpi.
        rng = np.random.default_rng(seed)
        style = rng.choice(plt.style.available)
        pltr = RNG_Plot(seed=seed)
        
        # Generate points for plot.
        num_plots = 1
        pltr.gen_pts(num_plots)
        # pltr.gen_ellipse(num_plots=num_plots, phi=rng.integers(0, 180))
        
        # Plot randomly generated points.
        pltr.plot(plot_style=style, fig_dpi=dpi, save_fig=True)
        
        # Initialize the PlotDigitizer class.
        dgtzr = PlotDigitizer(pltr.fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=False)
        
        # Digitize the randomly generated plot.
        data = dgtzr._timed_execution(f"digitized plot {trial+1} in", dgtzr.fit_data, radius_factor=0.022)
        
        # Visualize digitized data.
        plt.style.use("default")
        plt.figure(figsize=(7., 5.25), dpi=150)
        plt.title("Fit Data")
        for j in range(num_plots):
            p = plt.plot(pltr.data[j][:,0], pltr.data[j][:,1], markeredgecolor="None", marker=".", linestyle="None", markersize=20, alpha=0.35)
        for j in range(len(dgtzr.cropped_layer_attributes["datasets_and_other"])):
            p = plt.plot(data[j][:,0], data[j][:,1], "k", marker=("$"+str(j+1)+"$"), linestyle="None", markersize=5, alpha=0.5)
        plt.show()
    
    