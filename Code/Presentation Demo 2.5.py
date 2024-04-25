# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 21:36:39 2024

@author: heckeh
Demo 2 part 2 for MANE-4962 presentation.
"""

from PD22 import PlotDigitizer
from gen_plot3 import RNG_Plot
from acc_score import order_data
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np

# Set seed for demo.
# seed = int(np.random.random()*1e6)
seed = 845680

# Initialize plot generator and random number generator for plot style and dpi.
rng = np.random.default_rng(seed)
style = rng.choice(plt.style.available)
pltr = RNG_Plot(seed=seed)

# Generate points for plot.
num_plots = 2
pltr.gen_pts(num_plots)
# pltr.gen_ellipse(num_plots=num_plots, phi=rng.integers(0, 180))

# Plot randomly generated points.
# dpi = rng.integers(5, 8)*10
dpi = 150
pltr.plot(plot_style=style, fig_dpi=dpi, save_fig=True)

# Initialize the PlotDigitizer class.
dgtzr = PlotDigitizer(pltr.fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=True)

# Digitize the randomly generated plot.
data = dgtzr._timed_execution("digitized plot in", dgtzr.fit_data, radius_factor=0.015)

# Reorder generated plot data and digitized plot data for visualization.
gen_plot_data, digitized_data, acc = order_data(pltr.data, data)

# Visualize digitized data in comparison with generated data.
plt.style.use("default")
plt.figure(figsize=(7., 5.25), dpi=150)
plt.title("Fit Data")
for j in range(num_plots):
    p = plt.plot(gen_plot_data[j][:,0], gen_plot_data[j][:,1], markerfacecolor=f"C{j}", markeredgecolor="None", marker=".", linestyle="None", markersize=20, alpha=0.5)
    p = plt.plot(digitized_data[j][:,0], digitized_data[j][:,1], f"C{j}", marker="+", linestyle="None", markersize=15)
plt.show()

# Output normalized digitization accuracy
x_range = (pltr._x_lim[1] - pltr._x_lim[0])
print(np.round((acc/x_range)*100, 3))