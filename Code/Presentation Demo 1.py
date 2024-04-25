# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 20:45:50 2024

@author: heckeh
Demo 1 for MANE-4962 presentation.
"""

from AutoPD import PlotDigitizer
from gen_plot3 import RNG_Plot
from acc_score import order_data
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np

# Set seed for demo.
seed = int(np.random.random()*1e6)
# seed = 419025

# Initialize plot generator and random number generator for plot style and dpi.
rng = np.random.default_rng(seed)
style = rng.choice(plt.style.available)
# style = "Solarize_Light2"
pltr = RNG_Plot(seed=seed)

# Generate points for plot.
num_plots = 1
pltr.gen_pts(num_plots=num_plots, pts_min=140, pts_max=150)
# pltr.gen_ellipse(num_plots=num_plots, phi=rng.integers(0, 180))

# Plot randomly generated points.
dpi = 80
pltr.plot(plot_style=style, fig_dpi=dpi, save_fig=True)

# Initialize the PlotDigitizer class.
dgtzr = PlotDigitizer(pltr.fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=True)

# Digitize the randomly generated plot.
data = dgtzr._timed_execution("digitized plot in", dgtzr.fit_data, radius_factor=0.01, denoise_padding=0.03)

# Display digitization process
plt.style.use("default")

plt.figure(figsize=(7., 5.25), dpi=150)
plt.imshow(dgtzr._color_img)
plt.title("Color Image")

plt.figure(figsize=(7., 5.25), dpi=150)
plt.imshow(dgtzr.uncropped_clusters)
plt.title("Clustered Image")

for i in range(dgtzr.uncropped_isolated_clusters.shape[2]):    
    plt.figure(figsize=(7., 5.25), dpi=150)
    plt.imshow(dgtzr.uncropped_isolated_clusters[:, :, i], cmap="Greys_r")
    plt.title(f"Uncropped and Isolated Cluster {i}")

plt.figure(figsize=(7., 5.25), dpi=150)
plt.imshow(dgtzr.cropped_image)
plt.title("Cropped Image")

plt.figure(figsize=(7., 5.25), dpi=150)
plt.imshow(dgtzr.cropped_clusters)
plt.title("Clustered Image")

for i in range(dgtzr.cropped_isolated_clusters.shape[2]):    
    plt.figure(figsize=(7., 5.25), dpi=150)
    plt.imshow(dgtzr.cropped_isolated_clusters[:, :, i], cmap="Greys_r")
    plt.title(f"Cropped and Isolated Cluster {i}")    
    plt.figure(figsize=(7., 5.25), dpi=150)
    plt.imshow(dgtzr.cropped_preprocessed_clusters[:, :, i], cmap="Greys_r")
    plt.title(f"Cropped and Preprocessed Cluster {i}")

# Reorder generated plot data and digitized plot data for visualization.
gen_plot_data, digitized_data, acc = order_data(pltr.data, data)

# Visualize digitized data in comparison with generated data.
plt.figure(figsize=(7., 5.25), dpi=150)
plt.title("Fit Data")
for j in range(num_plots):
    p = plt.plot(gen_plot_data[j][:,0], gen_plot_data[j][:,1], markerfacecolor=f"C{j}", markeredgecolor="None", marker=".", linestyle="None", markersize=20, alpha=0.5)
    p = plt.plot(digitized_data[j][:,0], digitized_data[j][:,1], f"C{j}", marker="+", linestyle="None", markersize=15)
plt.legend(["Randomly Generated Points", "Fit Points"])
plt.show()

# Output normalized digitization accuracy
x_range = (pltr._x_lim[1] - pltr._x_lim[0])
print("Normalized digitization accuracy:")
print(np.round((acc/x_range), 5))