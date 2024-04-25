# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:21:20 2024

@author: heckeh
Drawn plot demo for MANE-4962 presentation.
"""

from AutoPD import PlotDigitizer
import matplotlib.pyplot as plt

num_plots = 2

# Initialize the PlotDigitizer class for drawn plot.
dgtzr = PlotDigitizer("drawnplot2.png", [0, 100], [0, 10], num_plots, verbose=True)

# Digitize the randomly generated plot.
data = dgtzr._timed_execution("digitized plot in", dgtzr.fit_data, radius_factor=0.02, denoise_padding=0.03)

plt.style.use("default")
plt.figure(figsize=(7., 5.25), dpi=150)
plt.imshow(dgtzr._color_img)
plt.title("Color Image")

# Visualize digitized data.
plt.figure(figsize=(7., 5.25), dpi=150)
plt.title("Fit Data")
for j in range(num_plots):
    p = plt.plot(data[j][:,0], data[j][:,1], f"C{j}", marker="+", linestyle="None", markersize=15)
plt.show()