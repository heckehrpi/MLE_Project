
"""
Created on Thu Apr 11 19:36:01 2024

@author: henhe
Demo for PD22.
"""

from PD22 import PlotDigitizer
from gen_plot3 import RNG_Plot
from acc_score import calc_accuracy, order_data
import matplotlib.pyplot as plt
import numpy as np

num_plots = 2
seed = int(np.random.random()*1e6)
# seed = 261037

rng = np.random.default_rng(seed)
style = rng.choice(plt.style.available)
pltr = RNG_Plot(seed=seed)

pltr.gen_pts(num_plots)
# pltr.gen_ellipse(num_plots=num_plots, phi=rng.integers(0, 180))

dpi = rng.integers(5, 8)*10
pltr.plot(plot_style=style, fig_dpi=dpi, save_fig=True)

dgtzr = PlotDigitizer(pltr.fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=True)
data = dgtzr._timed_execution("digitized plot in", dgtzr.fit_data, radius_factor=0.015)

gen_plot_data, digitized_data, acc = order_data(pltr.data, data)

plt.style.use("default")
plt.figure(figsize=(7., 5.25), dpi=150)
plt.title("Fit Data")

for j in range(num_plots):
    p = plt.plot(gen_plot_data[j][:,0], gen_plot_data[j][:,1], markerfacecolor=f"C{j}", markeredgecolor="None", marker=".", linestyle="None", markersize=20, alpha=0.5)
    p = plt.plot(digitized_data[j][:,0], digitized_data[j][:,1], f"C{j}", marker="+", linestyle="None", markersize=15)

plt.show()
