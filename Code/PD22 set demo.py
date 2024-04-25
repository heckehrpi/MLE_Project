# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 00:51:17 2024

@author: henhe
calculates relative accuracy scores for different configurations.
"""

from PD22 import PlotDigitizer
from gen_plot3 import RNG_Plot
from acc_score import calc_accuracy
import matplotlib.pyplot as plt
import numpy as np

num_takes = 2
num_trials = 2
num_plots = 5

num_failures = []
for take in range(num_takes):
    print("take", take+1)
    seed = int(np.random.random()*1e6)
    
    rng = np.random.default_rng(seed)
    style = rng.choice(plt.style.available)
    pltr = RNG_Plot(seed=seed)
    
    pltr.gen_pts(num_plots)
    # pltr.gen_ellipse(num_plots=num_plots, phi=rng.integers(0, 180))
    
    dpi = rng.integers(6, 10)*10
    pltr.plot(plot_style=style, fig_dpi=dpi, save_fig=True)
    
    dgtzr = PlotDigitizer(pltr.fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=False)
    
    avg_acc = 0
    avg_no_pts = 0
    avg_multi_pts = 0
    for i in range(num_trials):
        
        num_failures.append(0)
        data = dgtzr._timed_execution(f"trial {i+1} in", dgtzr.fit_data, radius_factor=0.015)
        trial_acc = 1e6
        trial_no_pts = 1e6
        trial_multi_pts = 1e6
        
        plt.style.use("default")
        plt.figure(figsize=(7., 5.25), dpi=250)
        plt.title("Fit Data")
        for k in range(num_plots):
            p = plt.plot(pltr.data[k][:, 0], pltr.data[k][:, 1], ".", markersize=20, alpha=0.5)
            
            for j in range(num_plots):
                acc, no_pts, multi_pts = calc_accuracy(pltr.data[k], data[k])
                if acc <= trial_acc:
                    trial_acc = acc
                    trial_no_pts = no_pts
                    trial_multi_pts = multi_pts
            avg_acc += trial_acc/(num_trials*num_plots*num_takes)
            avg_no_pts += trial_no_pts/(num_trials*num_plots*num_takes)
            avg_multi_pts += trial_multi_pts/(num_trials*num_plots*num_takes)
        for k in range(len(dgtzr.cropped_layer_attributes["datasets_and_other"])):
            if len(dgtzr._cropped_layer_attributes) != num_plots:
                num_failures[i] += 1
                break
            plt.plot(data[k][:, 0], data[k][:, 1], "+", markersize=10)
        plt.xlim(pltr._x_lim)
        plt.ylim(pltr._y_lim)
        # plt.savefig(f"gif_1_plot\\frame{i+1}.png")

print("failures:", num_failures)
print("average accuracy:", round(avg_acc/(pltr._x_lim[1]-pltr._x_lim[0]), 5))
print("average no point score:", round(avg_no_pts, 5))
print("average multi point score:", round(avg_multi_pts, 5))