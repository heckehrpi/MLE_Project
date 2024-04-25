# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:21:49 2024

@author: heckeh
"""
from PD16 import PlotDigitizer
from gen_plot import RNG_Plot
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import acc_score

num_plots = 3
# seed = 692605
avg_num_plots = 5
avg_num_trials = 1

avg_acc_score = 0
avg_no_pts = 0
avg_multi_pts = 0


for i in range(avg_num_plots):
    seed = int(np.random.random()*1e6)
    print("Plot:", i+1)
    
    pltr = RNG_Plot(seed=seed)
    pltr.gen_pts(num_plots)
    pltr.plot(fig_dpi=150, save_fig=True)
    fig_name = pltr.fig_name
    dgtzr = PlotDigitizer(fig_name, pltr._x_lim, pltr._y_lim, num_plots, verbose=True)
    # # dgtzr = PlotDigitizer("exam_plot.png", [20,160], [2.5,8.5], 1)
    # plt.figure(figsize=(7., 5.25), dpi=250)
    # io.imshow(dgtzr._color_img)
    # for i in range(dgtzr._n_clusters):
    #     plt.figure(figsize=(7., 5.25))
    #     io.imshow(dgtzr._layered_clusters[:, :, i])
    #     plt.title(i)
    
    n_solves = 5
    
    for j in range(avg_num_trials):
        print("Trial:", j+1)
        # dgtzr._fit_centroids(epochs=25, radius_factor=0.01, init_split=0.05)            
        # data= dgtzr.get_centroid_coords()
        data = dgtzr.ensemble(n_solves=n_solves, radius=0.25, epochs=25, radius_factor=0.007, init_split=0.05)
        
        for plot in range(num_plots):
            pred = data.astype(np.float32)
            for trial in range(n_solves):
                acc, acc_no_pts, acc_multi_pts = acc_score.calc_accuracy(dgtzr.history[trial], pred)
                avg_acc_score += acc/(avg_num_plots*avg_num_trials*n_solves)
                plt.figure(dpi=250)
                plt.plot(pltr.data[plot][:, 0], pltr.data[plot][:, 1], '.', markersize=10)
                plt.plot(dgtzr.history[trial][:, 0], dgtzr.history[trial][:, 1], 'g+', markersize=10)
            
            acc, acc_no_pts, acc_multi_pts= acc_score.calc_accuracy(pltr.data[plot], pred)
            plt.figure(dpi=250)
            plt.title("Final Ensemble Solve")
            plt.plot(pltr.data[plot][:, 0], pltr.data[plot][:, 1], '.', markersize=10)
            plt.plot(data[:, 0], data[:, 1], 'g+', markersize=10)

print("Average Positional Accuracy Score:", round(avg_acc_score, 3))
print("Average No Points Score:", round(avg_no_pts, 3))
print("Average Multi Points Score:", round(avg_multi_pts, 3))