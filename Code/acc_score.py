# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:35:01 2024

@author: heckeh
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np

def calc_accuracy(true, pred):
    point_validate = NearestNeighbors(n_neighbors=1)
    point_validate.fit(true)
    dist, ind = point_validate.kneighbors(pred)
    
    acc = 0
    acc_no_pts = 0
    acc_multi_pts = 0
    
    points_so_far = []
    for i in range(pred.shape[0]):
        if ind[i] in points_so_far:
            acc_multi_pts += 1
            continue
        acc += get_dist(true[ind[i]], pred[i])
        points_so_far.append(ind[i])
    
    for i in range(true.shape[0]):
        if i not in points_so_far:
            acc_no_pts += 1
    
    return acc/pred.shape[0], acc_no_pts/true.shape[0], acc_multi_pts/true.shape[0]

def get_dist(x, y):
    return ((x[0, 0] - y[0])**2 + (x[0, 1] - y[1])**2)**0.5

def order_data(true, pred):
    assert true.size == pred.size, "warning: number of digitized plots not equal to number of input plots."
    plot_accs = np.ones([true.size, pred.size])
    pred_reorder = np.copy(pred)
    for j in range(true.size):
        for k in range(pred.size):
            acc, no_pts, multi_pts = calc_accuracy(true[j], pred[k])
            plot_accs[j, k] = acc
    for i in range(pred.size):
        min_ind = np.argmin(plot_accs[:, i])
        pred_reorder[i] = pred[min_ind]
    return true, pred_reorder, plot_accs