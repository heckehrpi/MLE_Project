# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:01:52 2024

@author: henhe
Second drawnplot demo for MANE-4962 presentation.
"""

from AutoPD import PlotDigitizer
import matplotlib.pyplot as plt

num_plots = 3

# Initialize the PlotDigitizer class for drawn plot.
dgtzr = PlotDigitizer("drawnplot7.png", [0, 100], [0, 10], num_plots, verbose=True)

# Digitize the randomly generated plot.
data = dgtzr._timed_execution("digitized plot in", dgtzr.fit_data, radius_factor=0.005, denoise_padding=0.05)

plt.style.use("default")
plt.figure(figsize=(7., 5.25), dpi=150)
plt.imshow(dgtzr._color_img)
plt.title("Color Image")


import numpy as np
from scipy.optimize import curve_fit

def linear(x, m, b):
    return m * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def quartic(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d*x**1 + e

def exponential(x, a, b):
    return a * np.exp(b * x)

def fit_curve_auto(x, y):
    models = [linear, quadratic, cubic, quartic, exponential]
    best_params = None
    best_model = None
    best_error = float('inf')

    for model in models:
        try:
            params, _ = curve_fit(model, x, y)
            fitted_y = model(x, *params)
            error = np.sum((y - fitted_y)**2)
            if error < best_error:
                best_error = error
                best_params = params
                best_model = model
        except:
            pass

    return best_model, best_params

def y_func(x, w):
    y = np.zeros_like(x)
    for i in range(x.size):
        for j in range(w.size):
            y[i] += w[j]*x[i]**j
    return y


# Visualize digitized data.
plt.figure(figsize=(7., 5.25), dpi=150)
plt.title("Fit Data")
for j in range(num_plots):
    p = plt.plot(data[j][:,0], data[j][:,1], f"C{j}", marker="+", linestyle="None", markersize=15)
    best_model, best_params = fit_curve_auto(data[j][:,0], data[j][:,1])
    print("Best model:", best_model.__name__)
    print("Best parameters:", best_params)
    x_plot = np.linspace(0, 100, 100)
    plt.plot(x_plot, y_func(x_plot, np.flip(best_params)), linewidth=3)
    plt.xlim([0, 100])
    plt.ylim([0, 10])
    
    