# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:35:52 2024

@author: heckeh
"""
from matplotlib.style import available
import matplotlib.pyplot as plt
import numpy as np

class RNG_Plot:
    
    def __init__(self, seed=None):
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        return None
        
    def gen_polynomial(self, num_plots, pts_min=10, pts_max=50, alpha=5, mu=0, sigma=1, x_min=-5., x_max=5., buffer=0.9):
        self._num_plots = num_plots
        self._x_lim = np.array([x_min, x_max])
        self._num_pts = self._rng.integers(low=pts_min, high=pts_max)
        self.data = np.zeros(self._num_plots, dtype=list)
        ceil = np.zeros(self._num_plots)
        floor = np.zeros(self._num_plots)
        self.w = np.zeros(self._num_plots, dtype=list)
        for i in range(self._num_plots):
            data = np.zeros([self._num_pts, 2])
            data[:, 0] = buffer*((x_max - x_min)*self._rng.random(self._num_pts) + x_min)
            order = self._rng.integers(low=2, high=4)
            self.w[i] = self._rng.normal(mu, sigma, size=order)
            data[:, 1] = self._y_func(data[:, 0], self.w[i], alpha)
            ceil[i] = np.max(np.ceil(data[:, 1]))
            floor[i] = np.min(np.floor(data[:, 1]))
            self.data[i] = data
        self._y_lim = np.array([np.min(floor), np.max(ceil)])
        return None
    
    def gen_ellipse(self, num_plots=2, a=2, b=1, focus=[0,0], num_pts=100, phi=0):
        assert a >= b, "semi-major axis must be larger than semi-minor axis."
        self._num_plots = num_plots
        self._num_pts = num_pts
        
        theta_ranges = 2*np.pi*np.sort(np.random.random(num_plots+1))
        theta_ranges[0] = 0
        theta_ranges[-1] = 2*np.pi
        self.data = np.zeros(num_plots, dtype=list)
        floor = np.zeros(num_plots)
        ceil = np.zeros(num_plots)
        for i in range(num_plots):
            n = round(num_pts*(theta_ranges[i+1]-theta_ranges[i])/(2*np.pi))+1
            theta = np.linspace(theta_ranges[i], theta_ranges[i+1], n)
            c = (abs(a**2 - b**2))**0.5
            phi_rad = -phi*np.pi/180
            xpos = a*np.cos(theta)+c+focus[0]
            ypos = b*np.sin(theta)+focus[1]
            data = np.zeros([n, 2])
            data[:, 0] = (xpos-focus[0])*np.cos(phi_rad)+(ypos-focus[1])*np.sin(phi_rad)+focus[0]
            data[:, 1] = -(xpos-focus[0])*np.sin(phi_rad)+(ypos-focus[1])*np.cos(phi_rad)+focus[1]
            ceil[i] = np.max(np.ceil(data[:, 0]))
            floor[i] = np.min(np.floor(data[:, 0]))
            self.data[i] = data[1:, :]
            ceil[i] = np.max(np.ceil(data[:, 0]))
            floor[i] = np.min(np.floor(data[:, 0]))
            
        self._x_lim = np.array([np.min(floor), np.max(ceil)])
        self._y_lim = self._x_lim
        return None
            
    def plot(self, fig_name="gen_plot.jpg", title=None, plot_style="default", fig_dpi=100, display_fig=True, save_fig=False):
        self.fig_name = fig_name
        if plot_style != "default":
            assert plot_style in plt.style.available, "Plot style must be valid matplotlib style sheet."
        markers = ['.', 'o', 'v', '^', '<', '>', '*', 'h', 's', 'X', 'D', 'd']
        marker_style = self._rng.choice(markers, size=self._num_plots, replace=False)
        if title == None:
            title = "Title of "+str(self._num_plots)+" Plot(s) of "+str(self._num_pts)+" Points"
            if self.seed != None:
                title += "\nSeed: "+str(self.seed)
        plt.figure(dpi=fig_dpi)
        plt.style.use(plot_style)
        plt.title(title)
        plt.xlabel("Label for x axis from "+str(self._x_lim[0])+" to "+str(self._x_lim[1]))
        plt.ylabel("Label for y axis from "+str(self._y_lim[0])+" to "+str(self._y_lim[1]))
        plt.xlim(self._x_lim)
        plt.ylim(self._y_lim)
        for i in range(self._num_plots):
            ms = marker_style[i]
            plt.plot(self.data[i][:, 0], self.data[i][:, 1], ms)
        if not display_fig:
            plt.close()
        if save_fig:
            plt.savefig(fig_name)
        return None
    
    def _y_func(self, x, w, alpha):
        y = np.zeros_like(x)
        for i in range(x.size):
            for j in range(w.size):
                y[i] += w[j]*x[i]**j
        rand_y_perturb = self._rng.random(self._num_pts)
        y += rand_y_perturb*alpha
        return y