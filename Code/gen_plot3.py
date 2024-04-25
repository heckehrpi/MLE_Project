# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:23:56 2024

@author: heckeh
gen_plot v0.3 
added gen_ellipse function.
"""
import matplotlib.pyplot as plt
import numpy as np

class RNG_Plot:
    
    def __init__(self, seed=None):
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        return None
        
    def gen_pts(self, num_plots, pts_min=10, pts_max=50, alpha=5, mu=0, sigma=1, x_min=-5., x_max=5.):
        self._num_plots = num_plots
        self._num_pts = self._rng.integers(low=pts_min, high=pts_max)
        
        self.data = np.zeros(self._num_plots, dtype=list)
        self.w = np.zeros(self._num_plots, dtype=list)
        
        for i in range(self._num_plots):
            data = np.zeros([self._num_pts, 2])
            data[:, 0] = ((x_max - x_min)*self._rng.random(self._num_pts) + x_min)
            order = self._rng.integers(low=2, high=4)
            self.w[i] = self._rng.normal(mu, sigma, size=order)
            data[:, 1] = self._y_func(data[:, 0], self.w[i], alpha)
            self.data[i] = data
        return None
    
    def gen_ellipse(self, num_plots=2, axis_range=[1,10], focus=[0,0], pts_min=10, pts_max=100, phi=0):
        self._num_plots = num_plots
        self._num_pts = self._rng.integers(low=pts_min, high=pts_max)
        axes = np.sort(self._rng.random(2))
        self.focus = focus
        self.b = axes[0]
        self.a = axes[1]
        self.phi = phi
        
        self.data = np.zeros(num_plots, dtype=list)
        theta_ranges = 2*np.pi*np.sort(self._rng.random(num_plots+1))
        theta_ranges[0] = 0
        theta_ranges[-1] = 2*np.pi
        
        for i in range(num_plots):
            n = round(self._num_pts*(theta_ranges[i+1]-theta_ranges[i])/(2*np.pi))+1
            theta = np.linspace(theta_ranges[i], theta_ranges[i+1], n)
            data = self._ellipse_func(n, theta)
            self.data[i] = data[1:, :]
        self._plot_type = "ellipse"
        return None
            
    def plot(self, fig_name="gen_plot.jpg", title=None, plot_style="default", fig_dpi=100, save_fig=False):
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
        
        for i in range(self._num_plots):
            ms = marker_style[i]
            plt.plot(self.data[i][:, 0], self.data[i][:, 1], ms, markersize=5)
        
        self._x_lim = plt.xlim()
        self._y_lim = plt.ylim()
        
        plt.title(title)
        plt.xlabel("Label for x axis from "+str(round(self._x_lim[0], 3))+" to "+str(round(self._x_lim[1], 3)))
        plt.ylabel("Label for y axis from "+str(round(self._y_lim[0], 3))+" to "+str(round(self._y_lim[1], 3)))
        
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
    
    def _ellipse_func(self, n, theta):
        phi_rad = -self.phi*np.pi/180
        c = (abs(self.a**2 - self.b**2))**0.5
        xpos = self.a*np.cos(theta)+c+self.focus[0]
        ypos = self.b*np.sin(theta)+self.focus[1]
        data = np.zeros([n, 2])
        data[:, 0] = (xpos-self.focus[0])*np.cos(phi_rad)+(ypos-self.focus[1])*np.sin(phi_rad)+self.focus[0]
        data[:, 1] = -(xpos-self.focus[0])*np.sin(phi_rad)+(ypos-self.focus[1])*np.cos(phi_rad)+self.focus[1]
        return data