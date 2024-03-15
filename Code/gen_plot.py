"""
Created on Wed Feb 28 14:30:47 2024

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
        
    def gen_pts(self, num_plots, alpha=5, mu=0, sigma=1, x_min=-5., x_max=5., buffer=0.9):
        self._num_plots = num_plots
        self._x_lim = np.array([x_min, x_max])
        
        self._num_pts = self._rng.integers(low=10, high=80)
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
            
    def plot(self, fig_name="gen_plot2.jpg", fig_dpi=100, save_fig=False):
        self.fig_name = fig_name
        
        markers = ['.', 'o', 'v', '^', '<', '>', '*', 'h', 's', 'X', 'D', 'd']
        # colors = ['b', 'g', 'r', 'c', 'm', 'y']
        colors = ['b', 'r', 'm']
        
        marker_style = self._rng.choice(len(markers), size=self._num_plots, replace=False)
        marker_color = self._rng.choice(len(colors), size=self._num_plots, replace=False)
        plot_style = self._rng.choice(len(available), replace=False)
                
        title = "Title of "+str(self._num_plots)+" Plot(s) of "+str(self._num_pts)+" Points"
        if self.seed != None:
            title += "\nSeed: "+str(self.seed)
        
        plt.figure(dpi=fig_dpi)
        plt.style.use(available[plot_style])
        plt.title(title)
        plt.xlabel("Label for x axis from "+str(self._x_lim[0])+" to "+str(self._x_lim[1]))
        plt.ylabel("Label for y axis from "+str(self._y_lim[0])+" to "+str(self._y_lim[1]))
        plt.xlim(self._x_lim)
        plt.ylim(self._y_lim)
        
        for i in range(self._num_plots):
            ms = markers[marker_style[i]]
            plt.plot(self.data[i][:, 0], self.data[i][:, 1], ms)
        
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