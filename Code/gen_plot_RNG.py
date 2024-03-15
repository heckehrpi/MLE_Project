"""
Created on Wed Feb 28 14:30:47 2024

@author: heckeh
"""
import matplotlib.pyplot as plt
import numpy as np

class RNG_plot:
    
    def __init__(self, seed=None, num_plots=1, alpha=2, mu=0, sigma=0.75, x_min=-3, x_max=5):
        self.seed = seed
        self._num_plots = num_plots
        self.a = alpha
        self.m = mu
        self.s = sigma
        self.rng = np.random.default_rng(self.seed)
        self.x_min = x_min
        self.x_max = x_max
        return None

    def plot(self, fig_name="gen_plot.png", fig_dpi=100, save_fig=False):
        self.plot_name = fig_name
        self.num_pts = self.rng.integers(low=20, high=60)
        self.x_range = np.sort((self.x_max - self.x_min)*self.rng.random(2) + self.x_min)
        self.x = np.linspace(self.x_range[0], self.x_range[1], self.num_pts)
        self.y = np.zeros([self.num_pts, self._num_plots])
        markers = ['.', 'o', 'v', '^', '<', '>', '*', 'h', 's', '+', 'x', 'X', 'D', 'd']
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        marker_style = self.rng.choice(len(markers), size=self._num_plots, replace=False)
        marker_color = self.rng.choice(len(colors), size=self._num_plots, replace=False)
        
        plt.figure(dpi=fig_dpi)
        for j in range(self._num_plots):
            self.w = self.rng.normal(self.m, self.s, size=4)
            y_func = lambda x: self.w[0] + self.w[1]*x + 0.5*self.w[2]*x**2 + 0.25*self.w[3]*x**3
            rand_y_perturb = self.rng.random(self.num_pts)
            for i in range(self.num_pts):
                self.y[i, j] = y_func(self.x[i]) + rand_y_perturb[i]*self.a
            self.y_range = np.array([np.min(self.y), np.max(self.y)])
            self.marker_size = self.rng.integers(3, 10)
            plt.plot(self.x, self.y[:, j], colors[marker_color[j]]+markers[marker_style[j]], ms = self.marker_size)
        
        plt.title("Plot of "+str(self.num_pts)+" Points")
        plt.xlabel("X Range: "+str(round(self.x_range[0], 3))+ " to "+str(round(self.x_range[1], 3)))
        self.x_range = np.array([np.floor(self.x_range[0]), np.ceil(self.x_range[1])])
        self.y_range = np.array([np.floor(self.y_range[0]), np.ceil(self.y_range[1])])
        plt.xlim(self.x_range)
        plt.ylim(self.y_range)
        if save_fig:
            plt.savefig(fig_name)
        