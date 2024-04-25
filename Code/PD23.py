# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:41:59 2024

@author: henhe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:13:47 2024

@author: henhe
v0.22 actually added docstrings to v0.21.
"""
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from skimage.io import imread
import numpy as np
from time import time

class PlotDigitizer:
    def __init__(self, fig_name, xlim, ylim, num_plots, verbose=False):
        """
        Initialize the PlotDigitizer class.
        PlotDigitizer automatically digitizes scatter plot data using 
        unsupervized machine learning, given only the plot file and basic 
        information regarding the nature of the plot.

        Parameters
        ----------
        fig_name : str or path-like
            The filename of the image to be digitized. Only .jpg and .png 
            filetypes currently supported.
        xlim : 2 element array-like
            The x bounds of the plot to be digitized. This means the bounds of 
            the axes visible on the plot, not the bounds of the plotted data.
        ylim : 2 element array-like
            The y bounds of the plot to be digitized. This means the bounds of 
            the axes visible on the plot, not the bounds of the plotted data.
        num_plots : int
            The number of different sets of data plotted on the image.
        verbose : bool, optional
            Controls the terminal output of the method. The default is False, 
            giving no output.

        Returns
        -------
        None.

        """
        self._verbose = verbose
        self._num_plots = num_plots
        self._color_img = imread(fig_name)
        self._x_lim = xlim
        self._y_lim = ylim
        self._layer_attributes = {"background": [], "plot_background": [], "axes": [], "datasets_and_other": []}
        self._cropped_layer_attributes = {"background": [], "axes": [], "datasets_and_other": []}
        if fig_name.endswith(".jpg") or fig_name.endswith(".jpeg"):
            self._filetype = "jpg"
        elif fig_name.endswith(".png"):
            self._filetype = "png"
        else:
            assert self._filetype == "jpg" or self._filetype == "png", "Only .jpg and .png filetypes supported."
            
        return None
    
    def _quiet_init(self, denoise_padding):
        """
        Initializes and executes all of the methods to be run before fitting 
        data. No terminal output.

        Returns
        -------
        None.

        """
        self.uncropped_clusters = self._create_kmeans_clusters(self._color_img, self._num_plots+4)
        self.uncropped_isolated_clusters = self._create_isolated_clusters(self.uncropped_clusters, self._num_plots+4)
        self.uncropped_layer_attributes = self._classify_cluster_features(self.uncropped_isolated_clusters)
        x_bounds, y_bounds = self._detect_plot_bounds(self.uncropped_isolated_clusters, self.uncropped_layer_attributes)
        self.cropped_image= self._crop_image(self._color_img, y_bounds, x_bounds)
        self.cropped_clusters = self._create_kmeans_clusters(self.cropped_image, self._num_plots+1)
        self.cropped_isolated_clusters = self._create_isolated_clusters(self.cropped_clusters, self._num_plots+1)
        self.cropped_layer_attributes = self._classify_cluster_features(self.cropped_isolated_clusters)
        self.cropped_preprocessed_clusters = self._preprocess_clusters(self.cropped_isolated_clusters, denoise_padding)
        self.cropped_layer_attributes = self._classify_cluster_features(self.cropped_preprocessed_clusters)
        if len(self.cropped_layer_attributes["axes"]) != 0:
            self.cropped_clusters = self._create_kmeans_clusters(self.cropped_image, self._num_plots+2)
            self.cropped_isolated_clusters = self._create_isolated_clusters(self.cropped_clusters, self._num_plots+2)
            self.cropped_layer_attributes = self._classify_cluster_features(self.cropped_isolated_clusters)
            self.cropped_preprocessed_clusters = self._preprocess_clusters(self.cropped_isolated_clusters, denoise_padding)
            self.cropped_layer_attributes = self._classify_cluster_features(self.cropped_preprocessed_clusters)
        return None
    
    def _verbose_init(self, denoise_padding):
        """
        Initializes and executes all of the methods to be run before fitting 
        data. Outputs the methods called and times of execution.

        Returns
        -------
        None.

        """
        self.uncropped_clusters = self._timed_execution("created uncropped clusters in", self._create_kmeans_clusters, self._color_img, self._num_plots + 4)
        self.uncropped_isolated_clusters = self._timed_execution("isolated uncropped clusters in", self._create_isolated_clusters, self.uncropped_clusters, self._num_plots + 4)
        self.uncropped_layer_attributes = self._timed_execution("identified plot features in", self._classify_cluster_features, self.uncropped_isolated_clusters)
        x_bounds, y_bounds = self._timed_execution("detected plot bounds in", self._detect_plot_bounds, self.uncropped_isolated_clusters, self.uncropped_layer_attributes)
        self.cropped_image = self._timed_execution("cropped image in", self._crop_image, self._color_img, y_bounds, x_bounds)
        self.cropped_clusters = self._timed_execution("created cropped clusters in", self._create_kmeans_clusters, self.cropped_image, self._num_plots + 1)
        self.cropped_isolated_clusters = self._timed_execution("isolated cropped clusters in", self._create_isolated_clusters, self.cropped_clusters, self._num_plots + 1)
        self.cropped_layer_attributes = self._timed_execution("identified plot features in", self._classify_cluster_features, self.cropped_isolated_clusters)
        self.cropped_preprocessed_clusters = self._timed_execution("preprocessed cropped clusters in", self._preprocess_clusters, self.cropped_isolated_clusters, denoise_padding)
        self.cropped_layer_attributes = self._timed_execution("identified preprocessed plot features in", self._classify_cluster_features, self.cropped_preprocessed_clusters)
        
        if len(self.cropped_layer_attributes["axes"]) != 0:
            print("axes identified in cropped image. reclustering cropped image.")
            self.cropped_clusters = self._timed_execution("created cropped clusters in", self._create_kmeans_clusters, self.cropped_image, self._num_plots + 2)
            self.cropped_isolated_clusters = self._timed_execution("isolated cropped clusters in", self._create_isolated_clusters, self.cropped_clusters, self._num_plots + 2)
            self.cropped_layer_attributes = self._timed_execution("identified plot features in", self._classify_cluster_features, self.cropped_isolated_clusters)
            self.cropped_preprocessed_clusters = self._timed_execution("preprocessed cropped clusters in", self._preprocess_clusters, self.cropped_isolated_clusters, denoise_padding)
            self.cropped_layer_attributes = self._timed_execution("identified preprocessed plot features in", self._classify_cluster_features, self.cropped_preprocessed_clusters)
        return None
    
    def _create_kmeans_clusters(self, image, n_clusters):
        """
        Clusters the image into sets of pixels of similar color, using a KMeans 
        method.

        Parameters
        ----------
        image : array_like
            The numpy array of the image to be clustered, having been passed 
            through imread.
        n_clusters : int
            The number of clusters to break the image into.

        Returns
        -------
        Numpy array
            The clustered image.

        """
        kmeans = KMeans(n_clusters, n_init=3, init='k-means++', max_iter=100)
        reshape_params = {"jpg": (-1, 3), "png": (-1, 4)}
        km = image.reshape(reshape_params[self._filetype])
        kmeans.fit(km)
        clusters = kmeans.predict(km)
        return clusters.reshape(image.shape[0], image.shape[1])
    
    def _create_isolated_clusters(self, clusters, n_clusters):
        """
        Isolates the clustered image input by cluster,

        Parameters
        ----------
        clusters : array_like
            The clustered image array.
        n_clusters : int
            The number of clusters in the clustered image.

        Returns
        -------
        isolated_clusters : Numpy array
            The 3D isolated cluster array of size [clusters.shape, n_clusters].
            Each "layer" of the isolated array equal to 1 at each pixel in the 
            corresponding cluster, 0 otherwise.

        """
        isolated_clusters = np.zeros([clusters.shape[0], clusters.shape[1], n_clusters])
        for k in range(n_clusters):
            isolated_clusters[:, :, k] = (clusters == k).astype(int)
        return isolated_clusters
    
    def _classify_cluster_features(self, isolated_clusters, background_tol=0.25, axis_tol = 0.33,):
        """
        Automatically identifies the plot features present in each isolated cluster.

        Parameters
        ----------
        isolated_clusters : array_like
            The isolated cluster array having been passed through 
            _create_isolated_clusters.
        background_tol : float, optional
            The parameter used to identify the background of the plot. The 
            default is 0.25.
        axis_tol : float, optional
            The parameter used to identify the axes of the plot. The 
            default is 0.2.

        Returns
        -------
        layer_attributes : dict
            A dictionary containing all the plot attributes, and corresponding 
            clusters containing those attributes. Background, plot background, 
            axes, and datasets are classified.

        """
        avg = np.average(isolated_clusters, axis=(0,1))
        v_avg = np.average(isolated_clusters, axis=0)
        h_avg = np.average(isolated_clusters, axis=1)
        
        layer_attributes = {"background": [], "plot_background": [], "axes": [], "datasets_and_other": []}
        layer_attributes["background"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] >= background_tol and 1.0 in [np.max(v_avg[:, k]), np.max(h_avg[:, k])]]
        layer_attributes["plot_background"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] >= background_tol and 1.0 not in [np.max(v_avg[:, k]), np.max(h_avg[:, k])]]
        layer_attributes["axes"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] <= background_tol and (np.max(v_avg[:, k]) >= axis_tol or np.max(h_avg[:, k]) >= axis_tol)]
        layer_attributes["datasets_and_other"] = [k for k in range(isolated_clusters.shape[2]) if avg[k] <= background_tol and (np.max(v_avg[:, k]) <= axis_tol and np.max(h_avg[:, k]) <= axis_tol)]
        return layer_attributes
    
    def _detect_plot_bounds(self, isolated_clusters, layer_attributes, background_tol=0.25, plot_background_tol=0.45, axis_tol = 0.35, buffer=10):
        """
        Detects the bounds of the plot on the figure. The bounds of the plot 
        include the axes of the plot, or the bounds of the background color of 
        the plot.

        Parameters
        ----------
        isolated_clusters : array_like
            The isolated clusters of the plot image.
        layer_attributes : dict
            The dictionary containing all of the layer attributes of
            isolated_clusters.
        background_tol : float, optional
            The tolerance used to calculate the background. 
            Equal to the relative amount of the total frame taken up by the 
            background color. The default is 0.25.
        plot_background_tol : float, optional
            The tolerance used to calculate the presence of a plot background. 
            Equal to the relative amount of the total frame taken up by the 
            plot background color. The default is 0.45.
        axis_tol : foat, optional
            The tolerance used to calculate the presence of a plot axis. Equal 
            to the average amount of a horizontal or vertical slice of the plot 
            taken up by an axis color. The default is 0.35.
        buffer : int, optional
            Equal to the number of pixels the plot bounds will be cropped 
            inwards on each side to reduce noise on the outer pixels of the 
            cropped image. The default is 4.

        Returns
        -------
        x_bounds : numpy.ndarray
            Two element array containing the lower and upper horizontal bounds 
            of the plot on the image.
        y_bounds : numpy.ndarray
            Two element array containing the lower and upper horizontal bounds 
            of the plot on the image.

        """
        x_bounds = int(isolated_clusters.shape[1]/2)*np.ones(2)
        y_bounds = int(isolated_clusters.shape[0]/2)*np.ones(2)
        for k in range(isolated_clusters.shape[2]):
            v_avg = np.average(isolated_clusters[:, :, k], axis=0)
            h_avg = np.average(isolated_clusters[:, :, k], axis=1)
            x_range = np.array([])
            y_range = np.array([])
            if k in layer_attributes["plot_background"]:
                x_range = np.array([i for i in range(v_avg.size) if v_avg[i] >= plot_background_tol])
                y_range = np.array([i for i in range(h_avg.size) if h_avg[i] >= plot_background_tol])
            if k in layer_attributes["axes"]:
                x_range = np.array([j for j in range(buffer, v_avg.size-buffer) for i in range(h_avg.size) if h_avg[i] >= axis_tol and np.average(isolated_clusters[i, j-buffer:j+buffer, k]) > 0.75])
                y_range = np.array([i for i in range(buffer, h_avg.size-buffer) for j in range(v_avg.size) if v_avg[j] >= axis_tol and np.average(isolated_clusters[i-buffer:i+buffer, j, k]) > 0.75])
            if x_range.size >= 0.25*isolated_clusters.shape[1]:
                x_bounds = np.array([min(x_bounds[0], np.min(x_range)), max(x_bounds[1], np.max(x_range))]) 
            if y_range.size >= 0.25*isolated_clusters.shape[0]:
                y_bounds = np.array([min(y_bounds[0], np.min(y_range)), max(y_bounds[1], np.max(y_range))]) 
        return x_bounds, y_bounds
    
    def _crop_image(self, image, x_bounds, y_bounds, crop_buffer=-3):
        """
        Crops the input image to contain only the pixels inside the x and y 
        bounds. Used to crop the image to only contain plot data and exclude 
        axes and labels.

        Parameters
        ----------
        image : array-like
            The uncropped input image.
        x_bounds : array-like
            A two element strictly increasing array containing the x bounds of 
            the crop area.
        y_bounds : array-like
            A two element strictly increasing array containing the y bounds of 
            the crop area.
        crop_buffer : int, optional
            The buffer used to crop the image inward. Equal to the number of 
            pixels on each side to crop inward. The default is -3.

        Returns
        -------
        2D numpy ndarray
            The cropped image.

        """
        return image[x_bounds[0]+crop_buffer:x_bounds[1]-crop_buffer, y_bounds[0]+crop_buffer:y_bounds[1]-crop_buffer]
    
    def _preprocess_clusters(self, isolated_clusters, denoise_padding=0.01):
        """
        Preprocesses the isolated clusters conatining datasets. Preprocessing 
        consists of deleting extraneous data on the peripheries of each 
        cluster.

        Parameters
        ----------
        isolated_clusters : numpy array
            The 3D array containing the isolated clusters of the cropped image.
        denoise_padding : float, optional
            The amount of padding applied to the preprocessed images. Equal to 
            the percent in each dimension to delete data on every side. The 
            default is 0.01.

        Returns
        -------
        preprocessed_clusters : numpy array
            The 3D array containing the isolated and preprocessed clusters.

        """
        padding_px = round(isolated_clusters.shape[0]*denoise_padding)
        preprocessed_clusters = np.copy(isolated_clusters)
        for k in range(isolated_clusters.shape[2]):
            if k not in self.cropped_layer_attributes["background"] or self.cropped_layer_attributes["plot_background"]:
                preprocessed_clusters[:, :padding_px+1, k] = 0
                preprocessed_clusters[:, -padding_px+1:, k] = 0
                preprocessed_clusters[:padding_px+1, :, k] = 0
                preprocessed_clusters[-padding_px+1:, :, k] = 0
        return preprocessed_clusters
    
    def _get_cluster_inds(self, isolated_cluster):
        """
        Gets the indeces of the isolated cluster where the cluster is equal to 
        1.

        Parameters
        ----------
        isolated_cluster : numpy array
            The 2D array containing the isolated cluster to get the indeces of. 
            Should only contain 1 isolated cluster.

        Returns
        -------
        numpy array
            The array containing the indeces of isolated_cluster where the 
            array is equal to 1.

        """
        return np.argwhere(isolated_cluster[:, :] == 1.)
    
    def _fit_centroids(self, cluster, epochs=25, init_split=0.25, denoise_tol=0.5, radius_factor=0.01):
        """
        Fits the centroids of the detected data points.

        Parameters
        ----------
        cluster : numpy array
            The 2D array containing the isolated cluster to fit.
        epochs : int, optional
            The maximum number of iterations to run. Note this is rarely an 
            issue, the model converges in ~5-10 epochs. The default is 25.
        init_split : float, optional
            The amount of pixels in the isolated cluster to start with as 
            potential centroids. The default is 0.25.
        radius_factor : float, optional
            The radius factor used to determin the radius used in the 
            KNearestNeighbors models used to fit the centroids and cull 
            unnecessary pixels. A larger radius factor will eliminate fitting 
            more than one centroid to a given point, but may result in culling 
            too many centroids and missing plot points entirely. Smaller 
            factors will digitize more data points, but run the risk of fitting 
            multiple centroids to a single plot point, especially if the plot 
            points appear relatively large on the figure. Each plot will demand 
            a different factor to optimally digitize the figure, so the user is 
            encouraged to change this value if digitization does not yield 
            desired data. The default is 0.01.

        Returns
        -------
        centroids : numpy array
            The array of the fitted centroids.

        """
        culling_radius = radius_factor*cluster.shape[0]
        self._centroid_validate = NearestNeighbors(radius=culling_radius)
        self._centroid_fit = NearestNeighbors(radius=1.25*culling_radius)
        cluster_inds = self._get_cluster_inds(cluster)
        if cluster_inds.size == 0:
            return np.empty([0, 2])
        points_out, centroids = train_test_split(cluster_inds, test_size=init_split)
        num_iter = 0
        while num_iter <= epochs:
            centroids_temp = np.copy(centroids)
            centroids_temp = self._fit_to_centers(centroids_temp, points_out, denoise_tol)
            centroids_temp = self._cull_centroids(centroids_temp)
            if np.array_equal(centroids_temp, centroids):
                if self._verbose:
                    print("centroids unchanging at epoch", num_iter)
                break
            centroids = centroids_temp
            num_iter += 1
        return centroids
    
    def _cull_centroids(self, centroids):
        """
        Culls the fitted centroids by removing points that lie too close to 
        other points.

        Parameters
        ----------
        centroids : numpy array
            The 2D array containing the centroids.

        Returns
        -------
        centroids_culled : numpy array
            The 2D array containing the remaining valid centroids.

        """
        if centroids.size == 0:
            return centroids
        centroids_culled = np.copy(centroids)
        self._centroid_validate.fit(centroids_culled)
        _, inds = self._centroid_validate.radius_neighbors(centroids_culled)
        inds_delete = np.array([])
        for i in range(inds.shape[0]):
            if len(inds[i]) != 0:
                inds_delete = np.append(inds_delete, inds[i][1:])
        inds_delete = np.unique(inds_delete, axis=0)
        if len(inds_delete) != 0:
            centroids_culled = np.delete(centroids_culled, inds_delete.astype(int), axis=0)
        return centroids_culled
    
    def _fit_to_centers(self, centroids, points_out, denoise_tol=0.5):
        """
        Fits the given centroids to the center of the nearest points in the 
        plot, updating the centroids positions.
    
        Parameters
        ----------
        centroids : numpy array
            The 2D array containing the centroids.
            
        points_out : numpy array
            The 2D array containing the data points to fit the centroids 
            towards.
    
        Returns
        -------
        centroids_fit : numpy array
            The 2D array containing the fitted centroids after adjusting their positions.
    
        """
        if centroids.size == 0:
            return centroids
        self._centroid_fit.fit(points_out)
        _, inds = self._centroid_fit.radius_neighbors(centroids)
        centroids_fit = np.zeros_like(centroids)
        inds_delete = []
        inds_avg = sum([inds[i].size for i in range(inds.size)])/inds.size
        for i in range(inds.size):
            if inds[i].size <= denoise_tol*inds_avg:
                inds_delete.append(i)
                continue
            centroids_fit[i, 0] += np.average(points_out[inds[i], 0])
            centroids_fit[i, 1] += np.average(points_out[inds[i], 1])
        centroids_fit = np.round(centroids_fit)
        if len(inds_delete) != 0:
            centroids_fit = np.delete(centroids_fit, inds_delete, axis=0)
        
        return centroids_fit
    
    def _get_centroid_coords(self, centroids):
        """
        Converts centroid coordinates from pixel indeces to X-Y coordinates.
    
        Parameters
        ----------
        centroids : numpy array
            The 2D array containing the centroid coordinates in terms of their 
            pixel index.
    
        Returns
        -------
        data : numpy array
            The 2D array containing the centroid coordinates in X-Y 
            coordinates.
            
        """
        
        data = np.fliplr(centroids).astype(float)
        
        height = self.cropped_image.shape[0]
        width = self.cropped_image.shape[1]
        
        data[:, 0] = (self._x_lim[1] - self._x_lim[0])*(data[:, 0]/width) + self._x_lim[0]
        data[:, 1] = (self._y_lim[1] - self._y_lim[0])*((height - data[:, 1])/height) + self._y_lim[0]
            
        return data
    
    def _timed_execution(self, string, func, *args, **kwargs):
        """
    Executes a function and prints the elapsed time.

    Parameters
    ----------
    string : str
        A string to print along with the timing information.
    func : function
        The function to execute.
    *args : tuple
        Positional arguments to pass to the function.
    **kwargs : dict
        Keyword arguments to pass to the function.

    Returns
    -------
    result : object
        The result returned by the function.

    """
        t0 = time()
        result = func(*args, **kwargs)
        t1 = time()
        print(f"{string} {round(t1 - t0, 3)} seconds")
        return result
       
    def _quiet_fit_data(self, epochs, init_split, denoise_tol, radius_factor):
        """
        Executes the fit_centroids function for each detected set of data. 
        No terminal output.

        Parameters
        ----------
        epochs : int, optional
            The maximum number of iterations to run. Note this is rarely an 
            issue, the model converges in ~5-10 epochs. The default is 25.
        init_split : float, optional
            The amount of pixels in the isolated cluster to start with as 
            potential centroids. The default is 0.25.
        radius_factor : float, optional
            The radius factor used to determin the radius used in the 
            KNearestNeighbors models used to fit the centroids and cull 
            unnecessary pixels. A larger radius factor will eliminate fitting 
            more than one centroid to a given point, but may result in culling 
            too many centroids and missing plot points entirely. Smaller 
            factors will digitize more data points, but run the risk of fitting 
            multiple centroids to a single plot point, especially if the plot 
            points appear relatively large on the figure. Each plot will demand 
            a different factor to optimally digitize the figure, so the user is 
            encouraged to change this value if digitization does not yield 
            desired data. The default is 0.01.

        Returns
        -------
        data : numpy array
            A numpy array containing each fit set of data.

        """
        n_plots = len(self.cropped_layer_attributes["datasets_and_other"])
        self.data = np.zeros(n_plots, dtype=list)
        for plot in range(n_plots):
            cluster = self.cropped_preprocessed_clusters[:, :, self.cropped_layer_attributes["datasets_and_other"][plot]]
            centroids = self._fit_centroids(cluster, epochs=epochs, init_split=init_split, denoise_tol=denoise_tol, radius_factor=radius_factor)
            self.data[plot] = self._get_centroid_coords(centroids)
        return self.data
    
    def _verbose_fit_data(self, epochs, init_split, denoise_tol, radius_factor):
        """
        Executes the fit_centroids function for each detected set of data. 
        Verbose terminal output.

        Parameters
        ----------
        epochs : int, optional
            The maximum number of iterations to run. Note this is rarely an 
            issue, the model converges in ~5-10 epochs. The default is 25.
        init_split : float, optional
            The amount of pixels in the isolated cluster to start with as 
            potential centroids. The default is 0.25.
        radius_factor : float, optional
            The radius factor used to determin the radius used in the 
            KNearestNeighbors models used to fit the centroids and cull 
            unnecessary pixels. A larger radius factor will eliminate fitting 
            more than one centroid to a given point, but may result in culling 
            too many centroids and missing plot points entirely. Smaller 
            factors will digitize more data points, but run the risk of fitting 
            multiple centroids to a single plot point, especially if the plot 
            points appear relatively large on the figure. Each plot will demand 
            a different factor to optimally digitize the figure, so the user is 
            encouraged to change this value if digitization does not yield 
            desired data. The default is 0.01.

        Returns
        -------
        data : numpy array
            A numpy array containing each fit set of data.

        """
        n_plots = len(self.cropped_layer_attributes["datasets_and_other"])
        self.data = np.zeros(n_plots, dtype=list)
        for plot in range(n_plots):
            cluster = self.cropped_preprocessed_clusters[:, :, self.cropped_layer_attributes["datasets_and_other"][plot]]
            centroids = self._timed_execution("fit centroids in", self._fit_centroids, cluster, epochs=epochs, denoise_tol=denoise_tol, init_split=init_split, radius_factor=radius_factor)
            self.data[plot] = self._timed_execution("calculated plot coordinates in", self._get_centroid_coords, centroids)
        return self.data
    
    def fit_data(self, denoise_padding=0.01, epochs=25, init_split=0.25, denoise_tol=0.5, radius_factor=0.01):
        """
        The function intended to be directly used by the user.
        Fits the detected data of the input image.

        Parameters
        ----------
    *args : tuple
        Positional arguments to pass specifically to the _fit_centroids 
        function.
    **kwargs : dict
        Keyword arguments to pass specifically to the _fit_centroids function.

        Returns
        -------
        numpy array
            The fit data output by the verbose of quiet fit_data functions.

        """
        if self._verbose:
            self._verbose_init(denoise_padding)
            return self._verbose_fit_data(epochs, init_split, denoise_tol, radius_factor)
        else:
            self._quiet_init(denoise_padding)
            return self._quiet_fit_data(epochs, init_split, denoise_tol, radius_factor)

    