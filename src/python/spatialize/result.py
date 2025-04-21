import numpy as np
from matplotlib import pyplot as plt

from spatialize import SpatializeError
from spatialize.viz import plot_colormap_data


class GridSearchResult:
    def __init__(self, search_result_data):
        self.search_result_data = search_result_data

        data = self.search_result_data
        self.cv_error = data[['cv_error']]
        min_error = self.cv_error.min()['cv_error']
        self.best_params = data[data.cv_error <= min_error]

    def plot_cv_error(self, **kwargs):
        """
        It shows a graph of the cross-validation errors of the hyperparameter
        search process. The graph has two components: the first is the error histogram,
        and the second is the error level for each of the estimation scenarios generated
        by the gridded parameter search.

        :param kwargs: Additional keyword arguments.
        """
        fig = plt.figure(figsize=(10, 4), dpi=150)
        gs = fig.add_gridspec(1, 2, wspace=0.45)
        (ax1, ax2) = gs.subplots()
        fig.suptitle("Cross Validation Error")
        self.cv_error.plot(kind='hist', ax=ax1,
                           title="Histogram",
                           rot=25,
                           color='skyblue',
                           legend=False)
        self.cv_error.plot(kind='line', ax=ax2,
                           y='cv_error',
                           xlabel="Search result data index",
                           ylabel="Error",
                           color='skyblue',
                           legend=False)

    def best_result(self, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class EstimationResult:
    def __init__(self, estimation, griddata=False, original_shape=None, xi=None):
        self._estimation = estimation
        self.griddata = griddata
        self.original_shape = original_shape
        self._xi = xi

    def estimation(self):
        """
        Returns the estimated values at locations `xi` by aggregating all ESI samples
        using the aggregation function provided in the `agg_function` argument (in both
        function :func:`esi_griddata` and :func:`esi_nongriddata`). This estimate can be changed
        using another aggregation function with the :func:`re_estimate` method of this same class.

        Returns
        =======
        estimation : numpy.ndarray
            An array of dimension $N_{x^*}$, for non-gridded data, and of dimension $d_1 \times d_2$
            for gridded data -- remember that, in this case, $d_1 \times d_2 = N_{x^*}$
        """
        if self.griddata:
            return self._estimation.reshape(self.original_shape)
        else:
            return self._estimation

    def plot_estimation(self, ax=None, w=None, h=None, **figargs):
        """
        Plots the estimation using `matplotlib`.

        Parameters
        ----------
        ax :  (`matplotlib.axes.Axes`, optional)
            The `Axes` object to render the plot on. If `None`, a new `Axes` object is created.
        w : (int, optional)
            The width of the image (if the data is reshaped).
        h : (int, optional)
            The height of the image (if the data is reshaped).
        **figargs : (optional)
            Additional keyword arguments passed to the figure creation (e.g., DPI, figure size).

        """
        if 'cmap' not in figargs:
            figargs['cmap'] = 'coolwarm'
        self._plot_data(self.estimation(), ax, w, h, **figargs)

    def _plot_data(self, data, ax=None, w=None, h=None, **figargs):
        plot_colormap_data(data, ax=ax, w=w, h=h, xi_locations=self._xi, griddata=self.griddata, **figargs)

    def quick_plot(self, w=None, h=None, **figargs):
        """
        Quickly plots the estimation using `matplotlib`.

        Parameters
        ----------
        w : (int, optional)
            The width of the image (if the data is reshaped).
        h : (int, optional)
            The height of the image (if the data is reshaped).
        **figargs : (optional)
            Additional keyword arguments passed to the figure creation (e.g., DPI, figure size).
        """
        if not self._xi is None:
            if self._xi.shape[1] > 2:
                raise SpatializeError("quick_plot() for 3D data is not supported")

        fig = plt.figure(dpi=150, **figargs)
        gs = fig.add_gridspec(1, 1, wspace=0.45)
        ax = gs.subplots()

        ax.set_title('Estimation')
        self.plot_estimation(ax, w=w, h=h)
        ax.set_aspect('equal')

        return fig

    def __repr__(self):
        min, max = np.nanmin(self.estimation()), np.nanmax(self.estimation())
        m, s, med = np.nanmean(self.estimation()), np.nanstd(self.estimation()), np.nanmedian(self.estimation())
        msg = (f"estimation results: \n"
               f"  minimum: {min:.3f}, maximum: {max:.3f}\n"
               f"  mean: {m:.2f}, std dev: {s:.2f}, median: {med:.2f}\n"
               f"to display the result, use the method ‘quick_plot()’.\n")
        return msg
