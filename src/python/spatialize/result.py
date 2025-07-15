import numpy as np
from matplotlib import pyplot as plt

from spatialize import SpatializeError, in_notebook
from spatialize.viz import plot_colormap_data, PlotStyle

from typing import Optional, Dict, Any


class GridSearchResult:
    def __init__(self, search_result_data):
        self.search_result_data = search_result_data

        data = self.search_result_data
        self.cv_error = data[['cv_error']]
        min_error = self.cv_error.min()['cv_error']
        self.best_params = data[data.cv_error <= min_error]

    def plot_cv_error(self,
                      fig_args: Optional[Dict[str, Any]] = None,
                      subplot_args: Optional[Dict[str, Any]] = None,
                      theme: Optional[str] = 'alges',
                      color: Optional[str] = None):
        """
        It shows a graph of the cross-validation errors of the hyperparameter
        search process. The graph has two components: the first is the error histogram,
        and the second is the error level for each of the estimation scenarios generated
        by the gridded parameter search.

        :param fig_args: Dictionary with figure configuration for plt.subplots().
            Default assigns figsize=(10, 4) if not specified.
        :param subplot_args: Dictionary with subplot configuration for
            plt.subplots_adjust(). Default assigns wspace=0.45 if not specified.
        :param theme: Theme name. Available: 'whitegrid', 'darkgrid', 'white', 'dark',
            'alges', 'minimal', 'publication'
        :param color: Color for the plots. If None, uses theme default or 'skyblue'
        :return: Tuple with matplotlib figure and tuple of axes (fig, (ax1, ax2))
        :raises ValueError: If the specified theme does not exist.
        """
        # Default values for figsize and wspace if not specified
        if fig_args is None:
            fig_args = {'figsize': (10, 4)}
        elif 'figsize' not in fig_args:
            fig_args['figsize'] = (10, 4)

        if subplot_args is None:
            subplot_args = {'wspace': 0.45}
        elif 'wspace' not in subplot_args:
            subplot_args['wspace'] = 0.45

        with PlotStyle(theme=theme, color=color) as style:
            fig, (ax1, ax2) = plt.subplots(1, 2, **fig_args)
            plt.subplots_adjust(**subplot_args)
            fig.suptitle("Cross Validation Error")

            counts, bin_edges, _ = ax1.hist(self.cv_error,
                                            color=style.color,
                                            alpha=0.9,
                                            rwidth=0.95,
                                            zorder=3)
            ax1.set_xlabel("Error")
            ax1.set_ylabel("Frequency")
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax1.set_xticks(bin_centers[counts > 0])
            ax1.tick_params(axis='x', rotation=30)
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

            self.cv_error.plot(kind='line',
                               ax=ax2,
                               y='cv_error',
                               xlabel="Search result data index",
                               ylabel="Error",
                               color=style.color,
                               lw=2,
                               legend=False)
            
            return fig, (ax1, ax2)

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

        if not in_notebook():
            return fig

    def __repr__(self):
        min, max = np.nanmin(self.estimation()), np.nanmax(self.estimation())
        m, s, med = np.nanmean(self.estimation()), np.nanstd(self.estimation()), np.nanmedian(self.estimation())
        msg = (f"estimation results: \n"
               f"  minimum: {min:.3f}, maximum: {max:.3f}\n"
               f"  mean: {m:.2f}, std dev: {s:.2f}, median: {med:.2f}\n"
               f"to display the result, use the method ‘quick_plot()’.\n")
        return msg
