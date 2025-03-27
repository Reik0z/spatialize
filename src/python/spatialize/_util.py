import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.progress import track

from spatialize import SpatializeError


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def signature_overload(pivot_arg, common_args, specific_args):
    def outer_function(func):
        def inner_function(*args, **kwargs):
            pivot_key, pivot_default_value, pivot_desc = pivot_arg

            if pivot_key not in common_args:
                common_args[pivot_key] = pivot_default_value

            # if a common argument is needed for the pivot argument
            # and is not in kwargs then add it with its declared
            # default value
            for arg in common_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = common_args[arg]

            # get the specific args for the current pivot key
            pk = kwargs[pivot_key]

            if pk not in specific_args:
                raise SpatializeError(f"{pivot_desc.capitalize()} '{pk}' not supported")

            spec_args = specific_args[pk]

            # if the specific argument is needed for the pivot argument
            # and is not in kwargs then add it with its declared
            # default value
            for arg in spec_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = spec_args[arg]

            # check that all arguments are consistent
            # with the pivot key
            for arg in kwargs.keys():
                if arg != pivot_key and arg not in spec_args and arg not in common_args:
                    raise SpatializeError(f"Argument '{arg}' not recognized for '{pk}' {pivot_desc.lower()}")

            return func(*args, **kwargs)

        return inner_function

    return outer_function


def get_progress_bar(list_like_obj, desc):
    if in_notebook():
        it = tqdm(range(len(list_like_obj)), desc=desc)
    else:
        it = track(range(len(list_like_obj)), description=desc)
    return it


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


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
        if self.griddata:
            im = data.T
        else:
            if w is None or h is None:
                if self._xi is None:
                    raise SpatializeError(f"Wrong image size (w: {w}, h: {h})")
                else:
                    h, w = len(np.unique(self._xi[:, 0]))-1, len(np.unique(self._xi[:, 1]))-1
                    print(f"Image size not provided. Using shapes: {w} x {h}")
            im = data.reshape(w, h)

        plotter = plt
        if ax is not None:
            plotter = ax

        img = plotter.imshow(im, origin='lower', **figargs)
        divider = make_axes_locatable(plotter)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        colorbar(img, orientation='vertical', cax=cax)

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
        if self._xi.shape[1] > 2:
            raise SpatializeError("quick_plot() for 3D data is not supported")

        fig = plt.figure(dpi=150, **figargs)
        gs = fig.add_gridspec(1, 1, wspace=0.45)
        ax = gs.subplots()

        ax.set_title('Estimation')
        self.plot_estimation(ax, w=w, h=h)
        ax.set_aspect('equal')

        return fig

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def __repr__(self):
        min, max = np.nanmin(self.estimation()), np.nanmax(self.estimation())
        m, s, med = np.nanmean(self.estimation()), np.nanstd(self.estimation()), np.nanmedian(self.estimation())
        msg = (f"estimation results: \n"
               f"  minimum: {min:.3f}, maximum: {max:.3f}\n"
               f"  mean: {m:.2f}, std dev: {s:.2f}, median: {med:.2f}\n"
               f"to display the result, use the method ‘quick_plot()’.\n")
        return msg
