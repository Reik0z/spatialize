import numpy as np
import random as rd
from spatialize import in_notebook, logging

import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

from spatialize import SpatializeError
from spatialize.logging import log_message

PALETTES = {
    'alges': ['#070f15', '#0f212d', '#19384d', '#275474', '#326c94', '#4b87af', '#78abce', '#a7c9e1', '#cfe2ef'],
    'alges_muted': ['#142b3b','#1e4058', '#285676', '#326c94', '#5a89a9', '#84a6be', '#c1d2de'],
    'noname': ['#ffcbb4', '#ad92be', '#786aa9', '#124dc5'],
    'crest': ['#7dba91', '#59a590', '#40908e', '#287a8c', '#1c6488', '#254b7f']}

class PlotStyle:
    """
    Manage matplotlib plot styles with predefined themes.
    
    Parameters
    ----------
    theme : str, optional
        Theme name. Available: 'darkgrid', 'whitegrid', 'dark', 'white', 
        'alges', 'minimal', 'publication'
    color : str, optional
        Primary color for plots
    cmap : str or colormap, optional
        Colormap for plots
        
    Attributes
    ----------
    theme : str
        Active theme name
    color : str
        Primary plot color
    cmap : str or colormap
        Plot colormap
        
    Examples
    --------
    Basic usage with context manager::

        with PlotStyle(theme='dark', color='#ff6b6b') as style:
            plt.plot(x, y, color=style.color)
            plt.imshow(data, cmap=style.cmap)

    Simple usage without context manager::

        style = PlotStyle(theme='alges')
        plt.plot(x, y, color=style.color)
        style.reset_to_original()       # Manual reset
    """
    THEMES = {
        'darkgrid': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': True,
                'axes.facecolor': '#EAEAF2',
                'axes.edgecolor': 'white',
                'axes.labelweight': 'demibold',
                'figure.titleweight': 'bold',
                'grid.color': 'white',
                'xtick.bottom': False,
                'ytick.left': False,
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small'
            },
            'color': '#59a590',
            'cmap': mcolors.LinearSegmentedColormap.from_list(
                'crest', PALETTES['crest'], N=256),
        },
        'whitegrid': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': True,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#EAEAF2',
                'axes.labelweight': 'demibold',
                'figure.titleweight': 'bold',
                'grid.color': '#EAEAF2',
                'xtick.bottom': False,
                'ytick.left': False,
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small'
            },
            'color': '#7dba91',
            'cmap': mcolors.LinearSegmentedColormap.from_list(
                'crest', PALETTES['crest'], N=256),
        },
        'dark': {
            'rcparams': {
                'figure.facecolor': '#0d121a',
                'axes.facecolor': '#16202c',
                'axes.grid': False,
                'axes.edgecolor': '#1e2832',
                'axes.labelcolor': 'white',
                'axes.labelweight': 'demibold',
                'figure.titleweight': 'bold',
                'text.color': 'white',
                'xtick.color': '#b3b9c1',
                'ytick.color': '#b3b9c1',
                'patch.edgecolor': '#27ccab',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small', 
            },
            'color': '#24a187',
            'cmap': 'viridis'
        },
        'white': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': False,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#EAEAF2',
                'grid.color': '#EAEAF2',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'axes.labelweight': 'demibold',
                'figure.titleweight': 'bold',
                'xtick.bottom': True,
                'ytick.left': True,
            },
            'color': '#7dba91',
            'cmap': mcolors.LinearSegmentedColormap.from_list(
                'crest', PALETTES['crest'], N=256),
        },
        'alges': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': False,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#808080',
                'axes.labelcolor': '#424e77',
                'axes.labelweight': 'demibold',
                'figure.titleweight': 'bold',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'xtick.color': '#808080',
                'ytick.color': '#808080',
                'patch.edgecolor': '#496070',
                'text.color': "#424e77",
                'xtick.bottom': True,
                'ytick.left': True
            },
            'color': "#8fb4cd",
            'cmap': mcolors.LinearSegmentedColormap.from_list(
                'alges_cmap', PALETTES['alges'], N=256)
        },
        'alges_muted': {
            'rcparams': {
                'lines.solid_capstyle': 'round',
                'axes.grid': False,
                'axes.facecolor': 'white',
                'axes.edgecolor': '#808080',
                'axes.labelcolor': '#545f84',
                'axes.labelweight': 'demibold',
                'figure.titleweight': 'bold',
                'xtick.labelsize': 'small',
                'ytick.labelsize': 'small',
                'xtick.color': '#808080',
                'ytick.color': '#808080',
                'patch.edgecolor': '#496070',
                'text.color': "#424e77",
                'xtick.bottom': True,
                'ytick.left': True
            },
            'color': "#84a6be",
            'cmap': mcolors.LinearSegmentedColormap.from_list(
                'alges_cmap', PALETTES['alges_muted'], N=256)
        },
        'minimal': {
            'rcparams': {
                'axes.grid': False,
                'axes.spines.top': False,
                'axes.spines.right': False,
            },
            'color': '#333333',
            'cmap': 'copper'
        },
        'publication': {
            'rcparams': {
                'figure.titleweight': 'bold',
                'axes.grid': True,
                'grid.color': '#E0E0E0',
                'axes.spines.top': False,
                'axes.spines.right': False,
                'font.family': 'serif',
                'font.size': 10,
            },
            'color': '#000000',
            'cmap': 'cividis'
        }
    }

    DEFAULT_COLOR = 'skyblue'
    DEFAULT_CMAP = 'coolwarm'
    DEFAULT_PRECISION_CMAP = 'bwr'

    def __init__(self,
                 theme = None,
                 color = None,
                 cmap = None,
                 precision_cmap = None,):
        
        if theme and theme not in self.THEMES:
            raise ValueError(f"Theme '{theme}' not found. Available: {list(self.THEMES.keys())}")
        
        self._original_rcparams = plt.rcParams.copy()

        self.theme = theme
        self.color = self._set_color(color)
        self.cmap = self._set_cmap(cmap)
        self.precision_cmap = self._set_precision_cmap(precision_cmap)

        if self.theme:
            self._apply_theme()

    def _set_color(self, color):
        if color is not None:
            return color
        elif self.theme is not None:
            return self.THEMES[self.theme]['color']
        else:
            return self.DEFAULT_COLOR
    
    def _set_cmap(self, cmap):
        if cmap is not None:
            return cmap
        elif self.theme is not None:
            return self.THEMES[self.theme]['cmap']
        else:
            return self.DEFAULT_CMAP
        
    def _set_precision_cmap(self, cmap):
        if cmap is not None:
            return cmap
        #elif self.theme is not None:
        #    return self.THEMES[self.theme]['cmap']
        else:
            return self.DEFAULT_PRECISION_CMAP

    def _apply_theme(self):
        if self.theme and self.theme in self.THEMES:
            theme_config = self.THEMES[self.theme]['rcparams']
            plt.rcParams.update(theme_config)
    
    def get_available_themes(self):
        """Returns the list of available themes."""
        return list(self.THEMES.keys())
    
    def reset_to_original(self) -> None:
        """Restore the initial matplotlib configuration."""
        plt.rcParams.update(self._original_rcparams)
    
    def get_theme_info(self, theme_name: str):
        """Returns information about a specific theme."""
        if theme_name not in self.THEMES:
            raise ValueError(f"Theme '{theme_name}' not found.")
        return self.THEMES[theme_name].copy()
    
    def __repr__(self):
        """String representation of the object."""
        return f"PlotStyle(theme='{self.theme}', color='{self.color}', cmap='{self.cmap}')"

    def __enter__(self):
        """Context manager support - apply the theme."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support - restores original configuration."""
        self.reset_to_original()


def plot_histogram(data, ax, style, alpha = 0.9, rwidth=0.92, show_empty = False):
    """
    Plots a histogram of the given data using matplotlib
    :param data: The data to visualize. Should be either a 1D array-like object 
                 (list, numpy array, pandas Series) containing numerical values.
    :param ax: Matplotlib Axes object where the histogram will be plotted.
    :param style: Instance of PlotStyle object as Matplotlib context manager.
    :param alpha: Sets transparency of the histogram bars. Default assigns alpha=0.9 if not specified.
    :param rwidth: Sets relative width of the histogram bars. Default assigns rwidth=0.92 if not specified.
    :param show_empty: Whether to show tick labels for empty histogram bins. Default assigns False.
    """
    counts, bin_edges, _ = ax.hist(data,
                                   color=style.color,
                                   alpha=alpha,
                                   rwidth=rwidth,
                                   zorder=3)
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if show_empty:
        ax.set_xticks(bin_centers)
    else:
        ax.set_xticks(bin_centers[counts > 0])
    ax.tick_params(axis='x', rotation=30)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))


def plot_colormap_data(data, ax=None, w=None, h=None, xi_locations=None, griddata=False, title="", **figargs):
    """
    Plots a colormap (heatmap-like visualization) of the given data using matplotlib.

    :param data: The data to visualize. Should be either 2D grid data (if `griddata=True`)
                 or a flat array that can be reshaped into a 2D array.
    :param ax: Matplotlib axes object to plot into. If None, a new figure and axes will be created.
    :param w: Width of the image (number of columns). Required if `griddata=False` and `xi_locations` is not provided.
    :param h: Height of the image (number of rows). Required if `griddata=False` and `xi_locations` is not provided.
    :param xi_locations: Coordinates of the data points, used to infer shape (h, w) if not provided explicitly.
    :param griddata: If True, assumes `data` is already in 2D format and transposes it for display.
    :param title: Title to display on the plot (only used if `ax` is None and a new figure is created).
    :param figargs: Additional keyword arguments to pass to `imshow()`, such as `cmap`, `vmin`, or `vmax`.

    :raises SpatializeError: If neither `w`/`h` nor `xi_locations` are provided when `griddata` is False,
                             or if reshaping fails due to inconsistent dimensions.
    """
    if griddata:
        im = data.T
    else:
        if w is None or h is None:
            if xi_locations is None:
                raise SpatializeError(f"Wrong image size (w: {w}, h: {h})")
            else:
                h, w = len(np.unique(xi_locations[:, 0])) - 1, len(np.unique(xi_locations[:, 1])) - 1
                if len(data) != h * w:
                    h, w = len(np.unique(xi_locations[:, 0])), len(np.unique(xi_locations[:, 1]))
                log_message(logging.logger.debug(f"using h={h}, w={w}"))
        im = data.reshape(w, h)

    if ax is not None:
        plotter = ax
    else:
        fig = plt.figure(dpi=150)
        gs = fig.add_gridspec(1, 1)
        plotter = gs.subplots()
        plotter.set_title(title)

    img = plotter.imshow(im, origin='lower', **figargs)
    divider = make_axes_locatable(plotter)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    colorbar(img, orientation='vertical', cax=cax)


def plot_colormap_array(data, n_imgs=9, n_cols=3, norm_lims=False, xi_locations=None, reference_map=None,
                        cmap='coolwarm', title="", title_prefix="scenario", seed=None, **figargs):
    """
      Plots an array of colormap visualizations (heatmaps) for a subset of data columns in a grid layout.

      :param data: The data to visualize. A 2D array where each column represents a separate dataset to plot.
      :param n_imgs: The number of images (subplots) to display from the data. Defaults to 9.
      :param n_cols: The number of columns in the plot grid. Defaults to 3.
      :param norm_lims: If True, normalizes the colormap limits using a reference map. Defaults to False.
      :param xi_locations: Coordinates of the data points. Used for reshaping the data if necessary.
      :param reference_map: A map to use for normalizing the colormap range (`vmin`, `vmax`).
                             The min and max of this map are used to adjust the normalization.
      :param cmap: The colormap to use for plotting. Defaults to 'coolwarm'.
      :param title: The title for the entire plot. If `None`, no title will be set.
      :param title_prefix: Prefix for individual subplot titles (e.g., "scenario 1", "scenario 2").
      :param seed: The seed for random number generation. If `None`, a random seed will be generated for reproducibility.
      :param figargs: Additional arguments to pass to the `plot_colormap_data` function, such as `vmin` or `vmax`.

      :return: A Matplotlib figure object and the seed used for random sampling.
      :rtype: matplotlib.figure.Figure, int

      :raises ValueError: If the number of images to plot exceeds the number of columns in the data.
      """
    if n_imgs > data.shape[1]:
        n_imgs = data.shape[1]

    n_rows = n_imgs // n_cols if n_imgs % n_cols == 0 else (n_imgs // n_cols + 1)

    if seed is not None:
        rd.seed(seed)
    else:
        # if no seed is provided, we generate and capture a random one
        # to make the results reproducible
        rd.seed()
        seed = rd.randint(0, 10000)
        rd.seed(seed)

    # random sample of colormap images to plot
    sim_idx = rd.sample(range(data.shape[1]), k=n_imgs)

    if norm_lims and reference_map is not None:
        # if a reference map is provided, we use its min and max values
        # to make the images more comparable and we normalize the values
        # by sharpening the range of the data
        vmin = np.nanmin(reference_map) - 0.1
        vmax = np.nanmax(reference_map) - 0.1
    else:
        vmin, vmax = None, None

    # plot the simulation results
    fig = plt.figure(dpi=150, figsize=(10, 10))
    gs = fig.add_gridspec(n_rows, n_cols)
    axis = gs.subplots().flatten()

    for i in range(n_imgs):
        plot_colormap_data(data[:, sim_idx[i]], ax=axis[i],
                           xi_locations=xi_locations,
                           cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           **figargs)
        axis[i].set_title(f"{title_prefix} {sim_idx[i] + 1}")

    for i in range(n_imgs, n_rows * n_cols):
        axis[i].set_axis_off()
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.suptitle(title, fontsize=14)

    if in_notebook():
        return seed

    return fig, seed
