import numpy as np
import random as rd
from spatialize import in_notebook, logging

import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatialize import SpatializeError
from spatialize.logging import log_message


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
