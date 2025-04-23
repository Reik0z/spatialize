import numpy as np
import random as rd
from spatialize import in_notebook, logging

import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatialize import SpatializeError
from spatialize.logging import log_message


def plot_colormap_data(data, ax=None, w=None, h=None, xi_locations=None, griddata=False, title="", **figargs):
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
