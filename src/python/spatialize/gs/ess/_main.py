import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.exceptions import ConvergenceWarning

# just to turn warnings off
import warnings

from spatialize import logging
from spatialize.logging import log_message, default_singleton_callback
from spatialize.viz import plot_colormap_array

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ESSResult:
    def __init__(self, ess_scenarios, esi_result, desc):
        self.esi_result = esi_result
        self.scenarios = ess_scenarios
        self.desc = desc

    def __repr__(self):
        return str(self.desc)

    def quick_plot(self, n_imgs=9, n_cols=3, norm_lims=False, title_prefix="scenario", title=None):
        if title is None:
            title = self.desc
        return plot_colormap_array(self.scenarios, n_imgs=n_imgs,
                                   n_cols=n_cols, norm_lims=norm_lims,
                                   xi_locations=self.esi_result._xi,
                                   reference_map=self.esi_result.estimation(),
                                   title_prefix=title_prefix,
                                   title=title)


def ess_sample(esi_result,
               n_sims=100,
               point_model_name="kde",
               nan_model_name="ignore",
               nan_replace_func_name="median",
               kernel="tophat",
               n_components=1,
               desc=None,
               callback=default_singleton_callback):
    """
    To sample scenarios out of an interpolation result generative model.

    :param esi_result: The interpolation result. It has to be an ESIResult object.
    :param n_sims: Number of scenarios to be generated. Its has to be a positive integer (default is 100).
    :param point_model_name: The name of the model used to estimate the probability density at each point in space.
    It can be “kde” for using a kernel density estimate (the fastest),
    "emm" for fitting a Gaussian mixture model with the expectation maximisation algorithm or “vim” for the same
    model as the previous one, but fitted with variational inference (the slowest). The default is ‘kde’.
    :param nan_model_name: Set "ignore" to have the nan discarded from the dataset or "replace" to have it replaced.
    The default value is "ignore".
    :param nan_replace_func_name: Valid when nan_model_name="replace". Set "mean" to replace with the mean of the
    point data at each position or "median" to replace with the median.
    :param callback: Callback function for logging progress.
    :return: ESSResult containing the scenarios sampled out of an interpolation result.
    """

    # work always with the flattened array just to support arbitrary dimensions
    esi_samples = np.array(esi_result.esi_samples(raw=True))

    log_message(logging.logger.debug(f"esi_samples shape: {esi_samples.shape}"))

    if nan_replace_func_name == "median":
        nan_replace_func_def = np.median
    elif nan_replace_func_name == "mean":
        nan_replace_func_def = np.mean
    else:
        raise ValueError("nan_replace_func_name must be either 'mean' or 'median'")

    # just to memoize some common operations for the for-loop
    def replace_nan_with(arr, idx, with_function=nan_replace_func_def):
        nan_value = with_function(arr[idx, :])
        return np.nan_to_num(arr[idx, :], nan=nan_value)

    def ignore_nan(arr, idx):
        return arr[idx, :][~np.isnan(arr[idx, :])]

    if nan_model_name == "replace":
        nan_model = replace_nan_with
    elif nan_model_name == "ignore":
        nan_model = ignore_nan
    else:
        raise ValueError("nan_model_name must be either 'replace' or 'ignore'")

    # the bandwidth is calculated automatically using the silverman method
    if point_model_name == "kde":
        model = KernelDensity(kernel=kernel, bandwidth="silverman", atol=0.5, rtol=0.5)
    elif point_model_name == "vim":  # Variational inference dirichlet process gaussian mixture model
        model = BayesianGaussianMixture(n_components=n_components, covariance_type='full')
    elif point_model_name == "emm":  # Expectation-Maximisation gaussian mixture model
        model = GaussianMixture(n_components=2, covariance_type='full')
    else:
        raise ValueError("point_model_name must be either 'kde', 'vim', or 'emm'")

    scenarios = np.empty([esi_samples.shape[0], n_sims])

    callback(logging.progress.init(esi_samples.shape[0], 1))
    for esi_sample_idx in range(esi_samples.shape[0]):
        # dealing with nans
        point_data = nan_model(esi_samples, esi_sample_idx)

        # getting the model for this position
        model.fit(point_data.reshape(-1, 1))

        # sampling from the fitted model
        if point_model_name == "vim" or point_model_name == "emm":
            s = model.sample(n_sims)[0].reshape(1, n_sims)[0]

        if point_model_name == "kde":
            s = model.sample(n_sims).reshape(1, n_sims)[0]

        scenarios[esi_sample_idx, :] = s[:]

        callback(logging.progress.inform())

    callback(logging.progress.stop())

    if desc is None:
        if point_model_name == "kde":
            desc = f"{n_sims}sims_kde_{kernel}"
        else:
            desc = f"{n_sims}sims_{point_model_name}_{n_components}_components"

    return ESSResult(scenarios, esi_result, desc)
