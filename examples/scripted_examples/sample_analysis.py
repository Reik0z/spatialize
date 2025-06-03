import numpy as np
import matplotlib.pyplot as plt

from spatialize import logging
from spatialize.data import load_drill_holes_andes_2D
from spatialize.gs.esi import esi_nongriddata, esi_hparams_search
from spatialize.gs.spa import cv_sample_pred_posterior

# for a more explanatory output of the spatialize functions
logging.log.setLevel("INFO")

model_dir_path = "./andes_2D"

# number of simulations to generate
n_sims = 100

# the samples included in the spatialize package
samples, locations, krig, _ = load_drill_holes_andes_2D()

# input variables for non gridded estimation spatialize functions
points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['x', 'y']].values

fixed_esi_n_partitions = 500
adaptive_esi_idw_n_partitions = 200


def esi_idw(p_process, values):
    search_result = esi_hparams_search(points, values, xi,
                                       local_interpolator="idw", griddata=False, k=10,
                                       p_process=p_process,
                                       exponent=list(np.arange(1.0, 15.0, 1.0)),
                                       alpha=(0.5, 0.6, 0.8, 0.9, 0.95, 0.98),
                                       seed=1500)

    result = esi_nongriddata(points, values, xi,
                             local_interpolator="idw",
                             p_process=p_process,
                             n_partitions=fixed_esi_n_partitions,
                             best_params_found=search_result.best_result())

    result.quick_plot(figsize=(10, 5))
    result.preview_esi_samples(n_imgs=9, n_cols=3)
    return result


def adaptive_esi_idw(values):
    search_result = esi_hparams_search(points, values, xi,
                                       local_interpolator="adaptiveidw", griddata=False, k=10,
                                       n_partitions=[5, 10],
                                       alpha=(0.5, 0.8, 0.9, 0.95),
                                       seed=1500)

    result = esi_nongriddata(points, values, xi,
                             local_interpolator="adaptiveidw",
                             n_partitions=adaptive_esi_idw_n_partitions,
                             best_params_found=search_result.best_result())

    result.quick_plot(figsize=(10, 5))
    result.preview_esi_samples(n_imgs=9, n_cols=3)
    return result


def cv_post_samples_adaptive_esi_idw():
    result = cv_sample_pred_posterior(points, values, xi,
                                      local_interpolator="adaptiveidw",
                                      n_partitions=adaptive_esi_idw_n_partitions,
                                      alpha=0.5)
    return result


def cv_post_samples_esi_idw(p_process):
    search_result = esi_hparams_search(points, values, xi,
                                       local_interpolator="idw", griddata=False, k=10,
                                       p_process=p_process,
                                       exponent=list(np.arange(1.0, 15.0, 1.0)),
                                       alpha=(0.5, 0.6, 0.8, 0.9, 0.95, 0.98))

    result = cv_sample_pred_posterior(points, values, xi,
                                      local_interpolator="idw",
                                      p_process=p_process,
                                      n_partitions=fixed_esi_n_partitions,
                                      best_params_found=search_result.best_result())

    return result


if __name__ == '__main__':
    sample_analyzer = cv_post_samples_esi_idw("mondrian")
    # sample_analyzer = cv_post_samples_adaptive_esi_idw()

    sample_analyzer.plot_summary(figsize=(14, 5))
    sample_analyzer.quick_plot_models(n_imgs=9, n_cols=3, figsize=(14, 10))

    samples_ranking = sample_analyzer.rank_samples()
    sample_analyzer.plot_ranking(samples_ranking)
    plt.show()
