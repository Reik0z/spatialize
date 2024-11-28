import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from spatialize import logging
from spatialize.gs.esi import esi_hparams_search, esi_nongriddata
import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.lossfunction as lf

logging.log.setLevel("DEBUG")

# the samples

# NOTE: modify this to make it download the CSV from
# the GitHub repo
samples = pd.read_csv('../../test/testdata/data.csv')
with open('../../test/testdata/grid.dat', 'r') as data:
    lines = data.readlines()
    lines = [l.strip().split() for l in lines[5:]]
    aux = np.float32(lines)
locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

w, h = 300, 200

points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['X', 'Y']].values

# operational error function for the observed dynamic range
op_error = lf.OperationalErrorLoss(np.abs(np.nanmin(values) - np.nanmax(values)))


# operational error function for the observed mean law
# op_error_precision=lf.OperationalErrorLoss(np.nanmean(values))

def esi_idw(p_process):
    search_result = esi_hparams_search(points, values, xi,
                                       local_interpolator="idw", griddata=False, k=10,
                                       p_process=p_process,
                                       exponent=list(np.arange(1.0, 15.0, 1.0)),
                                       alpha=(0.5, 0.6, 0.8, 0.9, 0.95, 0.98))

    search_result.plot_cv_error()
    plt.show()

    result = esi_nongriddata(points, values, xi,
                             local_interpolator="idw",
                             p_process=p_process,
                             n_partitions=500,
                             best_params_found=search_result.best_result())

    result.precision(op_error)
    result.quick_plot(w=w, h=h)
    plt.show()


def esi_kriging():
    search_result = esi_hparams_search(points, values, xi,
                                       local_interpolator="kriging", griddata=False, k=10,
                                       model=["spherical", "exponential", "cubic", "gaussian"],
                                       nugget=[0.5, 1.0],
                                       range=[100.0, 500.0, 1000.0],
                                       alpha=list(np.flip(np.arange(0.90, 0.95, 0.01))),
                                       sill=[0.9, 1.0, 1.1])

    search_result.plot_cv_error()
    plt.show()

    result = esi_nongriddata(points, values, xi,
                             local_interpolator="kriging",
                             n_partitions=500,
                             best_params_found=search_result.best_result())

    result.precision(op_error)
    result.quick_plot(w=w, h=h)
    plt.show()


if __name__ == '__main__':
    #esi_kriging()
    esi_idw("mondrian")
