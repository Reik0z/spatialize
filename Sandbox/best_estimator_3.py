import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.lossfunction as lf
from spatialize import logging
from spatialize.gs.esi import esi_hparams_search, esi_nongriddata


logging.log.setLevel("DEBUG")

samples = pd.read_csv('/Users/alejandro/Software/spatialize/test/testdata/data.csv')
with open('/Users/alejandro/Software/spatialize/test/testdata/grid.dat', 'r') as data:
    lines = data.readlines()
    lines = [l.strip().split() for l in lines[5:]]
    aux = np.float32(lines)
locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

w, h = 300, 200

grid_cmap, prec_cmap = 'coolwarm', 'bwr'

points = samples[['x', 'y']].values
values = samples[['cu']].values[:, 0]
xi = locations[['X', 'Y']].values

search_result = esi_hparams_search(points, values, xi,
                                   local_interpolator="idw", griddata=False, k=10,
                                   p_process="mondrian",
                                   exponent=list(np.arange(1.0, 15.0, 1.0)),
                                   alpha=(0.5, 0.6, 0.8, 0.9, 0.95, 0.98))
#search_result.plot_cv_error()
#plt.show()

result = esi_nongriddata(points, values, xi,
                         local_interpolator="idw",
                         p_process="mondrian",
                         n_partitions=500,
                         best_params_found=search_result.best_result())


# operational error function for the observed dynamic range
op_error_cube = lf.OperationalErrorLoss(np.abs(np.nanmin(result.esi_samples())
                                               - np.nanmax(result.esi_samples())), use_cube=True)

op_error = lf.OperationalErrorLoss(np.abs(np.nanmin(result.esi_samples())
                                          - np.nanmax(result.esi_samples())))

# aggregation function for esi_samples estimation
dirichlet_weighted_average = af.WeightedAverage(normalize=True)

switch = 0
bill = {}
max_range_sum = []
weights_arr = []

for i in range(100):
    result.re_estimate(dirichlet_weighted_average)
    if switch == 0:
        weights_arr = dirichlet_weighted_average.weights
        switch = 1
    else:
        weights_arr = np.vstack((weights_arr, dirichlet_weighted_average.weights))

    prec = result.precision_cube(op_error_cube)
    prec_rs = prec.reshape(w, h, prec.shape[1])
    bill[i] = af.bilateral_filter(prec_rs)
    max_range_sum.append(np.sum(bill[i]))
        #np.clip(bill[i], np.max(bill[i]) * 0.50, np.max(bill[i]))))

max_range_sum = np.array(max_range_sum)
worst_prec = np.argmax(max_range_sum)
best_prec = np.argmin(max_range_sum)

dwa = af.WeightedAverage(normalize=True, weights=weights_arr[worst_prec])

est_worst = dwa(result.esi_samples()).reshape(w, h)

dwa = af.WeightedAverage(normalize=True, weights=weights_arr[best_prec])
est_best = dwa(result.esi_samples()).reshape(w, h)

fig = plt.figure(dpi=150, figsize=(10, 5))
gs = fig.add_gridspec(2, 4, wspace=0.1, hspace=0.47)
(ax1, ax2) = gs.subplots()
ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = ax1[0], ax1[1], ax1[2], ax1[3], ax2[0], ax2[1], ax2[2], ax2[3]

# plot
img1 = ax1.imshow(est_worst, cmap=grid_cmap, origin='lower')
ax1.set_title("Est worst case")
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img1, orientation='vertical', cax=cax1)

# plot estimation
img2 = ax2.imshow(est_best, cmap=grid_cmap, origin='lower')
ax2.set_title("Est best case")
divider = make_axes_locatable(ax2)
cax2 = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img2, orientation='vertical', cax=cax2)

mean_r = result.re_estimate(af.mean).reshape(w,h)

img3 = ax3.imshow(mean_r, cmap=grid_cmap, origin='lower')
ax3.set_title('Mean est')
divider = make_axes_locatable(ax3)
cax3 = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img3, orientation='vertical', cax=cax3)

arr4 = op_error(est_worst.reshape(w*h), result.esi_samples()).reshape(w, h)

img4 = ax4.imshow(arr4, cmap=prec_cmap, origin='lower')
ax4.set_title('Op Error worst')
divider = make_axes_locatable(ax4)
cax4 = divider.append_axes("right", size="5%", pad=0.1)
plt.contourf(arr4, levels=np.linspace(0, 0.015, 20))
colorbar(img4, orientation='vertical', cax=cax4)

# plot the default mse precision
img5 = ax5.imshow(np.flip(bill[worst_prec], axis=1), cmap=prec_cmap, origin='lower')
ax5.set_title('Bill worst')
#ax5.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)
divider = make_axes_locatable(ax5)
cax5 = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img5, orientation='vertical', cax=cax5)

# plot a custom precision
img6 = ax6.imshow(np.flip(bill[best_prec], axis=1), cmap=prec_cmap, origin='lower')
ax6.set_title('Bill best')
#ax6.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)
divider = make_axes_locatable(ax6)
cax6 = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img6, orientation='vertical', cax=cax6)

arr7 = op_error(est_best.reshape(w*h), result.esi_samples()).reshape(w, h)

img7 = ax7.imshow(arr7, cmap=prec_cmap, origin='lower')
ax7.set_title('Op Error best')
divider = make_axes_locatable(ax7)
cax7 = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img7, orientation='vertical', cax=cax7)

img8 = ax8.imshow(result.precision(op_error).reshape(w, h), cmap=prec_cmap, origin='lower')
ax8.set_title('Op Error Mean est')
divider = make_axes_locatable(ax8)
cax8 = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img8, orientation='vertical', cax=cax8)

plt.show()
