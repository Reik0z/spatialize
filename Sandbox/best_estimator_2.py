import numpy as np
from matplotlib import pyplot as plt

import spatialize.gs.esi.aggfunction as af
import spatialize.gs.esi.precfunction as pf
from spatialize import logging
from spatialize.gs.esi import esi_griddata

#logging.log.setLevel("DEBUG")


def func(x, y):  # a kind of "cubic" function
    return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

rng = np.random.default_rng()
points = rng.random((1000, 2))
values = func(points[:, 0], points[:, 1])

grid_cmap, prec_cmap = 'coolwarm', 'bwr'

result = esi_griddata(points, values, (grid_x, grid_y),
                      local_interpolator="idw",
                      p_process="mondrian",
                      data_cond=False,
                      exponent=1.0,
                      n_partitions=500, alpha=0.985,
                      agg_function=af.mean
                      )

# operational error function for the observed dynamic range
op_error = pf.OperationalErrorPrecision(np.abs(np.nanmin(result.esi_samples())
                                               - np.nanmax(result.esi_samples())))

# aggregation function for esi_samples estimation
dirichlet_weighted_average = af.WeightedAverage(normalize=True)

switch = 0
bill = {}
min_range_sum = []
weights_arr = []

for i in range(100):
    result.re_estimate(dirichlet_weighted_average)
    if switch == 0:
        weights_arr = dirichlet_weighted_average.weights
        switch = 1
    else:
        weights_arr = np.vstack((weights_arr, dirichlet_weighted_average.weights))

    prec = result.precision_cube(op_error)
    bill[i] = af.Bilateral_Filter(prec)
    min_range_sum.append(np.sum(np.clip(bill[i], 0, np.max(bill[i])*0.5)))

min_range_sum = np.array(min_range_sum)
worst_prec = np.argmax(min_range_sum)
best_prec = np.argmin(min_range_sum)

dwa = af.WeightedAverage(normalize=True, weights=weights_arr[worst_prec])
s = result.esi_samples().shape
est_worst = np.flip(dwa(result.esi_samples().reshape(s[0]*s[1], s[2])).reshape(s[0], s[1]), 1)

dwa = af.WeightedAverage(normalize=True, weights=weights_arr[best_prec])
est_best = np.flip(dwa(result.esi_samples().reshape(s[0]*s[1], s[2])).reshape(s[0], s[1]), 1)

fig = plt.figure(dpi=150, figsize=(6, 5))
gs = fig.add_gridspec(2, 3, wspace=0.1, hspace=0.47)
(ax1, ax2) = gs.subplots()
ax1, ax2, ax3, ax4, ax5, ax6 = ax1[0], ax1[1], ax1[2], ax2[0], ax2[1], ax2[2]

# plot
ax1.imshow(est_worst.T, cmap=grid_cmap)
ax1.set_title("Est worst case")

# plot estimation
ax2.imshow(est_best.T, cmap=grid_cmap)
ax2.set_title("Est best case")

ax3.imshow(np.flip(result.re_estimate(af.mean), 1).T, cmap=grid_cmap)
ax3.set_title('Mean est')

# plot the default mse precision
ax4.imshow(bill[worst_prec].T, cmap=prec_cmap)
ax4.set_title('Bill worst')
ax4.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)

# plot a custom precision
ax5.imshow(bill[best_prec].T, cmap=prec_cmap)
ax5.set_title('Bill best')
ax5.plot(points[:, 0], points[:, 1], 'y.', ms=0.5)

ax6.imshow(np.flip(pf.mse_loss(est_best.reshape(s[0]*s[1]),
                               result.esi_samples().reshape(s[0]*s[1],
                               s[2])).reshape(s[0], s[1]), 1).T, cmap=prec_cmap)
ax6.set_title('MSE best')

plt.show()