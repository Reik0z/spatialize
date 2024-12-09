import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gstools as gs

samples = pd.read_csv('../../test/testdata/data.csv')
with open('../../test/testdata/grid.dat', 'r') as data:
    lines = data.readlines()
    lines = [l.strip().split() for l in lines[5:]]
    aux = np.float32(lines)
locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

w, h = 300, 200

points = samples[['x', 'y']].values
x, y = samples[['x']].values, samples[['y']].values

values = samples[['cu']].values[:, 0]

xi = locations[['X', 'Y']].values
xi_x, xi_y = locations[['X']].values, locations[['Y']].values

# bins = np.arange(200)
# bin_center, gamma = gs.vario_estimate((x, y), values, bins)
# models = {
#     "Gaussian": gs.Gaussian,
#     "Exponential": gs.Exponential,
#     "Circular": gs.Circular,
#     "Spherical": gs.Spherical,
# }
# scores = {}
# fitted_models = {}
#
# # plot the estimated variogram
# plt.scatter(bin_center, gamma, color="k", label="data")
# ax = plt.gca()
#
# # fit all models to the estimated variogram
# for model in models:
#     fitted_models[model] = models[model](dim=2)
#     para, pcov, r2 = fitted_models[model].fit_variogram(bin_center, gamma, return_r2=True)
#     # fit_model.plot(x_max=40, ax=ax)
#     scores[model] = r2
#
# ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
#
# print("RANKING by Pseudo-r2 score")
# for i, (model, score) in enumerate(ranking, 1):
#     print(f"{i:>6}. {model:>15}: {score:.5}")

# krig = gs.krige.Ordinary(model=fitted_models[ranking[0][0]], cond_pos=points, cond_val=values)

angle = np.pi
model = gs.Exponential(dim=2, len_scale=[10, 5], angles=angle)
bins = range(0, 200, 2)
angle = np.pi / 8
bin_center, dir_vario, counts = gs.vario_estimate(
    *((x, y), values, bins),
    direction=gs.rotated_main_axes(dim=2, angles=angle),
    angles_tol=np.pi / 2,
    bandwidth=8,
    return_counts=True,
)

model.fit_variogram(bin_center, dir_vario)
krig = gs.krige.Ordinary(model=model, cond_pos=points, cond_val=values)

krig((xi_x, xi_y))
krig.plot(x_max=w, y_max=h)
