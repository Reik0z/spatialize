import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gstools as gs
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable

# get the data
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

# get the empirical variogram
bins = np.arange(200)
bin_center, gamma = gs.vario_estimate((x, y), values, bins)

# models to test
models = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Circular": gs.Circular,
    "Spherical": gs.Spherical,
    # "Matern": gs.Matern,
    # "Stable": gs.Stable,
    # "Rational": gs.Rational,
    "SuperSpherical": gs.SuperSpherical,
    "JBessel": gs.JBessel,
}
scores = {}
fitted_models = {}

# fit all models to the estimated variogram
for model in models:
    fitted_models[model] = models[model](dim=2)
    para, pcov, r2 = fitted_models[model].fit_variogram(bin_center, gamma, return_r2=True)
    scores[model] = r2

# build the ranking according to r2
ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
print("RANKING by Pseudo-r2 score")
for i, (model, score) in enumerate(ranking, 1):
    print(f"{i:>6}. {model:>15}: {score:.5}")

# run the ordinary kriging
krig = gs.krige.Ordinary(model=fitted_models[ranking[0][0]], cond_pos=points, cond_val=values)
estimate = krig((xi_x, xi_y))

# plot results
ax = plt.gca()
im = estimate[0].reshape(w, h)
#im = np.flipud(im)
img = ax.imshow(im, origin='lower', cmap='coolwarm')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img, orientation='vertical', cax=cax)
plt.show()
