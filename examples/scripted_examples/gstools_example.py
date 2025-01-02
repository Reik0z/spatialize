import numpy as np
from matplotlib import pyplot as plt
import gstools as gs
from matplotlib.pyplot import colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spatialize.data import load_drill_holes_andes_2D


# get the data
samples, locations, krig, _ = load_drill_holes_andes_2D()
locations = locations.sort_values(["z", "y", "x"])

w, h = 300, 200

points = samples[['x', 'y']].values
x, y = samples[['x']].values, samples[['y']].values

values = samples[['cu']].values[:, 0]

xi = locations[['x', 'y']].values
xi_x, xi_y = locations[['x']].values, locations[['y']].values

krig_im = krig[['est_cu_case_esipaper']].values[:, 0].reshape(300, 200)

# get the experimental variogram
bins = np.arange(200)
bin_center, gamma = gs.vario_estimate((x, y), values, bins)

# models to test
models = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Circular": gs.Circular,
    "Spherical": gs.Spherical,
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

print(estimate[0].shape)

# plot results
fig = plt.figure(dpi=150, figsize=(10,5))
gs = fig.add_gridspec(1, 2, wspace=0.4)
(ax1, ax2) = gs.subplots()

ax1.set_aspect('equal')
ax1.set_title('ordinary kriging')
img1 = ax1.imshow(krig_im, origin='lower', cmap='coolwarm')
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img1, orientation='vertical', cax=cax)

ax2.set_aspect('equal')
ax2.set_title('gstools automated o. kriging')
im = estimate[0].reshape(w, h)
#im = np.flipud(im)
img = ax2.imshow(im, origin='lower', cmap='coolwarm')
divider = make_axes_locatable(ax2)

cax = divider.append_axes("right", size="5%", pad=0.1)
colorbar(img, orientation='vertical', cax=cax)

plt.show()