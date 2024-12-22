import numpy as np
import matplotlib.pyplot as plt

from spatialize.data import load_drill_holes_andes_2D


def spherical_model(h, nugget, sill, range_):
    """
    Spherical variogram model for a single component.
    """
    return np.piecewise(h, [h <= range_, h > range_],
                        [lambda h: nugget + sill * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
                         lambda h: nugget + sill])


_, _, _, samples = load_drill_holes_andes_2D()

Y = [spherical_model(samples['lags_1'].values, 0, 0.17, 95) + spherical_model(samples['lags_1'].values, 0, 0.14, 220),
     samples['lags_1'].values]

plt.figure(dpi=150, figsize=(7, 5))
plt.title('Experimental and fitted variograms')
plt.xlabel('Distance(m)')
plt.ylabel('Variogram')
plt.scatter(x=samples['lags_1'].values, y=samples['gamma_1'].values, c=samples['npairs_1'].values, cmap='coolwarm')
plt.plot(Y[1], Y[0], color='slateblue')
plt.show()
