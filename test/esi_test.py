import unittest

import numpy as np

from spatialize.gs.esi._main import signature_overload
from spatialize import SpatializeError


class ESIModule(unittest.TestCase):
    def test_overload_decorator(self):
        @signature_overload(default_base_interpolator="idw",
                            common_args={"base_interpolator": "idw"},
                            specific_args={
                                "idw": {"c": 7},
                                "kriging": {"k": 100}
                            })
        def f(a, b, **kwargs):
            if kwargs["base_interpolator"] == "idw":
                extra = kwargs["c"]
            if kwargs["base_interpolator"] == "kriging":
                extra = kwargs["k"]
            return a + b + extra

        # take the default value for c (=7)
        r = f(2, 3)
        self.assertEqual(r, 12)

        # take the default value for c (=7)
        r = f(2, 3, base_interpolator="idw")
        self.assertEqual(r, 12)

        # modify the default value for c (=8)
        r = f(2, 3, base_interpolator="idw", c=8)
        self.assertEqual(r, 13)

        # try an argument not in the context of idw
        try:
            f(2, 4, base_interpolator="idw", d=5)
        except SpatializeError:
            self.assertTrue("Correct")

        # try an argument not in the context of kriging
        try:
            f(2, 4, base_interpolator="kriging", c=5)
        except SpatializeError:
            self.assertTrue("Correct")

        r = f(2, 4, base_interpolator="kriging")
        self.assertEqual(r, 106)

        r = f(2, 4, base_interpolator="kriging", k=101)
        self.assertEqual(r, 107)

        try:
            f(2, 4, base_interpolator="rbf", k=101)
        except SpatializeError:
            self.assertTrue("Correct")

    def test_griddata(self):
        def func(x, y):  # a kind of cubic function
            return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

        rng = np.random.default_rng()
        points = rng.random((1000, 2))
        values = func(points[:, 0], points[:, 1])

        import spatialize.gs.esi.aggfunction as af
        import spatialize.gs.esi.precfunction as pf
        from spatialize.gs.esi import griddata

        grid_z3, _ = griddata(points, values, (grid_x, grid_y), n_partitions=100, alpha=0.99, exponent=7.0,
                              agg_function=af.mean, prec_function=pf.mse_precision)


if __name__ == '__main__':
    unittest.main()
