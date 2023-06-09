import unittest

import numpy as np

from spatialize._util import signature_overload
from spatialize import SpatializeError


def test_griddata_kriging():
    def func(x, y):  # a kind of cubic function
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

    rng = np.random.default_rng()
    points = rng.random((1000, 2))
    values = func(points[:, 0], points[:, 1])

    import spatialize.gs.esi.aggfunction as af
    import spatialize.gs.esi.precfunction as pf
    from spatialize.gs.esi import esi_griddata

    grid_z3, grid_z3p = esi_griddata(points, values, (grid_x, grid_y),
                                     base_interpolator="idw",
                                     n_partitions=100, alpha=0.99,
                                     exponent=7.0,
                                     agg_function=af.mean, prec_function=pf.mae_precision)


class ESIModule(unittest.TestCase):
    def test_overload_decorator(self):
        @signature_overload(pivot_arg=("base_interpolator", "idw", "base interpolator"),
                            common_args={"m": 2},
                            specific_args={
                                "idw": {"c": 7},
                                "kriging": {"k": 100}
                            })
        def f(a, b, **kwargs):
            if kwargs["base_interpolator"] == "idw":
                extra = kwargs["c"]
            if kwargs["base_interpolator"] == "kriging":
                extra = kwargs["k"]

            m = kwargs["m"]
            return m + a + b + extra

        # take the default value for c (=7)
        r = f(2, 3)
        self.assertEqual(r, 14)

        # take the default value for c (=7)
        r = f(2, 3, base_interpolator="idw")
        self.assertEqual(r, 14)

        # modify the default value for c (=8)
        r = f(2, 3, base_interpolator="idw", c=8)
        self.assertEqual(r, 15)

        # try an argument not in the context of idw
        try:
            f(2, 4, base_interpolator="idw", d=5)
        except SpatializeError as e:
            self.assertEqual(str(e), "Argument 'd' not recognized for 'idw' base interpolator")

        # try an argument not in the context of kriging
        try:
            f(2, 4, base_interpolator="kriging", c=5)
        except SpatializeError as e:
            self.assertEqual(str(e), "Argument 'c' not recognized for 'kriging' base interpolator")

        r = f(2, 4, base_interpolator="kriging")
        self.assertEqual(r, 108)

        r = f(2, 4, base_interpolator="kriging", k=101)
        self.assertEqual(r, 109)

        try:
            f(2, 4, base_interpolator="rbf", k=101)
        except SpatializeError as e:
            self.assertEqual(str(e), "Base interpolator 'rbf' not supported")

    def test_griddata(self):
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

        rng = np.random.default_rng()
        points = rng.random((1000, 2))
        values = self.func(points[:, 0], points[:, 1])

        import spatialize.gs.esi.aggfunction as af
        import spatialize.gs.esi.precfunction as pf
        from spatialize.gs.esi import esi_griddata

        _, _ = esi_griddata(points, values, (grid_x, grid_y),
                            base_interpolator="idw",
                            exponent=7.0,
                            n_partitions=100, alpha=0.97,
                            agg_function=af.mean, prec_function=pf.mae_precision)

        _, _ = esi_griddata(points, values, (grid_x, grid_y),
                            base_interpolator="kriging",
                            model="cubic", nugget=0.1, range=5000.0,
                            n_partitions=100, alpha=0.97,
                            agg_function=af.mean, prec_function=pf.mae_precision)

    def test_hparams_search(self):
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

        rng = np.random.default_rng()
        points = rng.random((1000, 2))
        values = self.func(points[:, 0], points[:, 1])

        from spatialize.gs.esi import esi_hparams_search
        b_params = esi_hparams_search(points, values, (grid_x, grid_y),
                                      base_interpolator="kriging", griddata=True, k=-1,
                                      alpha=list(np.flip(np.arange(0.70, 0.75, 0.01))))
        print(b_params)

    @staticmethod
    def func(x, y):  # a kind of "cubic" function
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2


if __name__ == '__main__':
    unittest.main()
