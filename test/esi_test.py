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
                                     local_interpolator="idw",
                                     n_partitions=100, alpha=0.99,
                                     exponent=7.0,
                                     agg_function=af.mean, prec_function=pf.mae_loss)


class ESIModule(unittest.TestCase):
    def test_overload_decorator(self):
        @signature_overload(pivot_arg=("local_interpolator", "idw", "local interpolator"),
                            common_args={"m": 2},
                            specific_args={
                                "idw": {"c": 7},
                                "kriging": {"k": 100}
                            })
        def f(a, b, **kwargs):
            if kwargs["local_interpolator"] == "idw":
                extra = kwargs["c"]
            if kwargs["local_interpolator"] == "kriging":
                extra = kwargs["k"]

            m = kwargs["m"]
            return m + a + b + extra

        # take the default value for c (=7)
        r = f(2, 3)
        self.assertEqual(r, 14)

        # take the default value for c (=7)
        r = f(2, 3, local_interpolator="idw")
        self.assertEqual(r, 14)

        # modify the default value for c (=8)
        r = f(2, 3, local_interpolator="idw", c=8)
        self.assertEqual(r, 15)

        # try an argument not in the context of idw
        try:
            f(2, 4, local_interpolator="idw", d=5)
        except SpatializeError as e:
            self.assertEqual(str(e), "Argument 'd' not recognized for 'idw' local interpolator")

        # try an argument not in the context of kriging
        try:
            f(2, 4, local_interpolator="kriging", c=5)
        except SpatializeError as e:
            self.assertEqual(str(e), "Argument 'c' not recognized for 'kriging' local interpolator")

        r = f(2, 4, local_interpolator="kriging")
        self.assertEqual(r, 108)

        r = f(2, 4, local_interpolator="kriging", k=101)
        self.assertEqual(r, 109)

        try:
            f(2, 4, local_interpolator="rbf", k=101)
        except SpatializeError as e:
            self.assertEqual(str(e), "local interpolator 'rbf' not supported")

    def test_griddata(self):
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

        rng = np.random.default_rng()
        points = rng.random((1000, 2))
        values = self.func(points[:, 0], points[:, 1])

        import spatialize.gs.esi.aggfunction as af
        import spatialize.gs.esi.precfunction as pf
        from spatialize.gs.esi import esi_griddata

        _, _ = esi_griddata(points, values, (grid_x, grid_y),
                            local_interpolator="idw",
                            callback=self.progress,
                            exponent=7.0,
                            n_partitions=100, alpha=0.97,
                            agg_function=af.mean, prec_function=pf.mae_loss)

        _, _ = esi_griddata(points, values, (grid_x, grid_y),
                            local_interpolator="kriging",
                            callback=self.progress,
                            model="cubic", nugget=0.1, range=5000.0,
                            n_partitions=100, alpha=0.97,
                            agg_function=af.mean, prec_function=pf.mae_loss)

    def test_hparams_search(self):
        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]

        rng = np.random.default_rng()
        points = rng.random((1000, 2))
        values = self.func(points[:, 0], points[:, 1])

        # from spatialize.gs.esi import esi_hparams_search
        # b_params = esi_hparams_search(points, values, (grid_x, grid_y),
        #                               local_interpolator="kriging", griddata.py=True, k=10,
        #                               alpha=(0.70, 0.65))
        # print(b_params)




        # from spatialize.gs.esi import esi_hparams_search
        # b_params = esi_hparams_search(points, values, (grid_x, grid_y),
        #                               local_interpolator="idw", griddata.py=True, k=10,
        #                               alpha=list(reversed((0.5, 0.6, 0.8, 0.9, 0.95))))

        # from spatialize.gs.esi import esi_hparams_search
        # b_params = esi_hparams_search(points, values, (grid_x, grid_y),
        #                               local_interpolator="idw", griddata.py=True, k=10,
        #                               alpha=(0.985, 0.97, 0.95))
        # print(b_params)

    @staticmethod
    def func(x, y):  # a kind of "cubic" function
        return x * (1 - x) * np.cos(4 * np.pi * x) * np.sin(4 * np.pi * y ** 2) ** 2

    @staticmethod
    def progress(s):
        print(f'processing ... {int(float(s.split()[1][:-1]))}%\r', end="")


if __name__ == '__main__':
    unittest.main()
