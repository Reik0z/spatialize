import unittest

from spatialize.gs.esi._main import signature_overload, SpatializeError


class ESIModule(unittest.TestCase):
    def test_overload_decorator(self):
        @signature_overload(default_base_interpolator="idw", base_interp_case={
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


if __name__ == '__main__':
    unittest.main()
