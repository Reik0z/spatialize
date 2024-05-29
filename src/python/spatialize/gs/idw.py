from spatialize.gs.esi import esi_griddata
import spatialize.gs.esi.aggfunction as af


def griddata(points, values, xi, exponent=2.0):
    grid_z3, _ = esi_griddata(points, values, xi,
                              base_interpolator="idw",
                              exponent=exponent,
                              n_partitions=1, alpha=-1000,
                              # agg_function=af.identity
                              )

    return grid_z3
