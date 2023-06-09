import numpy as np

from spatialize import SpatializeError


def signature_overload(pivot_arg, common_args, specific_args):
    def outer_function(func):
        def inner_function(*args, **kwargs):
            pivot_key, pivot_default_value, pivot_desc = pivot_arg

            if pivot_key not in common_args:
                common_args[pivot_key] = pivot_default_value

            # if a common argument is needed for the pivot argument
            # and is not in kwargs then add it with its declared
            # default value
            for arg in common_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = common_args[arg]

            # get the specific args for the current pivot key
            pk = kwargs[pivot_key]

            if pk not in specific_args:
                raise SpatializeError(f"{pivot_desc.capitalize()} '{pk}' not supported")

            spec_args = specific_args[pk]

            # if the specific argument is needed for the pivot argument
            # and is not in kwargs then add it with its declared
            # default value
            for arg in spec_args.keys():
                if arg not in kwargs:
                    kwargs[arg] = spec_args[arg]

            # check that all arguments are consistent
            # with the pivot key
            for arg in kwargs.keys():
                if arg != pivot_key and arg not in spec_args and arg not in common_args:
                    raise SpatializeError(f"Argument '{arg}' not recognized for '{pk}' {pivot_desc.lower()}")

            return func(*args, **kwargs)

        return inner_function

    return outer_function


def is_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            raise ImportError("console")
            return False
        if "VSCODE_PID" in os.environ:  # pragma: no cover
            raise ImportError("vscode")
            return False
    except:
        return False
    else:  # pragma: no cover
        return True


def flatten_grid_data(xi):
    try:
        if len(xi) == 1:
            ng_xi = np.column_stack((xi.flatten()))
        elif len(xi) == 2:
            (grid_x, grid_y) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten()))
        elif len(xi) == 3:
            (grid_x, grid_y, grid_z) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()))
        elif len(xi) == 4:
            (grid_x, grid_y, grid_z, grid_t) = xi
            ng_xi = np.column_stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), grid_t.flatten()))
        else:
            raise Exception
    except:
        raise SpatializeError("No grid data positions found")
    return ng_xi, grid_x.shape
