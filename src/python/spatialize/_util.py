from matplotlib import pyplot as plt
from rich.progress import track

from spatialize import SpatializeError


def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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


def get_progress_bar(list_like_obj, desc):
    if in_notebook():
        it = tqdm(range(len(list_like_obj)), desc=desc)
    else:
        it = track(range(len(list_like_obj)), description=desc)
    return it


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class GridSearchResult:
    def __init__(self, search_result_data):
        self.search_result_data = search_result_data

        data = self.search_result_data
        self.cv_error = data[['cv_error']]
        min_error = self.cv_error.min()['cv_error']
        self.best_params = data[data.cv_error <= min_error]

    def plot_cv_error(self, **kwargs):
        fig = plt.figure(figsize=(10, 4), dpi=150)
        gs = fig.add_gridspec(1, 2, wspace=0.45)
        (ax1, ax2) = gs.subplots()
        fig.suptitle("Cross Validation Error")
        self.cv_error.plot(kind='hist', ax=ax1,
                           title="Histogram",
                           rot=25,
                           color='skyblue', # edgecolor='black',
                           # colormap="Accent",
                           legend=False)
        self.cv_error.plot(kind='line', ax=ax2,
                           y='cv_error',
                           xlabel="Search result data index",
                           ylabel="Error",
                           color='skyblue', # edgecolor='black',
                           # colormap="Accent",
                           legend=False)

    def best_result(self, **kwargs):
        pass
