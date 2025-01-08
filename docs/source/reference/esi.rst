.. _esi:

********************
ESI Functions
********************

.. automodule:: spatialize.gs.esi

.. autofunction:: esi_griddata

.. autofunction:: esi_nongriddata

.. function:: esi_hparams_search(points, values, xi)

      Perform a hyperparameter search for ESI.

      :param points: The input points. Contains the coordinates of known data points. 
       This is an $N_s \times D$ array, where $N_s$ is the number of data points, and
       $D$ is the number of dimensions.
      :param values: The input values associated with each point in points. This must
       be a 1D array of length $N_s$. 
      :param xi: The interpolation points. If the data are gridded, they correspond to an 
       array of grids of $D$ components, each with the dimensions of one of the grid
       faces, $d_1 \times d_2 = N_{x^*}$, where $N_{x^*}$ is the total number of 
       unmeasured locations to estimate. Each component of this array represents the
       coordinate matrix on the corresponding axis, as returned by the functions 
       ``numpy.mgrid`` in Numpy, or ``meshgrid`` in Matlab or R.

       If the data are not gridded, they are simply the locations at which to evaluate 
       the interpolation. It is then an $N_{x^*} \times D$ array.

       In both cases, $D$ is the dimensionality of each location, which coincides with the
       dimensionality of the ``points``.
      :param kwargs: Additional keyword arguments.
      :return: The grid search result.

.. autoclass:: ESIGridSearchResult
   :members: __init__
   :exclude-members: __init__, load, save
   :undoc-members:
   :inherited-members: 

.. autoclass:: ESIResult
   :members:
   :exclude-members: load, save
   :undoc-members:
   :show-inheritance: estimation
   :inherited-members: 

.. toctree::
   :maxdepth: 2
   :hidden:

   
