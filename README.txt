# spatialize
spatialize: A Python wrapper for C++ ESI library

requisites: numpy, pandas, jupyter, scipy, hvplot, scikit-learn
pip install --upgrade metpy

test_esi_idw_2d:  14.463987350463867 [s]
test_loo_esi_idw_2d:  0.15934371948242188 [s]
test_kfold_esi_idw_2d:  0.16301417350769043 [s]

test_esi_idw_3d:  99.11273074150085 [s]
test_loo_esi_idw_3d:  3.8330681324005127 [s]
test_kfold_esi_idw_3d:  3.43314790725708 [s]

test_esi_kri_2d:  32.16479682922363 [s]
test_loo_esi_kri_2d:  48.66352105140686 [s]
test_kfold_esi_kri_2d:  23.50082802772522 [s]

test_esi_kri_3d:  558.7661159038544 [s]
test_loo_esi_kri_3d:  52645.69627690315 [s]
test_kfold_esi_kri_3d:  20696.35566496849 [s]