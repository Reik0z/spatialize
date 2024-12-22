import os
import sys
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from spatialize.data import load_drill_holes_andes_2D, load_drill_holes_andes_3D

# if running from 'this' test directory then change to the
# project root directory
curr_dir = os.path.split(os.getcwd())[1]
if curr_dir == "test":
    os.chdir(".")

# load libspatialize
try:
    # check if it's already installed
    import libspatialize
except ImportError:
    # we are in dev env so the compiled library
    # must be in the project root directory.
    sys.path.append('.')
    sys.path.append('..')

sys.path.append('./src/python/')
sys.path.append('../src/python/')

import libspatialize as sp

from spatialize.logging import default_singleton_callback
# this is the 'k' for k-fold
k = 10


def pperc(s):
    default_singleton_callback(s)


def test_partitions():
    X = 100000.0 * np.random.random((10000, 3))
    partitions = sp.get_partitions_using_esi(np.float32(X), 10, 0.7, 206936)
    df = pd.DataFrame(partitions, columns=['p%02d' % (i + 1,) for i in range(10)])
    df['x'] = X[:, 0]
    df['y'] = X[:, 1]
    df['z'] = X[:, 2]
    df.to_csv('./test/testdata/output/partitions.csv', index=False)


def test_using_model(op):
    samples, locations = load_drill_holes_andes_3D()

    # with open('./test/testdata/esi_idw_model.json', 'r') as f:
    with open('./testdata/esi_kriging_model.json', 'r') as f:
        model = json.loads(f.read())

    if op == 'estimate':
        values = sp.estimation_using_model(model, np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/use_esi_%s_model.csv' % model['esi_type'], index=False)
    elif op == 'loo':
        values = sp.loo_using_model(model, pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_esi_%s_model.csv' % model['esi_type'], index=False)
    elif op == 'kfold':
        values = sp.kfold_using_model(model, k, 206936, pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_esi_%s_model.csv' % model['esi_type'], index=False)


def test_nn_idw(op='estimate'):
    samples, locations = load_drill_holes_andes_3D()

    if op == 'estimate':
        values = sp.estimation_nn_idw(np.float32(samples[['x', 'y', 'z']].values),
                                      np.float32(samples[['cu']].values[:, 0]), 100.0, 2.0,
                                      np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = values
        df.to_csv('./test/testdata/output/nn_idw_3d.csv', index=False)
    elif op == 'loo':
        values = sp.loo_nn_idw(np.float32(samples[['x', 'y', 'z']].values), np.float32(samples[['cu']].values[:, 0]),
                               100.0, 2.0, pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['py_cu'] = values
        df.to_csv('./test/testdata/output/loo_nn_idw_3d.csv', index=False)
    elif op == 'kfold':
        values = sp.kfold_nn_idw(np.float32(samples[['x', 'y', 'z']].values), np.float32(samples[['cu']].values[:, 0]),
                                 100.0, 2.0, 5, 206936, pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['py_cu'] = values
        df.to_csv('./test/testdata/output/kfold_idw_3d.csv', index=False)


def test_esi_idw_2d(op='estimate'):
    samples, locations, _, _ = load_drill_holes_andes_2D()

    if op == 'estimate':
        model, values = sp.estimation_esi_idw(np.float32(samples[['x', 'y']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936,
                                              np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_2d.csv', index=False)
    elif op == 'loo':
        model, values = sp.loo_esi_idw(np.float32(samples[['x', 'y']].values), np.float32(samples[['cu']].values[:, 0]),
                                       100, 0.7, 2.0, 206936, np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_2d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw(np.float32(samples[['x', 'y']].values),
                                         np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936, k, 84987,
                                         np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_2d.csv', index=False)


def test_esi_idw_3d(op='estimate'):
    samples, locations = load_drill_holes_andes_3D()

    if op == 'estimate':
        model, values = sp.estimation_esi_idw(np.float32(samples[['x', 'y', 'z']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936,
                                              np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_3d.csv', index=False)
        with open('./test/testdata/esi_idw_model.json', 'w') as f:
            json.dump(model, f, indent=4)
    elif op == 'loo':
        model, values = sp.loo_esi_idw(np.float32(samples[['x', 'y', 'z']].values),
                                       np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936,
                                       np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_3d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw(np.float32(samples[['x', 'y', 'z']].values),
                                         np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936, k, 84987,
                                         np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_3d.csv', index=False)


def test_esi_idw_anis_2d(op='estimate'):
    samples, locations, _, _ = load_drill_holes_andes_2D()

    if op == 'estimate':
        model, values = sp.estimation_esi_idw_anisotropic_2d(np.float32(samples[['x', 'y']].values),
                                                             np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936,
                                                             np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_anis_2d.csv', index=False)
    elif op == 'loo':
        model, values = sp.loo_esi_idw_anisotropic_2d(np.float32(samples[['x', 'y']].values),
                                                      np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936,
                                                      np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_anis_2d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw_anisotropic_2d(np.float32(samples[['x', 'y']].values),
                                                        np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936, k,
                                                        84987, np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_anis_2d.csv', index=False)


def test_esi_idw_anis_3d(op='estimate'):
    samples, locations = load_drill_holes_andes_3D()

    if op == 'estimate':
        model, values = sp.estimation_esi_idw_anisotropic_3d(np.float32(samples[['x', 'y', 'z']].values),
                                                             np.float32(samples[['cu']].values[:, 0]), 5, 0.7, 206936,
                                                             np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_anis_3d.csv', index=False)
        with open('./test/testdata/esi_idw_anis_model.json', 'w') as f:
            json.dump(model, f, indent=4)
    elif op == 'loo':
        model, values = sp.loo_esi_idw_anisotropic_3d(np.float32(samples[['x', 'y', 'z']].values),
                                                      np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936,
                                                      np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_anis_3d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw_anisotropic_3d(np.float32(samples[['x', 'y', 'z']].values),
                                                        np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936, k,
                                                        84987, np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_anis_3d.csv', index=False)


def test_esi_kri_2d(op='estimate'):
    samples, locations, _, _ = load_drill_holes_andes_2D()

    if op == 'estimate':
        model, values = sp.estimation_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                                     np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                     206936, np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_kri_2d.csv', index=False)
    elif op == 'loo':
        model, values = sp.loo_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                              206936, np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_kri_2d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                                np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                206936, k, 84987, np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_kri_2d.csv', index=False)


def test_esi_kri_3d(op='estimate'):
    samples, locations = load_drill_holes_andes_3D()

    if op == 'estimate':
        model, values = sp.estimation_esi_kriging_3d(np.float32(samples[['x', 'y', 'z']].values),
                                                     np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                     206936, np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_kri_3d.csv', index=False)
        with open('./test/testdata/esi_kriging_model.json', 'w') as f:
            json.dump(model, f, indent=4)
    elif op == 'loo':
        model, values = sp.loo_esi_kriging_3d(np.float32(samples[['x', 'y', 'z']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                              206936, np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_kri_3d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_kriging_3d(np.float32(samples[['x', 'y', 'z']].values),
                                                np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                206936, k, 84987, np.float32(locations[['x', 'y', 'z']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_kri_3d.csv', index=False)


def test_voronoi_idw(op='estimate'):
    samples, locations, _, _ = load_drill_holes_andes_2D()

    if op == 'estimate':
        model, values = sp.estimation_voronoi_idw(np.float32(samples[['x', 'y']].values),
                                                  np.float32(samples[['cu']].values[:, 0]), 50, 1.0, 2.0, 206936,
                                                  np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(locations, columns=['x', 'y', 'z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyvoronoi_idw.csv', index=False)
    elif op == 'loo':
        model, values = sp.loo_voronoi_idw(np.float32(samples[['x', 'y']].values), np.float32(samples[['cu']].values[:, 0]),
                                       500, -1.0, 2.0, 206936, np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyvoronoi_idw_2d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_voronoi_idw(np.float32(samples[['x', 'y']].values),
                                         np.float32(samples[['cu']].values[:, 0]), 500, 1.0, 2.0, 206936, k, 84987,
                                         np.float32(locations[['x', 'y']].values), pperc)
        df = pd.DataFrame(samples, columns=['x', 'y', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyvoronoi_idw_2d.csv', index=False)

if __name__ == '__main__':
    # t1 = time.time()
    # test_voronoi_idw(op='estimate')
    # print('test voronoi_idw: ', time.time() - t1, '[s]', flush=True)

    # t1 = time.time()
    # test_voronoi_idw(op='loo')
    # print('test loo_voronoi_idw: ', time.time() - t1, '[s]', flush=True)

    # t1 = time.time()
    # test_voronoi_idw(op='kfold')
    # print('test kfold_voronoi_idw: ', time.time() - t1, '[s]', flush=True)

    # t1 = time.time()
    # test_nn_idw(op='estimate')
    # print('test nn_idw: ', time.time() - t1, '[s]', flush=True)
    # t1 = time.time()
    # test_nn_idw(op='loo')
    # print('test loo_nn_idw: ', time.time() - t1, '[s]', flush=True)
    # t1 = time.time()
    # test_nn_idw(op='kfold')
    # print('test kfold_nn_idw: ', time.time() - t1, '[s]', flush=True)

    # t1 = time.time()
    # test_esi_idw_2d(op='estimate')
    # print('test esi_idw_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_2d(op='loo')
    # print('test loo_esi_idw_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_2d(op='kfold')
    # print('test kfold_esi_idw_2d: ', time.time()-t1, '[s]', flush=True)

    t1 = time.time()
    test_esi_idw_3d(op='estimate')
    print('test esi_idw_3d: ', time.time()-t1, '[s]', flush=True)
    t1 = time.time()
    test_esi_idw_3d(op='loo')
    print('test loo_esi_idw_3d: ', time.time()-t1, '[s]', flush=True)
    t1 = time.time()
    test_esi_idw_3d(op='kfold')
    print('test kfold_esi_idw_3d: ', time.time()-t1, '[s]', flush=True)

    # t1 = time.time()
    # test_esi_kri_2d(op='estimate')
    # print('test esi_kri_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_kri_2d(op='loo')
    # print('test loo_esi_kri_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_kri_2d(op='kfold')
    # print('test kfold_esi_kri_2d: ', time.time()-t1, '[s]', flush=True)

    # t1 = time.time()
    # test_esi_kri_3d(op='estimate')
    # print('test esi_kri_3d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_kri_3d(op='loo')
    # print('test loo_esi_kri_3d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_kri_3d(op='kfold')
    # print('test kfold_esi_kri_3d: ', time.time()-t1, '[s]', flush=True)

    # t1 = time.time()
    # test_esi_idw_anis_2d(op='estimate')
    # print('test esi_idw_anis_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_anis_2d(op='loo')
    # print('test loo_esi_idw_anis_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_anis_2d(op='kfold')
    # print('test kfold_esi_idw_anis_2d: ', time.time()-t1, '[s]', flush=True)

    # t1 = time.time()
    # test_esi_idw_anis_3d(op='estimate')
    # print('test esi_idw_anis_3d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_anis_3d(op='loo')
    # print('test loo_esi_idw_anis_3d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_anis_3d(op='kfold')
    # print('test kfold_esi_idw_anis_3d: ', time.time()-t1, '[s]', flush=True)

    # test_using_model(op='estimate')
    # print('estimation ok')
    # test_using_model(op='loo')
    # print('loo ok')
    # test_using_model(op='kfold')
    # print('kfold ok')

    # test_partitions()
