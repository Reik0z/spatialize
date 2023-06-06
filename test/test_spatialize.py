import os
import sys
import time

import numpy as np
import pandas as pd

from spatialize.gs.esi.precfunction import mse_precision, mae_precision

# if running from 'this' test directory then change to the
# project root directory
curr_dir = os.path.split(os.getcwd())[1]
if curr_dir == "test":
    os.chdir("..")

# load libspatialize
try:
    # check if it's already installed
    import libspatialize
except ImportError:
    # we are in dev env so the compiled library
    # must be in the project root directory.
    sys.path.append('.')
import libspatialize

# this is the 'k' for k-fold
k = 10


def test_esi_idw_2d(op='estimate'):
    samples = pd.read_csv('./test/testdata/data.csv')
    with open('./test/testdata/grid.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

    if op == 'estimate':
        values = libspatialize.esi_idw_2d(np.float32(samples[['x', 'y']].values),
                                          np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0,
                                          np.float32(locations[['X', 'Y']].values))
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        print(values.shape)
        print(np.nanmean(values, axis=1).shape)
        r = mae_precision(np.nanmean(values, axis=1).shape, values)
        print(r.shape)
        df.to_csv('./test/testdata/output/pyesi_idw_2d.csv', index=False)
    elif op == 'loo':
        values = libspatialize.loo_esi_idw_2d(np.float32(samples[['x', 'y']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0,
                                              np.float32(locations[['X', 'Y']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_2d.csv', index=False)
    elif op == 'kfold':
        values = libspatialize.kfold_esi_idw_2d(np.float32(samples[['x', 'y']].values),
                                                np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, k,
                                                np.float32(locations[['X', 'Y']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_2d.csv', index=False)


def test_esi_idw_3d(op='estimate'):
    locations = pd.read_csv('./test/testdata/box.csv')
    with open('./test/testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    if op == 'estimate':
        values = libspatialize.esi_idw_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                          np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0,
                                          np.float32(locations[['X', 'Y', 'Z']].values))
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        print(values.shape)
        print(np.nanmean(values, axis=1).shape)
        r = mae_precision(np.nanmean(values, axis=1).shape, values)
        print(r.shape)
        df.to_csv('./test/testdata/output/pyesi_idw_3d.csv', index=False)
    elif op == 'loo':
        values = libspatialize.loo_esi_idw_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0,
                                              np.float32(locations[['X', 'Y', 'Z']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_3d.csv', index=False)
    elif op == 'kfold':
        values = libspatialize.kfold_esi_idw_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, k,
                                                np.float32(locations[['X', 'Y', 'Z']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_3d.csv', index=False)


def test_esi_kri_2d(op='estimate'):
    samples = pd.read_csv('./test/testdata/data.csv')
    with open('./test/testdata/grid.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

    if op == 'estimate':
        values = libspatialize.esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                              np.float32(locations[['X', 'Y']].values))
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_kri_2d.csv', index=False)
    elif op == 'loo':
        values = libspatialize.loo_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                                  np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                  np.float32(locations[['X', 'Y']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_kri_2d.csv', index=False)
    elif op == 'kfold':
        values = libspatialize.kfold_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                                    np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                    k, np.float32(locations[['X', 'Y']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_kri_2d.csv', index=False)


def test_esi_kri_3d(op='estimate'):
    locations = pd.read_csv('./test/testdata/box.csv')
    with open('./test/testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    if op == 'estimate':
        values = libspatialize.esi_kriging_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                              np.float32(locations[['X', 'Y', 'Z']].values))
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_kri_3d.csv', index=False)
    elif op == 'loo':
        values = libspatialize.loo_esi_kriging_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                  np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                  np.float32(locations[['X', 'Y', 'Z']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_kri_3d.csv', index=False)
    elif op == 'kfold':
        values = libspatialize.kfold_esi_kriging_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                    np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                    k, np.float32(locations[['X', 'Y', 'Z']].values))
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_kri_3d.csv', index=False)


if __name__ == '__main__':
    op = 'kfold'
    outdir = './test/testdata/output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    t1 = time.time()
    test_esi_idw_2d(op='estimate')
    print('test_esi_idw_2d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_idw_2d(op='loo')
    # print('test_loo_esi_idw_2d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_idw_2d(op='kfold')
    # print('test_kfold_esi_idw_2d: ', time.time() - t1, '[s]')

    t1 = time.time()
    test_esi_idw_3d(op='estimate')
    print('test_esi_idw_3d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_idw_3d(op='loo')
    # print('test_loo_esi_idw_3d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_idw_3d(op='kfold')
    # print('test_kfold_esi_idw_3d: ', time.time() - t1, '[s]')
    #
    # t1 = time.time()
    # test_esi_kri_2d(op='estimate')
    # print('test_esi_kri_2d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_kri_2d(op='loo')
    # print('test_loo_esi_kri_2d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_kri_2d(op='kfold')
    # print('test_kfold_esi_kri_2d: ', time.time() - t1, '[s]')
    #
    # t1 = time.time()
    # test_esi_kri_3d(op='estimate')
    # print('test_esi_kri_3d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_kri_3d(op='loo')
    # print('test_loo_esi_kri_3d: ', time.time() - t1, '[s]')
    # t1 = time.time()
    # test_esi_kri_3d(op='kfold')
    # print('test_kfold_esi_kri_3d: ', time.time() - t1, '[s]')
