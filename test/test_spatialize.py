import os
import sys
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

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

import libspatialize as sp

# this is the 'k' for k-fold
k = 10


def pperc(s):
    print(f'processing ... {int(float(s.split()[1][:-1]))}%\r', end="")


def test_partitions():
    X = 100000.0 * np.random.random((10000, 3))
    partitions = sp.get_partitions_using_esi(np.float32(X), 10, 0.7, 206936)
    df = pd.DataFrame(partitions, columns=['p%02d' % (i + 1,) for i in range(10)])
    df['X'] = X[:, 0]
    df['Y'] = X[:, 1]
    df['Z'] = X[:, 2]
    df.to_csv('./test/testdata/output/partitions.csv', index=False)


def test_using_model(op):
    locations = pd.read_csv('./testdata/box.csv')
    # with open('./test/testdata/esi_idw_model.json', 'r') as f:
    with open('./testdata/esi_kriging_model.json', 'r') as f:
        model = json.loads(f.read())
    with open('./testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    if op == 'estimate':
        values = sp.estimation_using_model(model, np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/use_esi_%s_model.csv' % model['esi_type'], index=False)
    elif op == 'loo':
        values = sp.loo_using_model(model, pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_esi_%s_model.csv' % model['esi_type'], index=False)
    elif op == 'kfold':
        values = sp.kfold_using_model(model, k, 206936, pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_esi_%s_model.csv' % model['esi_type'], index=False)


def test_nn_idw(op='estimate'):
    locations = pd.read_csv('./test/testdata/box.csv')
    with open('./test/testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    if op == 'estimate':
        values = sp.estimation_nn_idw(np.float32(samples[['X', 'Y', 'Z']].values),
                                      np.float32(samples[['cu']].values[:, 0]), 100.0, 2.0,
                                      np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = values
        df.to_csv('./test/testdata/output/nn_idw_3d.csv', index=False)
    elif op == 'loo':
        values = sp.loo_nn_idw(np.float32(samples[['X', 'Y', 'Z']].values), np.float32(samples[['cu']].values[:, 0]),
                               100.0, 2.0, pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['py_cu'] = values
        df.to_csv('./test/testdata/output/loo_nn_idw_3d.csv', index=False)
    elif op == 'kfold':
        values = sp.kfold_nn_idw(np.float32(samples[['X', 'Y', 'Z']].values), np.float32(samples[['cu']].values[:, 0]),
                                 100.0, 2.0, 5, 206936, pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['py_cu'] = values
        df.to_csv('./test/testdata/output/kfold_idw_3d.csv', index=False)


def test_esi_idw_2d(op='estimate'):
    samples = pd.read_csv('./test/testdata/data.csv')
    with open('./test/testdata/grid.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

    if op == 'estimate':
        model, values = sp.estimation_esi_idw(np.float32(samples[['x', 'y']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936,
                                              np.float32(locations[['X', 'Y']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_2d.csv', index=False)
    elif op == 'loo':
        model, values = sp.loo_esi_idw(np.float32(samples[['x', 'y']].values), np.float32(samples[['cu']].values[:, 0]),
                                       100, 0.7, 2.0, 206936, np.float32(locations[['X', 'Y']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_2d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw(np.float32(samples[['x', 'y']].values),
                                         np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936, k, 84987,
                                         np.float32(locations[['X', 'Y']].values), pperc)
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
        model, values = sp.estimation_esi_idw(np.float32(samples[['X', 'Y', 'Z']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936,
                                              np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_3d.csv', index=False)
        with open('./test/testdata/esi_idw_model.json', 'w') as f:
            json.dump(model, f, indent=4)
    elif op == 'loo':
        model, values = sp.loo_esi_idw(np.float32(samples[['X', 'Y', 'Z']].values),
                                       np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936,
                                       np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_3d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw(np.float32(samples[['X', 'Y', 'Z']].values),
                                         np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 2.0, 206936, k, 84987,
                                         np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_3d.csv', index=False)


def test_esi_idw_anis_2d(op='estimate'):
    samples = pd.read_csv('./test/testdata./test/testdata.csv')
    with open('./test/testdata/grid.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

    if op == 'estimate':
        model, values = sp.estimation_esi_idw_anisotropic_2d(np.float32(samples[['x', 'y']].values),
                                                             np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936,
                                                             np.float32(locations[['X', 'Y']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_anis_2d.csv', index=False)
    elif op == 'loo':
        model, values = sp.loo_esi_idw_anisotropic_2d(np.float32(samples[['x', 'y']].values),
                                                      np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936,
                                                      np.float32(locations[['X', 'Y']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_anis_2d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw_anisotropic_2d(np.float32(samples[['x', 'y']].values),
                                                        np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936, k,
                                                        84987, np.float32(locations[['X', 'Y']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_anis_2d.csv', index=False)


def test_esi_idw_anis_3d(op='estimate'):
    locations = pd.read_csv('./test/testdata/box.csv')
    with open('./test/testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    if op == 'estimate':
        model, values = sp.estimation_esi_idw_anisotropic_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                             np.float32(samples[['cu']].values[:, 0]), 5, 0.7, 206936,
                                                             np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_idw_anis_3d.csv', index=False)
        with open('./test/testdata/esi_idw_anis_model.json', 'w') as f:
            json.dump(model, f, indent=4)
    elif op == 'loo':
        model, values = sp.loo_esi_idw_anisotropic_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                      np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936,
                                                      np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_idw_anis_3d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_idw_anisotropic_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                        np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 206936, k,
                                                        84987, np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_idw_anis_3d.csv', index=False)


def test_esi_kri_2d(op='estimate'):
    samples = pd.read_csv('./test/testdata./test/testdata.csv')
    with open('./test/testdata/grid.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    locations = pd.DataFrame(aux, columns=['X', 'Y', 'Z'])

    if op == 'estimate':
        model, values = sp.estimation_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                                     np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                     206936, np.float32(locations[['X', 'Y']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_kri_2d.csv', index=False)
    elif op == 'loo':
        model, values = sp.loo_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                              206936, np.float32(locations[['X', 'Y']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_kri_2d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_kriging_2d(np.float32(samples[['x', 'y']].values),
                                                np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                206936, k, 84987, np.float32(locations[['X', 'Y']].values), pperc)
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
        model, values = sp.estimation_esi_kriging_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                     np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                     206936, np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/pyesi_kri_3d.csv', index=False)
        with open('./test/testdata/esi_kriging_model.json', 'w') as f:
            json.dump(model, f, indent=4)
    elif op == 'loo':
        model, values = sp.loo_esi_kriging_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                              np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                              206936, np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['loo_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/loo_pyesi_kri_3d.csv', index=False)
    elif op == 'kfold':
        model, values = sp.kfold_esi_kriging_3d(np.float32(samples[['X', 'Y', 'Z']].values),
                                                np.float32(samples[['cu']].values[:, 0]), 100, 0.7, 1, 0.1, 5000.0,
                                                206936, k, 84987, np.float32(locations[['X', 'Y', 'Z']].values), pperc)
        df = pd.DataFrame(samples, columns=['X', 'Y', 'Z', 'cu'])
        df['kfold_py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./test/testdata/output/kfold_pyesi_kri_3d.csv', index=False)


if __name__ == '__main__':
    t1 = time.time()
    test_nn_idw(op='estimate')
    print('test nn_idw: ', time.time() - t1, '[s]', flush=True)
    t1 = time.time()
    test_nn_idw(op='loo')
    print('test loo_nn_idw: ', time.time() - t1, '[s]', flush=True)
    t1 = time.time()
    test_nn_idw(op='kfold')
    print('test kfold_nn_idw: ', time.time() - t1, '[s]', flush=True)

    # t1 = time.time()
    # test_esi_idw_2d(op='estimate')
    # print('test esi_idw_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_2d(op='loo')
    # print('test loo_esi_idw_2d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_2d(op='kfold')
    # print('test kfold_esi_idw_2d: ', time.time()-t1, '[s]', flush=True)

    # t1 = time.time()
    # test_esi_idw_3d(op='estimate')
    # print('test esi_idw_3d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_3d(op='loo')
    # print('test loo_esi_idw_3d: ', time.time()-t1, '[s]', flush=True)
    # t1 = time.time()
    # test_esi_idw_3d(op='kfold')
    # print('test kfold_esi_idw_3d: ', time.time()-t1, '[s]', flush=True)

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
