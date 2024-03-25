"""
CREATE UNIQUE INDEX idx_leaves_id ON leaves (id);
CREATE INDEX idx_leaves_axis ON leaves(axis);
CREATE INDEX idx_leaves_tree_id ON leaves(tree_id);
CREATE INDEX idx_leaves_tree_tau ON leaves(tau);
CREATE INDEX idx_leaves_tree_cut ON leaves(cut);
CREATE INDEX idx_leaves_tree_height ON leaves(height);
CREATE UNIQUE INDEX idx_queries_id ON queries (id);
CREATE UNIQUE INDEX idx_samples_id ON samples (id);
"""
import sys, time, json

sys.path.append('./lib.linux-x86_64-cpython-39')

# load libspatialize
try:
    # check if it's already installed
    import libspatialite
except ImportError:
    # we are in dev env so the compiled library
    # must be in the project root directory.
    sys.path.append('.')
    sys.path.append('..')

import pandas as pd, numpy as np, libspatialite as sp


def pperc(s):
    print('Processing... {0}'.format(s), flush=True)


def creation():
    locations = pd.read_csv('./testdata/box.csv')
    with open('./testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])
    bbox = [
        [np.min([locations['X'].min(), samples['X'].min()]), np.max([locations['X'].max(), samples['X'].max()])],
        [np.min([locations['Y'].min(), samples['Y'].min()]), np.max([locations['Y'].max(), samples['Y'].max()])],
        [np.min([locations['Z'].min(), samples['Z'].min()]), np.max([locations['Z'].max(), samples['Z'].max()])],
    ]

    sp.create_esi_idw(
        "./testdata/output/create_test_esi_idw.db",
        np.float32(samples[['X', 'Y', 'Z']].values),
        np.float32(samples[['cu']].values[:, 0]),
        100, 0.7, np.float32(bbox), 2.0, 2007203
    )
    print('ESI IDW created', flush=True)

    sp.create_esi_kriging(
        "./testdata/output/create_test_esi_kriging.db",
        np.float32(samples[['X', 'Y', 'Z']].values),
        np.float32(samples[['cu']].values[:, 0]),
        100, 0.7, np.float32(bbox), 1, 0.1, 5000.0, 2007203
    )
    print('ESI Kriging created', flush=True)


def esi_idw(op):
    locations = pd.read_csv('./testdata/box.csv')
    with open('./testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    # estimate
    if op == "estimate":
        print("ESI IDW ESTIMATE...", flush=True)
        values = sp.estimation_esi_idw(
            "./testdata/output/test_esi_idw.db",
            np.float32(samples[['X', 'Y', 'Z']].values),
            np.float32(samples[['cu']].values[:, 0]),
            100, 0.7, 2.0, 2007203,
            np.float32(locations[['X', 'Y', 'Z']].values),
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/estimation_esi_idw.csv', index=False)
        print("DONE", flush=True)

    # loo
    if op == "loo":
        print("ESI IDW LOO...", flush=True)
        values = sp.loo_esi_idw(
            "./testdata/output/test_esi_idw.db",
            np.float32(samples[['X', 'Y', 'Z']].values),
            np.float32(samples[['cu']].values[:, 0]),
            100, 0.7, 2.0, 2007203,
            np.float32(locations[['X', 'Y', 'Z']].values),
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(samples[['X', 'Y', 'Z']], columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/loo_esi_idw.csv', index=False)
        print("DONE", flush=True)

    # kfold
    if op == "kfold":
        print("ESI IDW KFOLD...", flush=True)
        values = sp.kfold_esi_idw(
            "./testdata/output/test_esi_idw.db",
            np.float32(samples[['X', 'Y', 'Z']].values),
            np.float32(samples[['cu']].values[:, 0]),
            100, 0.7, 2.0, 2007203, 10, 206936,
            np.float32(locations[['X', 'Y', 'Z']].values),
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(samples[['X', 'Y', 'Z']], columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/kfold_esi_idw.csv', index=False)
        print("DONE", flush=True)


def kriging(op):
    locations = pd.read_csv('./testdata/box.csv')
    with open('./testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    # estimate
    if op == "estimate":
        print("ESI Kriging ESTIMATE...", flush=True)
        values = sp.estimation_esi_kriging(
            "./testdata/output/test_esi_kriging.db",
            np.float32(samples[['X', 'Y', 'Z']].values),
            np.float32(samples[['cu']].values[:, 0]),
            100, 0.7, 1, 0.1, 5000.0, 2007203,
            np.float32(locations[['X', 'Y', 'Z']].values),
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/estimation_esi_kriging.csv', index=False)
        print("DONE", flush=True)

    # loo
    if op == "loo":
        print("ESI Kriging LOO...", flush=True)
        values = sp.loo_esi_kriging(
            "./testdata/output/test_esi_kriging.db",
            np.float32(samples[['X', 'Y', 'Z']].values),
            np.float32(samples[['cu']].values[:, 0]),
            100, 0.7, 1, 0.1, 5000.0, 2007203,
            np.float32(locations[['X', 'Y', 'Z']].values),
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(samples[['X', 'Y', 'Z']], columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/loo_esi_kriging.csv', index=False)
        print("DONE", flush=True)

    # kfold
    if op == "kfold":
        print("ESI Kriging KFOLD...", flush=True)
        values = sp.kfold_esi_kriging(
            "./testdata/output/test_esi_kriging.db",
            np.float32(samples[['X', 'Y', 'Z']].values),
            np.float32(samples[['cu']].values[:, 0]),
            100, 0.7, 1, 0.1, 5000.0, 2007203, 10, 206936,
            np.float32(locations[['X', 'Y', 'Z']].values),
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(samples[['X', 'Y', 'Z']], columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/kfold_esi_kriging.csv', index=False)
        print("DONE", flush=True)


def stored_model(suffix, path, op):
    locations = pd.read_csv('./testdata/box.csv')
    with open('./testdata/muestras.dat', 'r') as data:
        lines = data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    samples = pd.DataFrame(aux, columns=['X', 'Y', 'Z', 'cu', 'au', 'rocktype'])

    # estimate
    if op == "estimate":
        print("STORED MODEL ESTIMATE...", flush=True)
        values = sp.estimation_stored_model(
            path,
            np.float32(locations[['X', 'Y', 'Z']].values),
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(locations, columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/estimation_stored_model_' + suffix + '.csv', index=False)
        print("DONE", flush=True)

    # loo
    if op == "loo":
        print("STORED MODEL LOO...", flush=True)
        values = sp.loo_stored_model(
            path,
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(samples[['X', 'Y', 'Z']], columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/loo_stored_model_' + suffix + '.csv', index=False)
        print("DONE", flush=True)

    # kfold
    if op == "kfold":
        print("STORED MODEL KFOLD...", flush=True)
        values = sp.kfold_stored_model(
            path,
            10, 206936,
            pperc
        )
        print("STORE...", flush=True)
        df = pd.DataFrame(samples[['X', 'Y', 'Z']], columns=['X', 'Y', 'Z'])
        df['py_cu'] = np.nanmean(values, axis=1)
        df.to_csv('./testdata/output/kfold_stored_model_' + suffix + '.csv', index=False)
        print("DONE", flush=True)


def testing_exec_flow(create_data_base=True):
    it = time.time()
    if create_data_base:
        creation()
    print(f"time elapsed:{time.time() - it}")
    it = time.time()
    stored_model("idw", "./testdata/output/create_test_esi_idw.db", "estimate")
    print(f"time elapsed:{time.time() - it}")


if __name__ == '__main__':
    testing_exec_flow(False)

# creation()
# stored_model("idw", "./testdata/output/create_test_esi_idw.db", "estimate")
# stored_model("kriging", "./testdata/output/create_test_esi_kriging.db", "estimate")

# stored_model("idw", "./testdata/output/test_esi_idw.db", "estimate")
# stored_model("idw", "./testdata/output/test_esi_idw.db", "loo")
# stored_model("idw", "./testdata/output/test_esi_idw.db", "kfold")
#
# esi_idw("estimate")
# esi_idw("loo")
# esi_idw("kfold")
#
# stored_model("idw", "./testdata/output/test_esi_idw.db", "estimate")
# stored_model("idw", "./testdata/output/test_esi_idw.db", "loo")
# stored_model("idw", "./testdata/output/test_esi_idw.db", "kfold")
#
# kriging("estimate")
# kriging("loo")
# kriging("kfold")
#
# stored_model("kriging", "./testdata/output/test_esi_kriging.db", "estimate")
# stored_model("kriging", "./testdata/output/test_esi_kriging.db", "loo")
# stored_model("kriging", "./testdata/output/test_esi_kriging.db", "kfold")
