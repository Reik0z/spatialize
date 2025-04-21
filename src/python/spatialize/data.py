import importlib.resources as rs
import os, json
import numpy as np
import pandas as pd

from spatialize.gs.esi import ESIResult
from spatialize.gs.ess import ESSResult
from spatialize.resources import data


def load_result(result_dir_path):
    pass


def save_result(result_dir_path, result):
    def ensure_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)
            return False
        else:
            return True

    already_exists = ensure_directory(result_dir_path)

    if not already_exists:
        meta_data = {}
    else:
        try:
            meta_data = json.load(open(os.path.join(result_dir_path, 'metadata.json')))
        except FileNotFoundError:
            meta_data = {}

    if isinstance(result, ESSResult):
        meta_data["main_result"]: "simulation"
        est_result = result.esi_result

    else:
        if not(already_exists and meta_data["main_result"] == "simulation"):
            meta_data["main_result"]: "estimation"
        est_result = result

    meta_data.update({
        "estimation": "estimation.csv",
        "griddata": est_result.griddata,
        "original_shape": est_result.original_shape,
        "xi": "locations.csv"
    })

    # save the estimation
    fn = os.path.join(result_dir_path, meta_data['estimation'])
    columns = ["estimation"]
    pd.DataFrame(est_result.estimation()).to_csv(fn, index=False, header=columns)

    # save the xi locations
    fn = os.path.join(result_dir_path, meta_data['xi'])
    columns = ["x", "y"]
    pd.DataFrame(est_result._xi).to_csv(fn, index=False, header=columns)

    if isinstance(est_result, ESIResult):
        # save the esi_samples
        meta_data['esi_samples'] = "esi_samples.csv"
        fn = os.path.join(result_dir_path, meta_data['esi_samples'])
        columns = [f"es{i}" for i in range(est_result.esi_samples(raw=True).shape[1])]
        pd.DataFrame(est_result.esi_samples(raw=True)).to_csv(fn, index=False, header=columns)
        meta_data['n_esi_samples'] = len(columns)

    if isinstance(result, ESSResult):
        # save the simulations
        fn = os.path.join(result_dir_path, str(result) + ".csv")
        columns = [f"sim{i}" for i in range(result.scenarios().shape[1])]
        pd.DataFrame(est_result.esi_samples(raw=True)).to_csv(fn, index=False, header=columns)
        if not "simulations" in meta_data:
            meta_data["simulations"] = [fn]
        else:
            if fn not in meta_data["simulations"]:
                meta_data["simulations"].append(fn)

    # save the metadata file
    meta_data_fn = os.path.join(result_dir_path, "metadata.json")
    with open(meta_data_fn, "w") as outfile:
        json.dump(meta_data, outfile, indent=4)


# Toy data sets included in the library
def load_drill_holes_andes_2D():
    path = os.path.join(str(rs.files(data)), "dc1_input_data.csv")
    input_samples = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), "dc1_output_grid.dat")
    with open(path, 'r') as grid_data:
        lines = grid_data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    output_locations = pd.DataFrame(aux, columns=['x', 'y', 'z'])

    path = os.path.join(str(rs.files(data)), "dc1_ok_kriging_example.csv")
    ok_kriging_example = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), "dc1_vario_cu_omni.csv")
    omi_exp_variogram_example = pd.read_csv(path)

    return input_samples, output_locations, ok_kriging_example, omi_exp_variogram_example


def load_drill_holes_andes_3D():
    path = os.path.join(str(rs.files(data)), "dc2_output_box.csv")
    output_locations = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), "dc2_input_muestras.dat")
    with open(path, 'r') as in_data:
        lines = in_data.readlines()
        lines = [l.strip().split() for l in lines[8:]]
        aux = np.float32(lines)
    input_samples = pd.DataFrame(aux, columns=['x', 'y', 'z', 'cu', 'au', 'rocktype'])

    return input_samples, output_locations


def load_simulated_anisotropic_data():
    prefix = "sim_data_geom_anis_nugg0"
    path = os.path.join(str(rs.files(data)), prefix + ".csv")
    ground_truth = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_1perc.csv")
    input_samples_1perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_5perc.csv")
    input_samples_5perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_reduced.csv")
    input_samples_reduced = pd.read_csv(path)

    # ordinary kriging example
    prefix = "sim_kriging_geom_anis_nugg0"
    path = os.path.join(str(rs.files(data)), prefix + ".1_1perc.csv")
    kriging_1perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_5perc.csv")
    kriging_5perc = pd.read_csv(path)

    path = os.path.join(str(rs.files(data)), prefix + ".1_reduced.csv")
    kriging_reduced = pd.read_csv(path)

    return ((input_samples_1perc, kriging_1perc),
            (input_samples_5perc, kriging_5perc),
            (input_samples_reduced, kriging_reduced),
            ground_truth)
