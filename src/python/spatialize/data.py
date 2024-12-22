import importlib.resources as rs
import os
import numpy as np
import pandas as pd

from spatialize.resources import data


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
