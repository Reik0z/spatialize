import importlib.resources as rs
import os
import numpy as np
import pandas as pd

from spatialize.resources import data


def load_drill_holes_andes():
    path = os.path.join(rs.files(data), "dc1_input_data.csv")
    input_samples = pd.read_csv(path)

    path = os.path.join(rs.files(data), "dc1_output_grid.dat")
    with open(path, 'r') as grid_data:
        lines = grid_data.readlines()
        lines = [l.strip().split() for l in lines[5:]]
        aux = np.float32(lines)
    output_locations = pd.DataFrame(aux, columns=['x', 'y', 'z'])

    return input_samples, output_locations


samples, locations = load_drill_holes_andes()

print(samples.head())
print(locations.head())
