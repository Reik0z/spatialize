from multiprocessing import freeze_support

import pandas as pd
import time
from voronoi.ensemble import EnsembleIDW
from voronoi.logging import AsyncProgressBar


def main():
    trees = 1
    alpha = 0.7

    nugg_case = 'nugg0.1'
    file_sim = f'sim_data_{nugg_case}.csv'
    data_sim = pd.read_csv(file_sim)
    grid = data_sim[['X', 'Y']]

    col_sim = 'sim1'
    case_list = ['5perc']

    s1 = time.time()
    file_samples = f'sim_data_{nugg_case}_{case_list[0]}.csv'
    data_samples = pd.read_csv(file_samples)
    samples = data_samples[['X', 'Y', col_sim]]

    esi = EnsembleIDW(trees, alpha, samples, grid, value_col=col_sim, callback=AsyncProgressBar())
    s2 = time.time()
    result = esi.predict()
    print(f"prediction elapsed time: {time.time() - s2:.2f}s")

    grid_result = grid.copy()
    est_col = 'esi_original_' + col_sim
    grid_result[est_col] = result.estimates

    out_filename = f'esi_voronoi_{nugg_case}_{str(alpha)}_{str(trees)}_{case_list[0]}.csv'
    grid_result.to_csv(out_filename, index=False)
    print(f"total elapsed time: {time.time() - s1:.2f}s")


if __name__ == '__main__':
    freeze_support()
    main()
