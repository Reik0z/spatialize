import multiprocessing
import pandas as pd
import numpy as np

from dask.distributed import Client
from collections import deque

from .voronoi import VoronoiForest
from .interpolations import idw_anisotropic_interpolation, optim_anisotropic_idw, idw_interpolation
from . import logging

pd.options.mode.chained_assignment = None


class Partition:
    def __init__(self, tree, samples):
        self.data = [samples[[nuc == nucleus for nuc in tree.sample_indexes]] for nucleus in
                     tree.nuclei.index.values]  #Neighbors on each nucleus
        self.leaf_indices = tree.sample_indexes  #confirmar


class PartitionOptimize:
    def __init__(self, tree, samples, value_col='grade'):
        self.data = [samples[[nuc == nucleus for nuc in tree.sample_indexes]] for nucleus in
                     tree.nuclei.index.values]  #Neighbors on each nucleus
        self.params = [optim_anisotropic_idw(neighbors, value_col=value_col) for neighbors in
                       self.data]  #Params for each nucleus
        self.leaf_indices = tree.sample_indexes  #confirmar


class EnsembleIDW:
    def __init__(self, size, alpha, samples, locations, value_col='grade', callback=lambda x: x):
        self.callback = callback
        #self.size = size
        #self.alpha = alpha
        self.samples = samples
        self.locations = locations
        lamda = alpha * 40
        assert lamda > 1
        self.value_col = value_col

        self.callback(logging.logger.info('creating voronoi forest ...'))
        self.forest = VoronoiForest(samples, locations, size, lamda)

        self.callback(logging.logger.info('creating partitions ...'))
        self.ensemble = [Partition(tree, samples) for tree in self.forest.trees]

    def predict(self, mean=True, percentile=50, exp_dist=1):
        self.callback(logging.logger.info('creating predictions ...'))

        total = len(self.locations.index) * len(self.forest.trees)
        workers = multiprocessing.cpu_count()
        self.callback(logging.logger.debug(f"num of cores found: {workers}"))

        msg = f"---> total interpolations ({len(self.locations.index)} [locations] x {len(self.forest.trees)} [partitions]): {total}"
        self.callback(logging.logger.debug(msg))

        self.callback(logging.progress.init(total, workers))

        points = self.locations[['X', 'Y']].values

        def compute_predictions(locations):
            preds = self.locations[self.locations.index.isin(locations)]
            preds.loc[:, "pred"] = -99
            trees = deque(list(enumerate(self.forest.trees)))
            for location in locations:
                point = points[location]
                for tree_idx, tree in trees:
                    nucleus_index = tree.loc_indexes[location]
                    neighbors = self.ensemble[tree_idx].data[nucleus_index]
                    if not neighbors.empty:
                        pred = idw_interpolation(point, neighbors, exp_dist, value_col=self.value_col)
                        try:
                            preds.loc[location, "pred"] = pred
                        except KeyError:
                            pass
                        self.callback(logging.progress.inform())
            return preds

        client = Client(n_workers=workers, threads_per_worker=2)
        client.cluster.scale(workers)

        locations = np.split(self.locations.index, workers)

        futures = client.map(compute_predictions, locations)
        results = client.gather(futures)
        computations = pd.concat(results).sort_index()

        self.callback(logging.progress.stop())

        return Reduction(computations[['pred']].values, mean, percentile)

    def predict_old(self, mean=True, percentile=50, exp_dist=1):
        print('Creating Predictions...')
        predictions = []

        total = len(self.locations.index) * len(self.forest.trees)
        print(
            f"---> total interpolations ({len(self.locations.index)} [locations] x {len(self.forest.trees)} [partitions]): {total}")

        self.callback(logging.progress.init(total))

        points = self.locations[['X', 'Y']].values
        c = 0
        trees = deque(list(enumerate(self.forest.trees)))
        for location in deque(self.locations.index):
            point = points[location]
            values = []
            for tree_idx, tree in trees:
                c += 1
                nucleus_index = tree.loc_indexes[location]
                neighbors = self.ensemble[tree_idx].data[nucleus_index]
                if not neighbors.empty:
                    pred = idw_interpolation(point, neighbors, exp_dist, value_col=self.value_col)
                    values.append(pred)
                    self.callback(logging.progress.inform())
            predictions.append(np.array(values) if values else np.array([-99.]))
        self.callback(logging.progress.stop())
        return Reduction(predictions, mean, percentile)

    def cross_validation(self, mean=True, percentile=50, exp_dist=1):
        predictions = []
        for point_idx in range(len(self.samples.index)):
            values = []
            for tree_idx, tree in enumerate(self.forest.trees):
                leaf_idx = self.ensemble[tree_idx].leaf_indices[point_idx]
                neighbors = self.ensemble[tree_idx].data[leaf_idx].drop(index=point_idx)
                point = self.samples.loc[point_idx][['X', 'Y']].values
                pred = idw_interpolation(point, neighbors, exp_dist, value_col=self.value_col)
                values.append(pred)
            predictions.append(np.array(values) if values else np.array([-99.]))
        return Reduction(predictions, mean, percentile)


class EnsembleAnisotropicIDWOptimize:
    def __init__(self, size, alpha, samples, locations, value_col='grade'):
        #self.size = size
        #self.alpha = alpha
        self.samples = samples
        self.locations = locations
        lamda = alpha * 40
        assert lamda > 1
        self.value_col = value_col

        print('Creating Voronoi Forest...')
        self.forest = VoronoiForest(self.samples, self.locations, size, lamda)
        print('Creating Partitions...')
        self.ensemble = [PartitionOptimize(tree, self.samples, value_col=self.value_col) for tree in self.forest.trees]

    def predict_all_data(self, mean=True, percentile=50):
        print('Creating Predictions...')
        predictions = []
        est_params = {'exponent': [], 'af': [], 'azimuth': []}

        points = self.locations[['X', 'Y']].values
        for location in self.locations.index:
            point = points[location]
            values = []
            exps = []
            afs = []
            azimuths = []
            for tree_idx, tree in enumerate(self.forest.trees):
                nucleus_index = tree.loc_indexes[location]
                #if nucleus_index is not None:
                neighbors = self.ensemble[tree_idx].data[nucleus_index]
                if not neighbors.empty:
                    exp, af, azm = self.ensemble[tree_idx].params[nucleus_index]
                    pred = idw_anisotropic_interpolation(point, neighbors, exp_dist=exp, anis_factor=af, azimuth=azm,
                                                         value_col=self.value_col)
                    values.append(pred)
                    exps.append(exp)
                    afs.append(af)
                    azimuths.append(azm)

            predictions.append(np.array(values) if values else np.array([-99.]))
            est_params['exponent'].append(np.array(exps) if exps else np.array([-99.]))
            est_params['af'].append(np.array(afs) if afs else np.array([-99.]))
            est_params['azimuth'].append(np.array(azimuths) if azimuths else np.array([-99.]))

        return ReductionAnisotropicIDW(predictions, est_params, mean, percentile)

    def cross_validation_all_data(self, mean=True, percentile=50, prev_params=True):
        predictions = []
        est_params = {'exponent': [], 'af': [], 'azimuth': []}

        for point_idx in range(len(self.samples.index)):
            values = []
            exps = []
            afs = []
            azimuths = []
            if prev_params:
                for tree_idx, tree in enumerate(self.forest.trees):
                    leaf_idx = self.ensemble[tree_idx].leaf_indices[point_idx]
                    neighbors = self.ensemble[tree_idx].data[leaf_idx].drop(index=point_idx)
                    point = self.samples.loc[point_idx][['X', 'Y']].values
                    if not neighbors.empty:
                        optim_idw_params = self.ensemble[tree_idx].params[leaf_idx]
                        exp, af, azm = optim_idw_params
                        pred = idw_anisotropic_interpolation(point, neighbors, exp_dist=exp, anis_factor=af,
                                                             azimuth=azm, value_col=self.value_col)
                        values.append(pred)
                        exps.append(exp)
                        afs.append(af)
                        azimuths.append(azm)

            else:
                for tree_idx, tree in enumerate(self.forest.trees):
                    leaf_idx = self.ensemble[tree_idx].leaf_indices[point_idx]
                    neighbors = self.ensemble[tree_idx].data[leaf_idx].drop(index=point_idx)
                    point = self.samples.loc[point_idx][['X', 'Y']].values
                    if not neighbors.empty:
                        optim_idw_params = optim_anisotropic_idw(neighbors, value_col=self.value_col)
                        exp, af, azm = optim_idw_params
                        pred = idw_anisotropic_interpolation(point, neighbors, exp_dist=exp, anis_factor=af,
                                                             azimuth=azm, value_col=self.value_col)
                        values.append(pred)
                        exps.append(exp)
                        afs.append(af)
                        azimuths.append(azm)
            predictions.append(np.array(values) if values else np.array([-99.]))
            est_params['exponent'].append(np.array(exps) if exps else np.array([-99.]))
            est_params['af'].append(np.array(afs) if afs else np.array([-99.]))
            est_params['azimuth'].append(np.array(azimuths) if azimuths else np.array([-99.]))
        return ReductionAnisotropicIDW(predictions, est_params, mean, percentile)


class Reduction:
    def __init__(self, values, mean=True, percentile=50):
        if mean:
            self.estimates = np.array([np.mean(p) for p in values])
        else:
            self.estimates = np.array([np.percentile(p, percentile) for p in values])
        self.variances = np.array([np.var(p) for p in values])


class ReductionAnisotropicIDW:
    def __init__(self, values, est_params, mean=True, percentile=50):
        self.full_estimates = values
        self.estimated_params = {}
        if mean:
            self.estimates = np.array([np.mean(p) for p in values])
            self.estimated_params['exponent'] = np.array([np.mean(p) for p in est_params['exponent']])
            self.estimated_params['af'] = np.array([np.mean(p) for p in est_params['af']])
            self.estimated_params['azimuth'] = np.array([self.get_mean_azm(p) for p in est_params['azimuth']])
        else:
            self.estimates = np.array([np.percentile(p, percentile) for p in values])
            # self.estimated_params['exponent'] = np.array([np.percentile(p, percentile) for p in est_params['exponent']])
            # self.estimated_params['af'] = np.array([np.percentile(p, percentile) for p in est_params['af']])
            # self.estimated_params['azimuth'] = np.array([np.percentile(p, percentile) for p in est_params['azimuth']])
        self.variances = np.array([np.var(p) for p in values])

    def get_mean_azm(self, azimuths):
        sum_cos, sum_sin = np.sum(np.cos(azimuths)), np.sum(np.sin(azimuths))
        azm_result = np.arctan2(sum_sin, sum_cos)
        return azm_result
