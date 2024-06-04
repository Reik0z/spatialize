import numpy as np
import pandas as pd

from scipy.spatial import KDTree
import random


class VoronoiTree:
    def __init__(self, samples=None, locations=None, lamda=0, seed=None):
        self.samples = samples
        self.locations = locations
        self.lamda = lamda
        self.seed = seed
        self.voronoi_partition()

    def N_nuclei(self):
        n = len(self.samples)
        rng = np.random.default_rng(self.seed)
        N = n + 1
        while N > n or N < 1:
            N = rng.poisson(self.lamda)

        random_n = random.sample(self.samples.index.values.tolist(), N)
        nuclei = self.samples.loc[random_n, ['X', 'Y']]
        X = nuclei[['X', 'Y']].values
        tree = KDTree(X)

        return nuclei, tree

    def voronoi_partition(self):
        nuclei, tree = self.N_nuclei()

        x_s = self.samples[['X', 'Y']].values
        ds, inds = tree.query(x_s, 1)  #finds the nearest nucleus for each sample

        x_l = self.locations[['X', 'Y']].values
        ds, inds_l = tree.query(x_l, 1)  #finds the nearest nucleus for each location of the grid

        self.nuclei = nuclei.reset_index(drop=True)
        self.loc_indexes = inds_l
        self.sample_indexes = inds


class VoronoiForest:
    def __init__(self, samples=None, locations=None, size=1, l=0):
        self.samples = samples
        self.locations = locations
        self.size = size
        self.lamda = l

        ss = np.random.SeedSequence()
        self.seeds = ss.spawn(self.size)
        self.trees = [VoronoiTree(self.samples, self.locations, self.lamda, s) for s in self.seeds]

    def kn(self):
        sizes = [[self.samples[[nuc == nucleus for nuc in tree.sample_indexes]].shape[0] for nucleus in
                  tree.nuclei.index.values] for tree in self.trees]  #number of neighbors per nucleus per tree

        average_sizes = [sum(s) / len(s) for s in sizes]  #average neighborhood size per tree

        average_size = sum(average_sizes) / len(average_sizes)  #average neighborhood size in the forest
        return average_size
