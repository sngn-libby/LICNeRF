import torch
import math
from scipy.stats import poisson

import numpy as np
import pandas as pd

# k
# f: y_hat -> distribution model -> z_hat (selecting 'k')
# LUT: z_hat -> LUT -> mu, sigma

# transferring

class LookUpTable():
    def __init__(self, path, dist, k, n=4, **kwargs):
        self.dist = dist # distribution of z_hat
        self.k = k # number of resolution per index

        self.mean_lut = self._build_memory(self.k, n)
        self.sigma_lut = self._build_memory(self.k, n)

        x = pd.read_csv(path, sep=',')
        x = x.values().reshape(-1).tolist()
        z = self._sampling_var(x, self.dist, k)

    def _build_memory(self, k, n=4):
        return torch.Tensor(np.ndarray([k] * n))

    # def _build_distribution(self, x, lmbda=20) -> list:
        # To select variables
        # return poisson(lmbda).pmf(x)

    def _sampling_var(self, x, dist, k):
        if self.dist is None:
            return None
        # Major Implementation
        # picking k pieces of variables from dist (slicing with equal distance)








class LookUpTable3x3(LookUpTable):
    def __init__(self, k, n=4, **kwargs):
        super().__init__(k, n, **kwargs)


# def poisson_dist(n, lmbda=20):
#     return (lmbda ** n) * math.exp(-lmbda) / math.factorial(n)
