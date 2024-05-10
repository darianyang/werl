# WERL - Bowman FAST algorithm
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min
from werl_base import WERL


class FAST(WERL):

    def compute_reward(cluster):
         pass
        
    def fast(self):
        self._clustering()

        if len(self.counts) < self.n_clusters:
                return self.to_split, self.to_merge
        
        least_counts_clusters, least_counts_points = self.select_least_counts()


## Testing ##

if __name__ == "__main__":
    #pcoords = np.load('mm_potentials/pcoords_full.npy')
    #print(f"pcoord full shape: {pcoords.shape}")
    #pcoords = pcoords[40:,:,:]
    pcoords = np.ones(40) * 9.5
    weights = np.ones(40) / 40

    print(pcoords)
    print(weights)

