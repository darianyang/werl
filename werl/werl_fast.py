# WERL - Bowman FAST algorithm
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min
from werl_base import WERL


class FAST(WERL):
    def select_least_counts(self):
        '''
        Select new starting clusters.
        Returns indices of n_select clusters with least counts and closest points to said clusters.

        N: number of least count clusters to keep from the original self.n_clusters
        ## TODO maybe simply update self.n_clusters in the clustering method
        '''
        # Which clusters contain lowest populations
        print(self.counts)
        least_counts = np.asarray(self.counts[::-1][:self.n_clusters])[:, 0]
        closest, _ = pairwise_distances_argmin_min(self.centers, self.pcoords)
        return least_counts, self.pcoords[closest[least_counts]]    
    
    def directed_reward(self, minimize_val=True):
        # The target for the 1D ODLD is a "short" distance, i.e. low value
        # Range from 0 to 1 after scaling: 0 and 1 are assigned to furthest and closest to target, resp.
        # i refers to state, which here it is each cluster

        # Need to get the state centroids, as this reward component is calculated across states
        closest, _ = pairwise_distances_argmin_min(self.centers, self.pcoords)
        state_vals = []
        for close in closest:
            centroid_val = self.pcoords[close]
            state_vals.append(centroid_val) # Get the centroids for each state
        
        # Scaling values - requires the program to know if the value should be minimized or maximized
        # These values are the directed component of the reward for each state
        state_min, state_max = np.min(state_vals), np.max(state_vals)

        if minimize_val:
            scaled_vals = [(state_max-val)/(state_max-state_min) for val in state_vals]
        else:
            scaled_vals = [(val-state_min)/(state_max-state_min) for val in state_vals]
        print(state_vals)
        print(scaled_vals)

        # Traceback to the centroid and label of the cluster (state)
        state_reps = []
        for i in range(len(closest)):
            centroid_val = self.pcoords[close]
            centroid = list(self.pcoords).index(centroid_val)
            label = self.labels[centroid]
            state_reps.append((float(scaled_vals[i]), centroid, label))
        
        print(state_reps)
        
        pass
    
    def undirected_reward():
        pass

    def compute_reward(cluster, alpha=0.5):
        pass
        
    def fast(self):
        self._clustering()
        
        self.directed_reward()

        #if len(self.counts) < self.n_clusters:
        #        return self.to_split, self.to_merge
        
        #least_counts_clusters, least_counts_points = self.select_least_counts() # Use only the least count clusters?


## Testing ##

if __name__ == "__main__":
    #pcoords = np.load('mm_potentials/pcoords_full.npy')
    #print(f"pcoord full shape: {pcoords.shape}")
    #pcoords = pcoords[40:,:,:]
    #pcoords = np.ones(40) * 9.5
    pcoords = np.random.uniform(9, 10, size=40)
    weights = np.ones(40) / 40
    werl = FAST(pcoords, weights)
    fast = werl.fast()


