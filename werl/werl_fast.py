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
        '''
        Get the scaled values for the directed component of the reward function.

        minimize_val: Boolean for deciding whether the pcoord is to be minimized or maximized
        Example: The target for the 1D ODLD is a smaller distance, so here we are minimizing

        Returns a tuple (reward, cluster id, centroid pcoord id)
        '''
        closest, _ = pairwise_distances_argmin_min(self.centers, self.pcoords)
        state_vals = []
        for close in closest:
            centroid_val = self.pcoords[close]
            state_vals.append(centroid_val) # Get the centroids for each state
        
        # Scaling values - requires the program to know if the value should be minimized or maximized
        # These values are the directed component of the reward for each state
        # Scales from 0 (closest to target) to 1 (furthest from target)
        state_min, state_max = np.min(state_vals), np.max(state_vals)

        if minimize_val:
            scaled_rewards = [(state_max-val[0])/(state_max-state_min) for val in state_vals]
        else:
            scaled_rewards = [(val[0]-state_min)/(state_max-state_min) for val in state_vals]

        ## TODO: consider grabbing only the least counts clusters and returning the cluster ID and scaled value
        # In this case, the scaling would proably be done only with the least counts clusters

        ## TODO: consider using a heuristic similar to TSLC for adapting cluster counts over time

        # Traceback to the centroid and label of the cluster (state)
        state_reps = []
        for i in range(len(closest)):
            centroid_val = self.pcoords[closest[i]]
            centroid = list(self.pcoords).index(centroid_val)
            label = self.labels[centroid]
            state_reps.append((float(scaled_rewards[i]), label, centroid))
        
        return state_reps 
    
    def undirected_reward(self):
        '''
        Get the scaled values for the undirected component of the reward function.

        Returns a tuple (reward, cluster id)
        '''

        ## Get min and max instances across all states, as well as each state's instances
        ## TODO: What to do in the case of a tie? 

        clust_and_counts = np.unique(self.labels, return_counts=True)
        counts = [c for c in clust_and_counts[1]]
        min_count, max_count = np.min(counts), np.max(counts)

        ## Scaling values - need to put on same scale as the directed compounent
        ## Scales from 0 (most sampled) to 1 (least sampled)
        scaled_rewards = [(max_count-c)/(max_count-min_count) for c in counts]

        state_reps = []
        for i, reward in enumerate(scaled_rewards):
            state_reps.append((reward, clust_and_counts[0][i]))

        return state_reps

    def compute_reward(self, directed, undirected, alpha=1):
        '''
        Computes the reward for each state using scaled directed (phi) and undirected (psi) components
        R(i) = phi(i) + alpha * psi(i)

        alpha: float for scaling the value of undirected reward component
        '''

        dir_rewards = [r[0] for r in directed]
        undir_rewards = [r[0] for r in undirected]

        rewards = []
        for i in range(len(dir_rewards)):
            rewards.append(dir_rewards[i] + alpha*undir_rewards[i])

        return rewards
        
    def fast(self):
        
        ## Kmeans clustering
        self._clustering()
        
        ## Get rewards
        dir_state_rewards = self.directed_reward()
        undir_state_rewards = self.undirected_reward()

        ## Calculate overall reward per state
        rewards = self.compute_reward(dir_state_rewards, undir_state_rewards)
        print('TOT REWARD:', rewards)

        # if less than the requested amount of clusters was generated
        # then go with zero array return, currently useful for w_init
        # basically, if the clustering fails, do no split/merge operations
        if len(self.counts) < self.n_clusters:
            return self.to_split, self.to_merge

        ## First, re-sort self.counts by the cluster id to link rewards to cluster id
        reordered_counts = []
        for i in range(len(self.counts)):
            for j in range(len(self.counts)):
                if self.counts[j][0] == i:
                    reordered_counts.append(self.counts[j])
        self.counts = np.array(reordered_counts) # Update self.counts to be sorted by cluster id

        ## Next, sort self.counts again by reward
        sorted_indices = np.argsort(rewards)
        sorted_counts = []
        for ind in sorted_indices:
            sorted_counts.append(self.counts[ind])

        self.counts = np.array(sorted_counts)

        # run split merge methods, updating split/merge lists
        self.do_splitting()
        self.do_merging()

        ## TODO: Currently, this will split n_split walkers, starting from the highest reward cluster, 
        ##       but what we really need is for the splitting to occur proportionally to the reward.
        ##       This means that there should be some number of splits from al (or most) clusters



        import matplotlib.pyplot as plt
        colors = ['red', 'green']
        color_points = []
        for i in range(len(self.to_split)):
            color_points.append(colors[self.to_split[i]])

        plt.figure()
        plt.scatter(self.labels, self.pcoords, c=color_points, alpha=0.5)
        plt.show()


## Testing ##

if __name__ == "__main__":
    #pcoords = np.load('mm_potentials/pcoords_full.npy')
    #print(f"pcoord full shape: {pcoords.shape}")
    #pcoords = pcoords[40:,:,:]
    #pcoords = np.ones(40) * 9.5
    pcoords = np.random.uniform(8, 10, size=40) # Generate random initial states
    weights = np.ones(40) / 40
    werl = FAST(pcoords, weights)
    fast = werl.fast()


