'''
WERL class containing methods to implement various adaptive sampling
methods within the context of WE.

Example Return
--------------
To split: 50 total: 11 being split 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0]
To merge: 50 total: 11 being merged 
[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [40], [], [], [36, 21, 23, 26, 24, 22, 29, 25], [], [], [20], [], [19], [], [], []]
'''

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min

from .werl_base import WERL
#from werl.werl_base import WERL
import os

class REAP(WERL):
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    Using the REAP algorithm from the Shukla group.
    '''
    
    def select_least_counts(self, n_select):
        '''
        Select new starting clusters.
        Returns indices of n_select clusters with least counts and closest points to said clusters.
        n_select allows for a sub-selection of clusters, this way the worst case is LCAS-like.
        '''
        # Which clusters contain lowest populations
        #least_counts = np.asarray(self.counts[::-1][:])[:, 0]
        #least_counts = np.asarray(self.counts[::-1][:self.n_clusters])[:, 0]
        least_counts = np.asarray(self.counts[::-1][:n_select])[:, 0]
        
        closest, _ = pairwise_distances_argmin_min(self.centers, self.pcoords)
        return least_counts, self.pcoords[closest[least_counts]]

    def compute_structure_reward(self, selected_points, weights):
        '''
        Computes the reward for each structure and returns it as an array.
        '''
        #print("Selected Points: ", selected_points)
        mu = self.pcoords.mean(axis=0)
        sigma = self.pcoords.std(axis=0)
        #selected_points = selected_points[:, :2]  # Drop third dimension

        # Shape is (selected_points.shape[0],)
        # print(weights * np.abs(selected_points - mu) / sigma)
        # print((weights * np.abs(selected_points - mu) / sigma).shape)
        # print((weights * np.abs(selected_points - mu) / sigma).sum(axis=1))
        reward = (weights * np.abs(selected_points - mu) / sigma)
        # make sure 2d array sum axis indexable
        #reward = np.atleast_2d(reward)

        # print("Reward shape: ", reward.shape)
        # print("Reward: ", reward)
        # print("Reward SUM: ", reward.sum(axis=1))
        return reward.sum(axis=1)

    def compute_cumulative_reward(self, selected_points, weights):
        '''
        Returns the cumulative reward for current weights and a callable to the 
        cumulative reward function (necessary to finetune weights).
        '''

        def rewards_function(w):
            r = self.compute_structure_reward(selected_points, w)
            R = r.sum()
            return R

        R = rewards_function(weights)

        return R, rewards_function

    def tune_weights(self, rewards_function, weights, delta=0.02):
        '''
        Defines constraints for optimization and maximizes rewards function. 
        Returns OptimizeResult object.

        Parameters
        ----------
        delta : float, default 0.02
            Upper boundary for learning step.
        '''
        weights_prev = weights

        # Create constraints
        constraints = []

        # Inequality constraints (fun(x, *args) >= 0)
        constraints.append({
            'type': 'ineq',
            'fun': lambda weights, weights_prev, delta: delta - np.abs((weights_prev - weights)),
            'jac': lambda weights, weights_prev, delta: np.diagflat(np.sign(weights_prev - weights)),
            'args': (weights_prev, delta),
        })

        # This constraint makes the weights be always positive
        constraints.append({
            'type': 'ineq',
            'fun': lambda weights: weights,
            'jac': lambda weights: np.eye(weights.shape[0]),
        })

        # Equality constraints (fun(x, *args) = 0)
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: weights.sum() - 1,
            'jac': lambda weights: np.ones(weights.shape[0]),
        })

        #print(lambda x: -rewards_function(x))
        #print(weights_prev)
        results = minimize(lambda x: -rewards_function(x), weights_prev, method='SLSQP', constraints=constraints)

        return results

    def reap(self, op_weights_file='op_weights.txt', n_select=None, delta=0.02):
        '''
        WE version of the REinforcement learning based Adaptive SamPling (REAP) algorithm.
        
        Parameters
        ----------
        op_weights_file : str
            Path to order parameter weights file.
            Weights for each order parameter / pcoord dimension used.
            By default will initialize a new file called 'op_weights.txt'.
            New weights are saved to the end of this file per iteration.
            The bottom of the file is appended with new optimized weights and 
            these are used for the next iteration.
        n_select : int
            Number of least-count candidates selected per epoch.
        delta : float
            Upper boundary for learning step, default 0.02.

        **kwargs: used to specify model hyperparameters. Must include following keys:
            n_agents (int): number of agents. --> In this implementation this is forced to be 1
                TODO: note that the 'multi agents' of MA-REAP is better since the initial states are more diverse,
                      I don't think it'll be really worth it here since the op weights need to be tracked for each
                      'agent', and the equal clusters won't give consistent clusters each time. 

        Returns
        -------
        self.to_split, self.to_merge : lists
            List of split merge decisions.
        '''
        # Step 1: define some hyperparameters and initialize arrays
        n_agents = 1  # Forced
        
        # Number of least-count candidates selected per epoch
        if n_select is None:
            # default to half of clusters if nothing is provided
            n_select = int(self.n_clusters / 2)
        
        # self.pcoords.shape[1] (n_pcoord_dims = n_columns of the pcoords array)
        n_features = self.pcoords.shape[1]  # Number of variables in OP space (2d pcoord for now)

        # Step 2: set initial OP weights (not WE traj weights)
        # only do this if input weights are None
        #if weights is None:
        
        # if the file already exists, grab last row
        if os.path.isfile(op_weights_file) is True:
            all_iter_weights = np.atleast_2d(np.loadtxt(op_weights_file))
            weights = [all_iter_weights[-1,:]]
        # otherwise set weights as equal and save new op_weight file
        else:
            all_iter_weights = weights = [np.ones((n_features)) / n_features for _ in range(n_agents)]
            np.savetxt(op_weights_file, weights)

        # WERL: cluster, compute rewards, tune weights, return split/merge decision lists
        # REAP: Steps 4-9: cluster, compute rewards, tune weights, run new simulations, and repeat

        # Logs
        least_counts_points_log = [[] for _ in range(n_agents)]
        cumulative_reward_log = [[] for _ in range(n_agents)]
        weights_log = [[] for _ in range(n_agents)]
        individual_rewards_log = [[] for _ in range(n_agents)]

        # TODO: the agents are kinda like different renditions of the REAP workflow
        #       with subsets of trajectories, I feel like this could work well
        #       in the context of basis states, so each basis state would get a REAP agent?
        #       Or each group of trajectories from the same parent would get it's own REAP agent?
        #       for now not using "multi-agent" (just using 1).
        #       If I were to switch to using multiple sets of traj data from WE bstates/parents,
        #       I would also need to parse and update the pcoords array, 
        #       equivalent to the trajectories array here.
        for i in range(n_agents):

            #clusters = clustering(self.pcoords, self.n_clusters, max_frames=max_frames, b=b, gamma=gamma, d=d)
            self._clustering()

            # if less than the requested amount of clusters was generated
            # then go with zero array return, currently useful for w_init
            # basically, if the clustering fails, do no split/merge operations
            if len(self.counts) < self.n_clusters:
                return self.to_split, self.to_merge

            # grab a n_select subset of the least count clusters to do reward sorting
            least_counts_clusters, least_counts_points = self.select_least_counts(n_select)
            #print("LC_CLUSTERS: ", least_counts_clusters)
            #print("LC_CLUSTERS_POINTS: ", least_counts_points)

            least_counts_points_log[i].append(least_counts_points)

            R, reward_fun = self.compute_cumulative_reward(least_counts_points, weights[i])

            # appending negative since minimizing the negative (really maximizing)
            cumulative_reward_log[i].append(-R)

            optimization_results = self.tune_weights(reward_fun, weights[i], delta=delta)
            # Check if optimization worked
            if optimization_results.success:
                print(optimization_results)
                pass
            else:
                print("ERROR: CHECK OPTIMIZATION RESULTS")
                break

            # get and save out new weights
            weights[i] = optimization_results.x
            weights_log[i].append(weights[i])
            # vstack to put new weights on bottom
            np.savetxt(op_weights_file, np.vstack((all_iter_weights, weights[i])))

            rewards = self.compute_structure_reward(least_counts_points, weights[i])

            individual_rewards_log[i].append(rewards)

        ### print results ###
        #print("pcoords:", self.pcoords)
        print("LC_CLUSTERS: ", least_counts_clusters)
        print(f"pcoord shape: {self.pcoords.shape}")
        print("OP weight log:", weights_log)
        print("LC point log:", least_counts_points_log)
        print("LC point log shape:", np.array(least_counts_points_log).shape)
        print("Cumulative reward log:", cumulative_reward_log)
        print("Individual reward log:", individual_rewards_log)

        # TODO: I have the individual rewards for each walker
        #       I can now split the highest reward contributers
        #       also using the LC clusters?
        #       How to determine the ideal split / merge amount?
        #       maybe based on number of walkers, % of them? 20% or 30%?
        #       REAP just uses a user-parameter, e.g. 50
        #
        #       For now, using a user-parameter that defaults to n_clusters is not provided
        #       This kinda works since n_clusters is based on heuristic from dataset size (n_walkers)

        # TODO: put this into base class as a sorting method
        # update self.counts array from least count cluster labels to highest reward labels
        # Sort least_count_clusters and rewards in descending order based on the values in rewards
        sorted_indices = np.argsort(rewards)
        least_counts_clusters = least_counts_clusters[sorted_indices]
        print("SORTED LCC: ", least_counts_clusters)
        rewards = rewards[sorted_indices]
        print("SORTED REWARDS", rewards)

        # make a full length least_counts_clusters since using a n_select subset
        # replace the end of the self.counts list of clusters with sorted n_select subset of clusters
        #print(self.counts[:,0])
        self.counts[:,0][-len(least_counts_clusters):] = least_counts_clusters
        #print(self.counts[:,0])

        # Insert least_count_clusters as a new column before the first column of cluster_counts
        # replaces the previous column sorting
        # NOTE: using the above replacement step instead
        #self.counts = np.insert(self.counts, 0, least_counts_clusters, axis=1)

        #print("SORTED COUNTS:", self.counts)

        # run split merge methods, updating split/merge lists
        self.do_splitting()
        self.do_merging()

        # return finalized split and merge lists
        return self.to_split, self.to_merge

if __name__ == "__main__":
    # test data (1D array of 1D ODLD endpoints)
    # pcoords = np.loadtxt('mm_potentials/pcoords.txt')
    # weights = np.loadtxt('mm_potentials/weights.txt')

    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    pcoords = np.load('pcoords_full.npy')
    print(f"pcoord full shape: {pcoords.shape}")
    weights = np.ones(80) / 80

    # cut to n walkers for testing
    # pcoords = pcoords[40:,:,:]
    # weights = np.ones(40) / 40

    # test init data
    #pcoords = np.array([9.5] * 50).reshape(-1,1)
    #pcoords = np.array([np.random.uniform(5, 10, 50)]).reshape(-1,1)
    #weights = np.array([0.02] * 50)

    # REAP test
    werl = REAP(pcoords, weights)
    print("N_CLUSTERS: ", werl.n_clusters)
    split, merge = werl.reap()
    print("CLUSTER COUNTS: ", werl.counts)
    print("CLUSTER COUNTS: ", werl.counts.shape)
    print("LABELS", werl.labels)
    
    print("SPLIT: ", split, "\nMERGE: ", merge)
    print("SPLIT INDICES: ", [i for i, e in enumerate(split) if e != 0])
    print("split sum: ", np.sum(split))
    print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))
    #print("counts: ", werl.counts)

 