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

from werl_base import WERL

class REAP(WERL):
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    '''
    
    def select_least_counts(self):
        '''
        Select new starting clusters.
        Returns indices of n_select clusters with least counts and closest points to said clusters.
        '''
        # Which clusters contain lowest populations
        least_counts = np.asarray(self.counts[::-1][:self.n_clusters])[:, 0]
        closest, _ = pairwise_distances_argmin_min(self.centers, self.pcoords)
        return least_counts, self.pcoords[closest[least_counts]]

    def compute_structure_reward(self, selected_points, weights):
        '''
        Computes the reward for each structure and returns it as an array.
        '''
        mu = self.pcoords.mean(axis=0)
        sigma = self.pcoords.std(axis=0)
        #selected_points = selected_points[:, :2]  # Drop third dimension

        # Shape is (selected_points.shape[0],)
        # print(weights * np.abs(selected_points - mu) / sigma)
        # print((weights * np.abs(selected_points - mu) / sigma).shape)
        # print((weights * np.abs(selected_points - mu) / sigma).sum(axis=1))
        reward = (weights * np.abs(selected_points - mu) / sigma)
        # make sure 2d array sum axis indexable
        reward = np.atleast_2d(reward)
        print("Reward shape: ", reward.shape)
        return reward.sum(axis=1)

    def compute_cumulative_reward(self, weights, selected_points):
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

    def reap(self):
        '''
        WE version of the REinforcement learning based Adaptive SamPling (REAP) algorithm.
        
        Args
        -------------
        **kwargs: used to specify model hyperparameters. Must include following keys:
            n_agents (int): number of agents. --> In this implementation this is forced to be 1
            traj_len (int): length of each trajectory ran.
            delta (float): upper boundary for learning step.
            n_features (int): number of collective variables. (Currently ignored because only two-dimensional systems are used.)
            d (int): dimensionality of the slow manifold (used to compute number of clusters to use). (Should be two for these trials.)
            gamma (float): parameter to determine number of clusters (theoretically in [0.5, 1]).
            b (float): parameter to determine number of clusters.
        Returns
        -------
        '''
        # Step 1: define some hyperparameters and initialize arrays --> To be provided via init_variables
        n_agents = 1  # Forced
        delta = 0.02  # Upper boundary for learning step
        n_features = 2  # Number of variables in OP space (2d pcoord for now)

        # Step 2: set initial OP weights (not WE traj weights)
        weights = [np.ones((n_features)) / n_features for _ in range(n_agents)]

        # WERL: data is already collected from WE iteration
        # TODO: Could use the full trajectory data from WE instead of just the endpoints
        # Step 3: collect some initial data
        # trajectories = [[] for _ in range(n_agents)]
        # trajectories = collect_initial_data(num_spawn * 2, traj_len, potential, initial_positions,
        #                                     trajectories)  # Multiply by 2 to use same initial data as in MA REAP

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

            # TODO: REAP will only select and continue from the least count cluster.
            #       for WERL, I have a few options:
            #           * I can just split the least count cluster only
            #               - but I know from LCAS this isn't super effective
            #           * I can split n_cluster amount of times starting from the LC cluster
            #               - this is the current LCAS setup that works better than just LC split
            #           * I can also optimize the reward, choosing the subset of trajectories that 
            #             would maximize cumulative reward
            #           * I can also skip the LC clusters and just split/merge based on reward,
            #             similar to how REVO does it based on variation
            least_counts_clusters, least_counts_points = self.select_least_counts()
            print("LC CLUSTERS: ", least_counts_clusters)

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

            weights[i] = optimization_results.x

            weights_log[i].append(weights[i])

            rewards = self.compute_structure_reward(least_counts_points, weights[i])

            individual_rewards_log[i].append(rewards)

        ### print results ###
        #print("pcoords:", pcoords)
        print(f"pcoord shape: {pcoords.shape}")
        print("OP weight log:", weights_log)
        print("LC point log:", least_counts_points_log)
        print("LC point log shape:", np.array(least_counts_points_log).shape)
        print("Cumulative reward log:", cumulative_reward_log)
        print("Individual reward log:", individual_rewards_log)

        # TODO: New method for the next steps to be modular with TSLC?
        #       Could also take and use logic from LCAS method
        # TODO: I have the individual rewards for each walker
        #       I can now split the highest reward contributers
        #       also using the LC clusters?
        #       How to determine the ideal split / merge amount?
        #       maybe based on number of walkers, % of them? 20% or 30%?
        #       REAP just uses a user-parameter, e.g. 50
        # TODO: adapting to use the full pcoord array dataset - NEXT:
		#       Need to make a label array and then use this to find 
        #       the identity of the least count walkers after selection
        #       Maybe then use the same split merge logic as LCAS but 
        #       just adjust the selection/indexing to find right seg_id

if __name__ == "__main__":
    # test data (1D array of 1D ODLD endpoints)
    # pcoords = np.loadtxt('mm_potentials/pcoords.txt')
    # weights = np.loadtxt('mm_potentials/weights.txt')

    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    pcoords = np.load('mm_potentials/pcoords_full.npy')
    print(f"pcoord full shape: {pcoords.shape}")

    # test init data
    #pcoords = np.array([9.5] * 50).reshape(-1,1)
    #pcoords = np.array([np.random.uniform(5, 10, 50)]).reshape(-1,1)
    #weights = np.array([0.02] * 50)

    # REAP test
    werl = REAP(pcoords)
    print("N_CLUSTERS: ", werl.n_clusters)
    werl.reap()
    print("CLUSTER COUNTS: ", werl.counts)

    # # test output
    # split = np.loadtxt('split.txt')
    # merge = np.load('merge.npy', allow_pickle=True)
    # print(split, merge)

 