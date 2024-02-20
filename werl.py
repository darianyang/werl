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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
#from itertools import combinations
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform

class WERL:
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    '''

    def __init__(self, pcoords, n_clusters=None, b=0.07, gamma=0.6, d=2):
        '''
        Parameters
        ----------
        segments : westpa segments object (TODO)
            Each segment also has a weight attribute.
        pcoords : ndarray
            Last pcoord value of each segment for the current iteration.
        n_clusters : int
            Number of clusters for k-means.
            If self.n_clusters is None, calculating optimal n_clusters using a
            heuristic approximation from Buenfil, Koelle, Meila, ICML (2021).
        b : float, default 1e-4 (REAP) or 0.07 (TSLC)
            Coefficient for n_clusters heuristic.
        gamma : float, default 0.7
            Exponent for n_clusters heuristic (should theoretically be in (0.5, 1)).
        d : float, default 2 
            Intrinsic dimensionality of slow manifold for the system. Used in n_clusters heuristic.
        '''
        # TODO: make sure all of these ndims are compatible with methods
        #       prob make all of them atleast3d?
        # need to reshape 1d arrays from (n) to (n, 1)+ for kmeans fitting
        # this is the case where you have 1 frame per walker and 1 pcoord dim
        if pcoords.ndim == 1:
            self.pcoords = pcoords.reshape(-1, 1)
        # case where you just have 1 frame per walker and 2 pcoord dims
        elif pcoords.ndim == 2:
            self.pcoords = pcoords
        # when using the full 3D pcoord array (walkers, frames, pcoords)
        # need to condense to 2D for km clustering
        elif pcoords.ndim == 3:
            # keep the columns as each pcoord dim but string all frames in rows
            #self.pcoords = pcoords.reshape(-1, pcoords.shape[2])
            # do nothing since the reshape is currently handled in clustering
            self.pcoords = pcoords
        else:
            raise ValueError(f"pcoord input array has {pcoords.ndim} dimensions, must be 1-3.")

        # TODO: need to fix this, so maybe I can first make a label array of the 3d pcoords
        #       then look at the label uniques to find the self.n_segments
        #       or I could just look before reshaping and reshape upon clustering
        #       I need to be able to get the original segment id / index using the label array
        # number of segments/walkers and frames during the iteration
        self.n_segments = self.pcoords.shape[0]
        self.n_frames = self.pcoords.shape[1]

        if n_clusters is None:
            # calculate optimal amount of n_clusters
            #self.n_clusters = np.floor(b * self.n_segments ** (gamma * d)).astype(int)
            self.n_clusters = int(b * (self.n_segments ** (gamma * d)))
        else:
            # otherwise use input n_cluster number
            self.n_clusters = n_clusters

        print("n_clusters: ", self.n_clusters)

        # list of walker positions to split, begin as all zeros (no splitting)
        self.to_split = [0] * self.n_segments
        # list of lists for walker positions to merge, begin as all empty
        self.to_merge = [[] for _ in range(self.n_segments)]

        # list of all possible merge pairs
        #self.merge_pairs = list(combinations([i for i in range(self.n_segments)], 2))

    def _clustering(self):
        '''
        Shared clustering method with k-means using self.n_clusters.

        Updates
        -------
        self.labels : array
            Cluster labels for each data point.
        self.counts : list of tuples
            Count of each cluster label amount.
        '''
        print("PCOORD SHAPE CLUSTER: ", self.pcoords.shape)
        # TODO: universal reshape step here needed before clustering and for REAP?
        #       could also move to init
        # keep the columns as each pcoord dim but string all frames in rows
        self.pcoords = self.pcoords.reshape(-1, self.pcoords.shape[2])

        # kmeans clustering
        km = KMeans(n_clusters=self.n_clusters, random_state=1).fit(self.pcoords)
        self.centers = km.cluster_centers_
        #print(km.labels_)
        self.labels = km.labels_.reshape(self.n_segments, self.n_frames)
        #print(self.labels)

        # count each label amount (using km.labels_ since not reshaped)
        # example output: Counter({0: 21, 1: 20, 2: 4, 3: 3, 4: 2})
        # then with most_common: [(0, 21), (1, 20), (2, 4), (3, 3), (4, 2)]
        self.counts = Counter(km.labels_).most_common()
        print(self.counts)

    def FFT(self, n_clusters, l=2):
        '''
        Fastest First Traversal algorithm for pruning cluster centers
        From the implementation of TLSC by Kleiman and Shukla adapted from Buenfil et al.

        Updates: n_clusters, pcoords (shape)
        '''
        self.pcoords = self.pcoords.reshape(-1, self.pcoords.shape[2])
        n_clusters_prime = int(n_clusters * np.log(n_clusters) + l)
        rng = np.random.default_rng()
        rand_indices = rng.choice(len(self.pcoords), n_clusters_prime, replace=False)
        init_points = self.pcoords[rand_indices]

        # Pick mu1 (init center) at random from pcoords
        mu1 = 0 
        mus = [mu1]
        available_mus = [i for i in range(1, n_clusters_prime)] # All other mus are considered for next center 

        # Pick next center as point with greatest distance from previous point
        # Continue until we prune centers from n_cluster_prime to n_cluster
        D = squareform(pdist(init_points))
        np.fill_diagonal(D, np.inf)

        #### Vectorize highest minimum search ####
        for _ in range(1, n_clusters):
            max_sofar = available_mus[0] # init index for max distance cluster center
            min_dist = np.amin(D[max_sofar, np.asarray(mus)]) 

            for i in available_mus[1:]:
                if np.amin(D[i, np.array(mus)]) > min_dist:
                    max_sofar = i
                    min_dist = np.amin(D[max_sofar, np.array(mus)])
            available_mus.remove(max_sofar)
            mus.append(max_sofar)
        mus = np.asarray(mus)

        return self.pcoords[rand_indices[mus]] # Centroids optimally spread
    
    def tslc_clustering(self, lc=3):
        '''
        Clustering method adapted from the TSLC CLUST method using Fastest First Traversal.

        Args
        -------
        pcoords, pcoord array
        n_clusters, number of clusters - if None, then use the cluster number determined by Buenfil et al. approach
        b, coefficient for n_clusters determination
        gamma, exponents for n_clusters determination
        d, intrinsic dimensionality of slow manifold used for n_clusters determination
        l, constant for n_clusters determination

        Updates
        -------
        self.labels : array
            Cluster labels for each data point.
        self.counts : list of tuples
            Count of each cluster label amount.
        '''

        # First clustering attempt using FFT
        centroids = self.FFT(self.n_clusters)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=1, init=centroids).fit(self.pcoords) # Perform one round of kmeans
        
        while len(np.unique(kmeans.labels_)) != self.n_clusters:
            centroids = self.FFT(self.pcoords, self.n_clusters)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=1, init=centroids).fit(self.pcoords)

        return kmeans

    def tslc_weighting(self, kmeans, N=3, d=2):
        '''
        Weighting method adapted from TLSC weighting algorithm.

        Updates
        -------
        self.labels : array
            Cluster labels for each data point.
        self.counts : list of tuples
            Count of each cluster label amount.
        '''
        clusterid_counts = sorted([(i, np.unique(kmeans.labels_, return_counts=True)[1][i]) for i in range(len(kmeans.cluster_centers_))], key=lambda x: x[1])
        lc_clusters = clusterid_counts[:N]
        Vis = []

        # 4) PCA for points in each lc cluster -> d space
        lc_data_dict = {c[0]:[] for c in lc_clusters}
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] in [c[0] for c in lc_clusters]:
                lc_data_dict[kmeans.labels_[i]].append(list(self.pcoords[i]))

        for key in lc_data_dict.keys():
            points = np.array(lc_data_dict[key])
            print(key, points)
            print()
            pca = PCA(n_components=d)
            pca_trans = pca.fit(points)  # Projection onto d space
            Vi = pca_trans.components_.T  # Must transpose to get desired shape
            Vis.append(Vi)


        # 5) Get Gi matrices of shape (D, K) that contain the gradient of 
        # each CV w.r.t. to the cartesian coordinate of the point
        # Use a representative frame from the N least count clusters
        # The gradient will depend on the system used - i.e. some user defined function is necessary
        # This may be required in the constructor, but I am not sure how this could be passed
        # Currently, I have it implemented as is done in the MA REAP paper (arctan(y/x))
        
        Gis = []
        closest_frames, _ = pairwise_distances_argmin_min(lc_centers, self.pcoords) 
        norm = 0
        for point in closest_frames:
            x, y = self.pcoords[point]
            dx = np.asarray([1, 0, -y / (x ** 2 + y ** 2)])
            dy = np.asarray([0, 1, x / (x ** 2 + y ** 2)])
            Gi = np.vstack([dx, dy])
            norm += np.linalg.norm(Gi, axis=0)
            Gis.append(Gi)

        # 6) Normalize grads
        normed_Gis = []
        for Gi in Gis:
            normed_Gis.append(Gi * len(closest_frames) / norm)


        # 7) Get A - sum of elementwise multiplication (Gi.T * Vi) * (Vi.T * Gi) for each CV (i)
        K = Gis[0].shape[1]  # This is simply the number of CVs
        A = np.zeros((K, K)) # init matrix A (see step 7 of TSLC of algorithm)
        for i in range(len(Gis)):
            Gi, Vi = Gis[i], Vis[i]
            A += (Gi.T @ Vi) @ (Vi.T @ Gi)

        # 8) Compute largest eigenvector of A, w~
        eig_vals, eig_vecs = np.linalg.eig(A)
        largest_eigval = np.argmax(eig_vals)
        largest_eigvec = eig_vecs[largest_eigval]

        # 9) Compute w_k = w~_k ** 2 for k = [1, ..., K] in w~
        # MA-REAP code uses eig_vecs[:, largest_eigval], but I think that may be incorrect
        # since that will use the largest eigenvalue's index of each eigenvector.
        weights = eig_vecs[largest_eigval, :] ** 2 

        return weights

    def tslc(self, N=3):
        '''
        WE version of the Tangent Space Least Adaptive Clustering (TLSC) algorithm.
        
        Input: Differentiable energy E, collective variables theta1 ... thetaK , number of least count clusters to sample from
        N, number of parallel runs M, number of iterations I,timesteps per iteration s, intrinsic dimension d

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

        kmeans = self.tslc_clustering()
        # Get N lc clusters from the cluster id and its associated trajs 
        clusterid_counts = sorted([(i, np.unique(kmeans.labels_, return_counts=True)[1][i]) for i in range(len(kmeans.cluster_centers_))], key=lambda x: x[1])
        lc_clusters = clusterid_counts[:N]

        # Get centers for the N lc_clusters
        lc_centers = [kmeans.cluster_centers_[lc_clusters[i][0]] for i in range(len(lc_clusters))]

        # Find representatives of the clusters closest to their respective cluster center
        rep_frames = []
        closest_frames, dist = pairwise_distances_argmin_min(lc_centers, self.pcoords)

        # Test that the distances, etc are correctly calculated
        # print(clusterid_counts)
        # print(lc_centers)
        # print(closest_frames, dist)
        # print([self.pcoords[closest_frames[i]] for i in range(len(closest_frames))])

        return kmeans, lc_centers, closest_frames 
   
    def lcas(self, n_split=None):
        '''
        ************************************************************************************
        *** TODO: Note LCAS not compatible with full pcoord array yet, just final frames ***
        ************************************************************************************ 
        TODO: Perhaps it would be easiest to include the individual rewards and sort each 
              cluster in LCAS by the reward before splitting / merging
              
              After reading more of the REAP methods, I can prob look at the n clusters being
              considered in terms of the n_split, then from those sort all the structures by reward
              and then do splitting. Then for merging look at the lowest reward in top cluster(s).
              For regular LCAS, have then not sorted by reward, just by least cluster count.
              TODO: should also try it without LCAS, maybe just reward sorting and n_split/n_merge.

        Least Counts Adaptive Sampling. In LCAS, from segment data of the previous iteration, 
        cluster the trajectory endpoint pcoord (aux? TODO) data, then select the next starting 
        states from the smallest, or least count, clusters. The idea being that clusters with 
        fewer members correspond to more sparsely sampled regions of the configuration space, 
        and will tend to be towards the boundaries of the explored region.

        Parameters
        ----------
        n_split : int
            TEMP arg for how many segs to split, later determined from reward funct opt (TODO).
            n_merge will be based on the same value.
            Currently going to use the same value as n_clusters if None.
            TODO: I wonder if it would be useful to be able to split the same walker multiple
                  times, right now limited to once, but the driver code is written so it can be multiple.
        
        Returns
        -------
        self.to_split : list
        self.to_merge : list
        '''        
        # do clustering
        self._clustering()
        #print("shape labels: ", self.labels.shape)

        # if less than the requested amount of clusters was generated
        # then go with zero array return, currently useful for w_init
        # basically, if the clustering fails, do no split/merge operations
        if len(self.counts) < self.n_clusters:
            return self.to_split, self.to_merge
        
        # use n_clusters for n walkers to split if not given
        # TODO: or could just only split LC cluster when n_split is None
        #       this is how the code originally worked from Alex, when I
        #       tested it against splitting up to n_clusters though it 
        #       performed a bit better with more diverse splitting.
        if n_split is None:
            n_split = self.n_clusters

        # split n_split times the lowest count cluster(s)
        splits_remaining = n_split


        # may need to go through multiple cluster labels if count is small
        lc_cluster_counter = 1
        
        while splits_remaining > 0:
            # go through each segment index label and tag to split
            # i here is going to be the walker/segment index
            for i, seg_labels in enumerate(self.labels):
                
                # since labels can be 2D: (n_segments, n_frames)
                # loop, but only the unique elements, so no need to split >1 time
                # if there are multiple LC labels in n_frames of a segment
                # TODO: think about if it would be useful to have multiple splits
                #       for now I think minimal splitting is better
                for frame_label in np.unique(seg_labels):
                    
                    # if the segment frame is in the least count cluster
                    if frame_label == self.counts[-lc_cluster_counter][0]:
                        # increase split counter on this segment by 1
                        self.to_split[i] += 1
                        # note that one of the requested splits is done
                        splits_remaining -= 1
                        # # TODO: test, only splitting one per seg
                        # break

                    # extra precaution
                    if splits_remaining == 0:
                        break

                # extra precaution
                if splits_remaining == 0:
                    break
        
            # once done looping through all cluster labels, if there are still more
            # to split, then do it again using the next lowest populated cluster label count
            # to only split the lowest cluster, don't use the counter
            lc_cluster_counter += 1

        # TODO: can this be consolidated with splitting code?
        # TODO: there is issues with merging where after the main high cluster labels
        #       there is overlap in merging, e.g. [[], [2, 10, 11, 26, 28, 33, 36, 37], [1, 10], ...]
        #       here, 2 gets merged to 1 and then 1 to 2.
        #       I think this is fixed with frozen set checks
        #       Other issue now, need to prevent/check for merging what is already split
        #       I think this is fixed now as well with the disjoint
        #       But still having KeyError issues with removing certain segs
        # merge n_split times the highest count cluster(s)
        merges_remaining = n_split
        
        # may need to go through multiple cluster labels
        lc_cluster_counter = 0
        
        # make sure there isn't redundant merge pairs
        # using set since sets are unordered, unchangable, with no duplicates
        pairs_merged = set()
        
        # keep going until no more merges needed or until no more cluster labels avail
        #while merges_remaining > 0 and lc_cluster_counter < self.n_clusters:
        while merges_remaining > 0:
            
            # go through each segment index label and tag to merge
            for i, segi_labels in enumerate(self.labels):
                
                # TODO: might be slow but for now: using this to check to make sure that when
                #       moving from one seg to another for making merge pairs, ensures that
                #       the first selected of the pair isn't already set to be merged,
                #       keeps all merge operations unique and compatible with WERL driver
                # skip_i_seg needed to move past repeat i values (already in a merge pair)
                skip_i_seg = False
                for pair in pairs_merged:
                    if i in pair:
                        skip_i_seg = True
                        break
                if skip_i_seg:
                    continue
                
                # since labels can be 2D: (n_segments, n_frames)
                # loop, but only the unique elements
                for framei_label in np.unique(segi_labels):

                    # if the segment is in the currently considered highest count cluster
                    if framei_label == self.counts[lc_cluster_counter][0]:
                        
                        # now find a merge pair/partner (only merge within same cluster)
                        for j, segj_labels in enumerate(self.labels):
                            
                            # since labels can be 2D: (n_segments, n_frames)
                            # loop, but only the unique elements
                            for framej_label in np.unique(segj_labels):

                                # if the segment is in the same high count cluster but not same seg index as i
                                if framej_label == self.counts[lc_cluster_counter][0] and i != j:
                                    
                                    # make sure the current merge pair hasn't already happened
                                    frozen_pair = frozenset((i, j))
                                    
                                    # if the merge pair is new, add to set so it can't happen again
                                    # also need to prevent/check for merging what is already split
                                    # second statement is asking, do they not share any items? T/F
                                    # TODO: is this needed? If going from largest cluster for merging
                                    #       then it should avoid the small clusters members by default
                                    if frozen_pair not in pairs_merged and frozen_pair.isdisjoint(self.to_split):
                                        pairs_merged.add(frozen_pair)
                                        # mark these as a merge pair
                                        self.to_merge[i].append(j)
                                        merges_remaining -= 1
                                        #print(f"merged {i} and {j}")
                                
                            # TODO: is there a better way to have this n_remaining check
                            #       which must happen at every potential split/merge event?
                            # extra precaution
                            if merges_remaining == 0:
                                break

                        # extra precaution
                        if merges_remaining == 0:
                            break                            
                    
                # extra precaution
                if merges_remaining == 0:
                    break

            # extra precaution
            if merges_remaining == 0:
                break
            
            # once done looping through all cluster labels, if there are still more
            # to merge, then do it again using the next lowest populated cluster label count
            lc_cluster_counter += 1

        # return finalized split and merge lists
        return self.to_split, self.to_merge
    
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

    # LCAS test
    # werl = WERL(pcoords)
    # split, merge = werl.lcas()
    # print(split, "\n", merge)
    # print([i for i, e in enumerate(split) if e != 0])
    # print("split sum: ", np.sum(split))
    # print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))
    # print("counts: ", werl.counts)

    # # REAP test
    # werl = WERL(pcoords)
    # print("N_CLUSTERS: ", werl.n_clusters)
    # werl.reap()
    # print("CLUSTER COUNTS: ", werl.counts)

    # # test output
    # split = np.loadtxt('split.txt')
    # merge = np.load('merge.npy', allow_pickle=True)
    # print(split, merge)

    # TLSC test
    werl = WERL(pcoords)
    #print(pcoords)
    #print(weights)

    kmeans, lc_centers, closest_frames = werl.tslc()
    test_Vis = werl.tslc_weighting(kmeans)
    # Visualize pcoord in CV space
    #plt.scatter([np.transpose(frame)[0] for frame in pcoords], [np.transpose(frame)[1] for frame in pcoords], s=40, alpha=0.3)
    #plt.show()

    # Assess distribution of cluster centers
    # plt.scatter(werl.pcoords[:,0], werl.pcoords[:,1], marker='.')
    # plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], alpha=0.4, s=200)
    # plt.scatter([l[0] for l in lc_centers], [l[1] for l in lc_centers], alpha=0.4, s=50, c='purple')
    # #plt.savefig('2D_TSLC')
    # plt.show()
    
