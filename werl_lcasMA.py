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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from werl_base import WERL

class LCAS_MA(WERL):
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    Using Least Counts Adaptive Sampling (LCAS) with multiple agents (MA).

    The idea here is to first do some overall clustering with equal member cluster
    output, thus dividing the pcoord space evenly. These are the agents, and each 
    division of segments will then go through a round of LCAS, getting their respective
    split and merge decisions. LC clustering will be done using all pcoord dimensions
    for the ininital implementation, but later can selectively choose a pcoord dimension
    based on RL.

    After that is implemented, set up the RL component: where each recycling event is 
    an episode and the reward of each episode is -0.01 for each tau or timestep the walker
    took, and a reward of +1 for the recycling. For each representative centroid of the
    initial equal-member clustering, these are in continuous observation space, and the
    action space is the pcoord dimension choosen. These choices will be recorded in a 
    Q-table like structure, thus the agents will all contribute to learning which action
    leads to more rewards at each point in observation space. If needed, DQL can be used to
    approximate large state spaces.
    '''

    def __init__(self, pcoords, weights, pmin=1e-12, pmax=0.1, 
                 n_clusters=None, n_split=None, n_agents=1,
                 b=0.07, gamma=0.6, d=2):
        '''
        Parameters
        ----------
        pcoords : ndarray
            Last pcoord value of each segment for the current iteration.
        TODO: option for automatic pcoord input data standard scaling.
        weights : array
            Weight of each segment for the iteration. 
            The sum of all weights should be 1.0.
        pmin : float
            Minimum weight threshold.
        pmax : float
            Maximum weight threshold.
        n_clusters : int
            Number of clusters for k-means.
            If self.n_clusters is None, calculating optimal n_clusters using a
            heuristic approximation from Buenfil, Koelle, Meila, ICML (2021).
        n_split : int
            How many segs to split, later determined from reward funct opt ? (TODO).
            n_merge will be based on the same value.
            Currently going to use the same value as n_clusters if None.
            TODO: I wonder if it would be useful to be able to split the same walker multiple
                  times, right now limited to once, but the driver code is written so it can be multiple.
        n_agents : int
            Number of initial equal size clusters to split dataset into.
            Each cluster will act as an agent.
        b : float, default 1e-4 (REAP) or 0.07 (TSLC)
            Coefficient for n_clusters heuristic.
        gamma : float, default 0.7
            Exponent for n_clusters heuristic (should theoretically be in (0.5, 1)).
        d : float, default 2 
            Intrinsic dimensionality of slow manifold for the system. Used in n_clusters heuristic.
        '''
        super().__init__(pcoords, weights, pmin, pmax, n_clusters, n_split, b, gamma, d)

        self.n_agents = n_agents
        # list of walker positions to split, begin as all zeros (no splitting)
        self.multi_to_split = [0] * self.n_segments
        # list of lists for walker positions to merge, begin as all empty
        self.multi_to_merge = [[] for _ in range(self.n_segments)]

        # store entire pcoord array separately (TODO: actually, is this needed? prob not but may be cleaner)
        self.pcoords_full = self.pcoords

        # calc size of each eq cluster
        self.eq_cluster_size = self.n_segments // self.n_agents

        # for each eq member cluster: set amount to cluster and split with the reduced dataset
        if n_clusters is None:
            # calculate optimal amount of n_clusters for each eq cluster
            self.n_clusters = self.calc_n_clusters(self.eq_cluster_size, b, gamma, d)
            print("EQ Cluster Size: ", self.n_clusters)
        else:
            # otherwise use input n_cluster number
            self.n_clusters = n_clusters

        # use n_clusters for n walkers to split/merge if not given
        if n_split is None:
            self.n_split = self.n_clusters
        else:
            self.n_split = n_split

    def equal_size_clustering(self):
        '''
        Returns self.n_agents amount of clusters of equal size.
        Also returns updated representative cluster centroids.
        '''
        # run kmeans
        #km = KMeans(self.n_agents).fit(self.pcoords_full)
        km = KMeans(self.n_agents, random_state=1).fit(self.pcoords_full)

        # # TODO: could detect failed clustering with duplicate initial points here
        # if len(np.unique(km.cluster_centers_)) == 1:
        #     return
        
        # get cluster centers
        centers = km.cluster_centers_
        # centers gets repeated such that each center appears cluster_size times
        centers = centers.reshape(-1, 1, self.n_pcoords).repeat(self.eq_cluster_size, 1).reshape(-1, self.n_pcoords)
        # the distance_matrix has the distance of each point to each center repeated cluster_size times
        distance_matrix = cdist(self.pcoords_full, centers)
        # the linear_sum_assignment matches each point to the closest available center 
        # (remember the centers are repeated) in a 1-1 mapping. [1] of the output says 
        # which repeated center matches to [0] of the input ([0] is sorted so its just arange). 
        # Therefore, the // drops the repetition giving the correct cluster for each input. 
        cluster_labels = linear_sum_assignment(distance_matrix)[1]//self.eq_cluster_size

        # Find the closest point to the mean within each cluster
        updated_centers = np.zeros_like(km.cluster_centers_)
        for i in range(self.n_agents):
            # if you want the average centroids
            #updated_centers[i] = X[clusters == i].mean(axis=0)
            # but we want representative centroids
            cluster_points = self.pcoords_full[cluster_labels == i]
            cluster_mean = cluster_points.mean(axis=0)
            closest_point_idx = np.argmin(np.linalg.norm(cluster_points - cluster_mean, axis=1))
            updated_centers[i] = cluster_points[closest_point_idx]
        
        #print(f"{np.bincount(cluster_labels)=}")
        return cluster_labels, updated_centers

    def lcas_ma(self):
        '''
        Least Counts Adaptive Sampling. In LCAS, from segment data of the previous iteration, 
        cluster the trajectory endpoint pcoord data, then select the next starting 
        states from the smallest, or least count, clusters. The idea being that clusters with 
        fewer members correspond to more sparsely sampled regions of the configuration space, 
        and will tend to be towards the boundaries of the explored region.

        This implementation uses 'multiple agents', which in this context just means first discretizing
        the input data, then running LCAS in each portion of data, e.g. from clustering.

        Returns
        -------
        self.to_split : list
        self.to_merge : list
        '''        
        # do equal member clustering
        eq_labels, eq_centers = self.equal_size_clustering()

        # to handle initial equal points from w_init with failed clustering convergence
        # also consider when using 1 agent, which will also return 1 cluster
        if self.n_agents > 1 and len(np.unique(eq_labels)) == 1:
            # return empty split merge lists
            return self.multi_to_split, self.multi_to_merge

        #print(f"{self.pcoords_full}")
        #print(f"{eq_labels=}")

        # loop each eq cluster / agent, run LCAS for each eq cluster set
        # return split / merge decisions and fill out multi_to_split/merge lists
        for eq_cluster_i in range(self.n_agents):
            # extract data points belonging to the current equal member cluster
            cluster_points = self.pcoords_full[eq_labels == eq_cluster_i]
            # extract data indices belonging to the current equal member cluster
            cluster_indices = np.where(eq_labels == eq_cluster_i)[0]
            #print(f"{cluster_indices=}")
            #print(cluster_points)

            # replace pcoords instance attr with current eq cluster data points
            self.pcoords = cluster_points
            # run clustering on pcoord subset from eq cluster i
            self._clustering()
            #print(self.counts)
            # if less than the requested amount of clusters was generated
            # then go with zero array return, currently useful for w_init
            # basically, if the clustering fails, do no split/merge operations
            # useful for using 1 agent during init
            if len(self.counts) < self.n_clusters:
                return self.to_split, self.to_merge

            # make temp eq cluster size split and merge lists
            self.to_split = [0] * self.eq_cluster_size
            self.to_merge = [[] for _ in range(self.eq_cluster_size)]

            # run split merge methods, updating current eq cluster sized temp split/merge lists
            self.do_splitting()
            self.do_merging()

            print(f"{self.to_split=}, \n{self.to_merge=}")

            # now map these split and merge decisions back to the multi split/merge lists
            for i, mapping_idx in enumerate(cluster_indices):
                #print(i, mapping_idx)
                self.multi_to_split[mapping_idx] = self.to_split[i]
                self.multi_to_merge[mapping_idx] = self.to_merge[i]

        # return finalized split and merge lists
        return self.multi_to_split, self.multi_to_merge


if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    # test data (1D array of 1D ODLD endpoints)
    # pcoords = np.loadtxt('mm_potentials/pcoords.txt')
    # weights = np.loadtxt('mm_potentials/weights.txt')

    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    pcoords = np.load('pcoords_full.npy')
    #print(f"pcoord full shape: {pcoords.shape}")

    # cut to n walkers for testing
    n = 40
    pcoords = pcoords[:n,:,:]
    weights = np.ones(n) / n

    # test init data
    #pcoords = np.array([9.5] * 50).reshape(-1,1)
    #pcoords = np.array([np.random.uniform(5, 10, 50)]).reshape(-1,1)
    #weights = np.array([0.02] * 50)

    # LCAS test
    # TODO: note in docs: recommend not to use less than 20 segs per agent 
    # TODO: can probably replace the lcas resampler with this one since it defaults to 1 agent anyway
    #       which I tested with constant seed and it is the same result
    werl = LCAS_MA(pcoords, weights, n_agents=2)
    #print(werl.calc_n_clusters(17))
    split, merge = werl.lcas_ma()
    print("SPLIT: ", split, "\nMERGE: ", merge)
    print("SPLIT INDICES: ", [i for i, e in enumerate(split) if e != 0])
    print("split sum: ", np.sum(split))
    print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))
    #print("counts: ", werl.counts)

    # eq clustering test
    # even_clusts, updated_centers = werl.equal_size_clustering()
    # print(even_clusts)
    # print(updated_centers)
    # print(np.bincount(even_clusts))
    # fig, ax = plt.subplots()
    # ax.scatter(werl.pcoords[:,0], werl.pcoords[:,1], c=even_clusts, s=10)
    # ax.scatter(updated_centers[:,0], updated_centers[:,1], color='k', alpha=0.1)
    # plt.show()