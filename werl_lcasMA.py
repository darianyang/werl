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

    def equal_size_clustering(self):
        '''
        Returns self.n_clusters amount of clusters of equal size.
        '''
        # # Perform K-means clustering
        # km = KMeans(n_clusters=self.n_agents, random_state=1).fit(self.pcoords)
        # cluster_labels = km.labels_
        # self.original_labels = cluster_labels
        # #print(f"{cluster_labels=}")
        # print(f"{self.n_agents=}")

        # # Calculate the size of each cluster
        # cluster_sizes = np.bincount(cluster_labels)
        # #print(f"{cluster_sizes=}")

        # # Find the target size for each cluster (equal distribution)
        # # rows of pcoords / n_agents
        # target_size = self.pcoords.shape[0] // self.n_agents

        # #print(f"{self.pcoords.shape=}")

        # # Sort centroids by cluster size in descending order
        # centroids = km.cluster_centers_
        # sorted_centroids_indices = np.argsort(cluster_sizes)[::-1]
        # sorted_centroids = centroids[sorted_centroids_indices]
        # print(f"{sorted_centroids=}")

        # print("\nBegin reassignment: ")
        # # Iterate through clusters and balance sizes
        # for i in range(self.n_agents):
        #     print(f"Cluster {i}")
        #     # Cluster index
        #     idx = sorted_centroids_indices[i]
            
        #     # Calculate distances to other centroids
        #     distances_to_other_centroids = np.linalg.norm(
        #         centroids - centroids[idx], axis=1)
            
        #     # Indices of other centroids (i < j <= k)
        #     other_centroids_indices = sorted_centroids_indices[i+1:]
            
        #     # Iterate through other centroids
        #     for j in other_centroids_indices:
        #         # Find data points in cluster i closest to centroid j
        #         cluster_i_points = self.pcoords[cluster_labels == idx]
        #         distances_to_j = np.linalg.norm(
        #             cluster_i_points - centroids[j], axis=1)
                
        #         #closest_point_idx = np.argmin(distances_to_j)
        #         #closest_point_idx = np.argmin(distances_to_j[cluster_labels == idx])
        #         closest_point_idx = np.argmin(distances_to_j)
        #         #closest_point_idx = np.where(cluster_labels == idx)[0][closest_point_idx]

        #         closest_point = cluster_i_points[closest_point_idx]
                
        #         # Move closest point to cluster j
        #         cluster_labels[np.where((self.pcoords == closest_point).all(axis=1))] = j
                
        #         # Recompute centroid i (don't recompute the cluster)
        #         centroids[idx] = np.mean(self.pcoords[cluster_labels == idx], axis=0)
                
        #         # Update cluster sizes
        #         cluster_sizes[idx] -= 1
        #         cluster_sizes[j] += 1
                
        #         # Check if cluster i reaches target size
        #         if cluster_sizes[idx] <= target_size:
        #             print(f"Break at: {i=}{j=}{cluster_sizes=}")
        #             break
            
        # print(f"{cluster_labels=}")
        # print(f"{np.bincount(cluster_labels)=}")
        # self.final_labels = cluster_labels


        cluster_size = self.pcoords.shape[0] // self.n_agents
        km = KMeans(self.n_agents)
        km.fit(self.pcoords)
        centers = km.cluster_centers_
        # Centers gets repeated such that each center appears cluster_size times
        centers = centers.reshape(-1, 1, self.pcoords.shape[-1]).repeat(cluster_size, 1).reshape(-1, self.pcoords.shape[-1])
        # The distance_matrix has the distance of each point to each center repeated cluster_size times
        distance_matrix = cdist(self.pcoords, centers)
        # The linear_sum_assignment matches each point to the closest available center 
        # (remember the centers are repeated) in a 1-1 mapping. [1] of the output says 
        # which repeated center matches to [0] of the input ([0] is sorted so its just arange). 
        # Therefore, the // drops the repetition giving the correct cluster for each input. 
        cluster_labels = linear_sum_assignment(distance_matrix)[1]//cluster_size

        # Find the closest point to the mean within each cluster
        updated_centers = np.zeros_like(km.cluster_centers_)
        for i in range(self.n_agents):
            # if you want the average centroids
            #updated_centers[i] = X[clusters == i].mean(axis=0)
            # but we want representative centroids
            cluster_points = self.pcoords[cluster_labels == i]
            cluster_mean = cluster_points.mean(axis=0)
            closest_point_idx = np.argmin(np.linalg.norm(cluster_points - cluster_mean, axis=1))
            updated_centers[i] = cluster_points[closest_point_idx]
        
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
        self.equal_size_clustering()


        # # if less than the requested amount of clusters was generated
        # # then go with zero array return, currently useful for w_init
        # # basically, if the clustering fails, do no split/merge operations
        # if len(self.counts) < self.n_clusters:
        #     return self.to_split, self.to_merge

        # # do clustering to look for LC clusters
        # # self._clustering()
        # #print("shape labels: ", self.labels.shape)

        # # run split merge methods, updating split/merge lists
        # self.do_splitting()
        # self.do_merging()

        # # return finalized split and merge lists
        # return self.to_split, self.to_merge


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    # test data (1D array of 1D ODLD endpoints)
    # pcoords = np.loadtxt('mm_potentials/pcoords.txt')
    # weights = np.loadtxt('mm_potentials/weights.txt')

    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    pcoords = np.load('pcoords_full.npy')
    print(f"pcoord full shape: {pcoords.shape}")

    # cut to n walkers for testing
    n = 80
    pcoords = pcoords[:n,:,:]
    weights = np.ones(n) / n

    # def get_even_clusters(X, n_clusters):
    #     cluster_size = X.shape[0] // n_clusters
    #     kmeans = KMeans(n_clusters)
    #     kmeans.fit(X)
    #     centers = kmeans.cluster_centers_
    #     print(f"{centers}")
    #     centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
    #     distance_matrix = cdist(X, centers)
    #     clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size

    #     # Find the closest point to the mean within each cluster
    #     updated_centers = np.zeros_like(kmeans.cluster_centers_)
    #     for i in range(n_clusters):
    #         cluster_points = X[clusters == i]
    #         cluster_mean = cluster_points.mean(axis=0)
    #         closest_point_idx = np.argmin(np.linalg.norm(cluster_points - cluster_mean, axis=1))
    #         updated_centers[i] = cluster_points[closest_point_idx]

    #     return clusters, updated_centers, centers
    # pcoords = pcoords[:,-1,:].reshape(-1, pcoords.shape[2])
    # even_clusts, updated_centers, old_centers = get_even_clusters(pcoords, 4)
    # print(even_clusts)
    # print(updated_centers)
    # print(np.bincount(even_clusts))
    # fig, ax = plt.subplots()
    # ax.scatter(pcoords[:,0], pcoords[:,1], c=even_clusts, s=10)
    # ax.scatter(updated_centers[:,0], updated_centers[:,1], color='k', alpha=0.1)
    # ax.scatter(old_centers[:,0], old_centers[:,1], color='red', alpha=0.01)
    # plt.show()

    # test init data
    #pcoords = np.array([9.5] * 50).reshape(-1,1)
    #pcoords = np.array([np.random.uniform(5, 10, 50)]).reshape(-1,1)
    #weights = np.array([0.02] * 50)

    # LCAS test
    werl = LCAS_MA(pcoords, weights, n_agents=4)
    #werl.lcas_ma()
    even_clusts, updated_centers = werl.equal_size_clustering()
    print(even_clusts)
    print(updated_centers)
    print(np.bincount(even_clusts))
    fig, ax = plt.subplots()
    ax.scatter(werl.pcoords[:,0], werl.pcoords[:,1], c=even_clusts, s=10)
    ax.scatter(updated_centers[:,0], updated_centers[:,1], color='k', alpha=0.1)
    plt.show()

    #plt.scatter(werl.pcoords[:,0], werl.pcoords[:,1], c=werl.original_labels, s=10)
    #plt.scatter(werl.pcoords[:,0], werl.pcoords[:,1], c=werl.final_labels, s=10)
    #plt.show()

    #split, merge = werl.lcas_ma()
    # print("SPLIT: ", split, "\nMERGE: ", merge)
    # print("SPLIT INDICES: ", [i for i, e in enumerate(split) if e != 0])
    # print("split sum: ", np.sum(split))
    # print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))
    # print("counts: ", werl.counts)