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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform

from .werl_base import WERL
from collections import Counter

class TSLC(WERL):
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    '''

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

    def compute_structure_reward(self, selected_points, weights):
        '''
        Computes the reward for each structure and returns it as an array.
        This is the exact function DTY defined in werl_reap.py
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
            This is the exact function DTY defined in werl_reap.py
            '''

            def rewards_function(w):
                r = self.compute_structure_reward(selected_points, w)
                R = r.sum()
                return R

            R = rewards_function(weights)

            return R, rewards_function

    def FFT(self, l=2):
        '''
        Fastest First Traversal algorithm for pruning cluster centers
        From the implementation of TLSC by Kleiman and Shukla adapted from Buenfil et al.

        Updates: n_clusters, pcoords (shape)
        '''

        # Calculate n_clusters_prime 
        #self.pcoords = self.pcoords.reshape(-1, self.pcoords.shape[2])
        n_clusters_prime = int(self.n_clusters * np.log(self.n_clusters) + l)

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
        for _ in range(1, self.n_clusters):
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
    
    def tslc_clustering(self):
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
        centroids = self.FFT()
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=1, init=centroids).fit(self.pcoords) # Perform one round of kmeans
        
        while len(np.unique(kmeans.labels_)) != self.n_clusters:
            centroids = self.FFT()
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=1, max_iter=1, init=centroids).fit(self.pcoords)

        # Update centers, labels, and counts
        self.centers = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        self.counts = np.array(Counter(kmeans.labels_).most_common())

        return kmeans

    def tslc_weighting(self, kmeans, N=5, d=1):
        '''
        Weighting method adapted from TLSC weighting algorithm.

        Updates
        -------
        self.labels : array
            Cluster labels for each data point.
        self.counts : list of tuples
            Count of each cluster label amount.

        Returns: weights - size (1, num CVs)
        '''
        
        # Get N lc clusters from the cluster id and its associated trajs 
        clusterid_counts = sorted([(i, np.unique(kmeans.labels_, return_counts=True)[1][i]) for i in range(len(kmeans.cluster_centers_))], key=lambda x: x[1])
        lc_clusters = clusterid_counts[:N]

        # Get centers for the N lc_clusters
        lc_centers = [kmeans.cluster_centers_[lc_clusters[i][0]] for i in range(len(lc_clusters))]

        # 4) PCA for points in each lc cluster -> d space
        # The keys of this dictionary are the lc cluster IDs
        lc_data_dict = {c[0]:[] for c in lc_clusters}
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] in [c[0] for c in lc_clusters]:
                lc_data_dict[kmeans.labels_[i]].append(list(self.pcoords[i]))

        Vis = []
        for key in lc_data_dict.keys():
            points = np.array(lc_data_dict[key])  # Here, all points (x, y) in the N lc clusters
            pca = PCA(n_components=d)
            pca_trans = pca.fit(points)  # Projection onto d space
            Vi = pca_trans.components_.T  # Must transpose to get desired shape
            Vis.append(Vi)
        print(Vis)

        # 5) Get Gi matrices of shape (D, K) that contain the gradient of 
        # each CV w.r.t. to the cartesian coordinate of the point.
        # Use a representative frame from the N least count clusters
        # The gradient will depend on the system used - i.e. some user defined function is necessary
        # This may be required in the constructor, but I am not sure how this could be passed
        # Currently, I have it implemented as is done in the MA REAP paper (arctan(y/x))
        # There is probably a way to auto differentiate based on an input function
        # But here the derivatives are hard coded in.
        
        Gis = []
        closest_frames, _ = pairwise_distances_argmin_min(lc_centers, self.pcoords)  # Find reps of the clusters closest to cluster center
        norm = 0
        for point in closest_frames:
            x, y = self.pcoords[point]
            dx = np.asarray([1, 0, -y / (x ** 2 + y ** 2)])  # arctan used as placeholder fxn since we have 2D currently
            dy = np.asarray([0, 1, x / (x ** 2 + y ** 2)])   # results in len(arr) weights
            Gi = np.vstack([dx, dy])
            norm += np.linalg.norm(Gi, axis=0)
            Gis.append(Gi)

        # 6) Normalize grads
        normed_Gis = []
        for Gi in Gis:
            normed_Gis.append(Gi * len(closest_frames) / norm)

        # 7) Get A - sum of elementwise multiplication (Gi.T * Vi) * (Vi.T * Gi) for each CV (i)
        K = Gis[0].shape[1]  # This is simply the number of CVs
        A = np.zeros((K, K)) # init matrix A
        for i in range(len(Gis)):
            Gi, Vi = Gis[i], Vis[i]
            A += (Gi.T @ Vi) @ (Vi.T @ Gi)

        # 8) Compute largest eigenvector of A, w~
        eig_vals, eig_vecs = np.linalg.eig(A)
        largest_eigval = np.argmax(eig_vals)
        # largest_eigvec = eig_vecs[largest_eigval]

        # 9) Compute w_k = w~_k ** 2 for k = [1, ..., K] in w~
        # MA-REAP code uses eig_vecs[:, largest_eigval], but I think that may be incorrect
        # since that will use the largest eigenvalue's index of each eigenvector.

        weights = eig_vecs[largest_eigval, :] ** 2  # Use the elements of the largest eigenvector for the weights
        #print(weights, np.sum(weights))

        return weights

    def tslc(self, N=5):
        '''
        WE version of the Tangent Space Least Adaptive Clustering (TLSC) algorithm.
        
        Input: Differentiable energy E, collective variables theta1 ... thetaK , number of least count clusters to sample from
        N, number of parallel runs M, number of iterations I, timesteps per iteration s, intrinsic dimension d

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


        if len(np.unique(self.pcoords)) == 1:
            return self.to_split, self.to_merge
        
        weights = [np.ones((n_features)) / n_features for _ in range(n_agents)]
        # Init logs
        least_counts_points_log = [[] for _ in range(n_agents)]
        cumulative_reward_log = [[] for _ in range(n_agents)]
        weights_log = [[] for _ in range(n_agents)]
        individual_rewards_log = [[] for _ in range(n_agents)]

        # Step 2: set initial OP weights (not WE traj weights)
        for i in range(n_agents):

            kmeans = self.tslc_clustering()

            # If clustering fails, do no split/merge operations
            if len(self.counts) < self.n_clusters:
               return self.to_split, self.to_merge

            # Get the closest (representative) points for each of the LC clusters
            least_counts_clusters, least_counts_points = self.select_least_counts()
            least_counts_points_log[i].append(least_counts_points)

            R, reward_fun = self.compute_cumulative_reward(least_counts_points, weights[i]) # reward_fxn used in weight opt
            cumulative_reward_log[i].append(-R)

            weights[i] = self.tslc_weighting(kmeans)
            weights_log[i].append(weights[i])

            # print(weights[i])
            # print(least_counts_points)

            ## This is sketchy - basically there are 3 weights (x, y, arctan)
            ## But we only have x,y pcoords, so I throw that arctan away - bad!
            ## Do I need to update the pcoords with whatever fxn is selected? 
            ## i.e. calculate arctan and store as a 3rd dimension? 

            rewards = self.compute_structure_reward(least_counts_points, weights[i][:len(least_counts_points[0])])

            individual_rewards_log[i].append(rewards)

        print("LC_CLUSTERS: ", least_counts_clusters)
        print(f"pcoord shape: {self.pcoords.shape}")
        print("OP weight log:", weights_log)
        print("LC point log:", least_counts_points_log)
        print("LC point log shape:", np.array(least_counts_points_log).shape)
        print("Cumulative reward log:", cumulative_reward_log)
        print("Individual reward log:", individual_rewards_log)

        
        sorted_indices = np.argsort(rewards)
        least_counts_clusters = least_counts_clusters[sorted_indices]
        print("SORTED LCC: ", least_counts_clusters)
        rewards = rewards[sorted_indices]
        print("SORTED REWARDS", rewards)


        ## TODO: Fix line below. It is trickier because there are essentially 
        ## 2 'n_clusters' variables that need to be tracked
        ## self.n_clusters in REAP refers to the number of kept LC clusters
        ## However, here, I am using it to get the number of original clusters first.

        # Insert least_count_clusters as a new column before the first column of cluster_counts
        # self.counts = np.insert(self.counts, 0, least_counts_clusters, axis=1)

        self.do_splitting()
        self.do_merging()

        return self.to_split, self.to_merge

if __name__ == "__main__":
    
    # this is the full pcoord array, in this case (80, 5, 2)
    # for 80 walkers, 5 frames of simulation each, and 2 pcoords (X and Y)
    pcoords = np.load('mm_potentials/pcoords_full.npy')

    pcoords = pcoords[40:,:,:]
    weights = np.ones(40) / 40
    print(f"pcoord full shape: {pcoords.shape}")

    ## Test data - uniform 
    #pcoords = np.array([9.5]*50).reshape(-1,1)
    #weights = np.array([0.02]*50)

    # TSLC test
    werl = TSLC(pcoords, weights)
    print("N_CLUSTERS: ", werl.n_clusters)
    split, merge = werl.tslc()
    #print("CLUSTER COUNTS: ", werl.counts)
    #print("CLUSTER COUNTS: ", werl.counts.shape)
    #print("LABELS", werl.labels)
    
    print("SPLIT: ", split, "\nMERGE: ", merge)
    print("SPLIT INDICES: ", [i for i, e in enumerate(split) if e != 0])
    print("split sum: ", np.sum(split))
    print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))
    #print("counts: ", werl.counts)

    # f = open('TSLC.out','w')
    # f.write("N_CLUSTERS: " + str(werl.n_clusters)+'\n')
    # f.write("CLUSTER COUNTS: " + str(werl.counts)+'\n')
    # f.write("CLUSTER COUNTS: " + str(werl.counts.shape)+'\n')
    # f.write("LABELS" + str(werl.labels)+'\n')
    
    # f.write("SPLIT: " + str(split)+ "\nMERGE: "+ str(merge)+'\n')
    # f.write("SPLIT INDICES: "+ str([i for i, e in enumerate(split) if e != 0])+'\n')
    # f.write("split sum: "+ str(np.sum(split))+'\n')
    # f.write("merge sum: "+ str(np.sum([np.count_nonzero(i) for i in merge]))+'\n')
    # f.write("counts: "+ str(werl.counts)+'\n')
    # f.close()

    # Visualize pcoord in CV space
    # plt.scatter([np.transpose(frame)[0] for frame in pcoords], [np.transpose(frame)[1] for frame in pcoords], s=40, alpha=0.3)
    # plt.title('pcoords')
    # plt.show()

    # Assess distribution of cluster centers
    # N = 5 # Set number of LC clusters
    # clusterid_counts = sorted([(i, np.unique(kmeans.labels_, return_counts=True)[1][i]) for i in range(len(kmeans.cluster_centers_))], key=lambda x: x[1])
    # lc_clusters = clusterid_counts[:N]

    # # Get centers for the N lc_clusters
    # lc_centers = [kmeans.cluster_centers_[lc_clusters[i][0]] for i in range(len(lc_clusters))]

    # # Plot to see clustering
    # plt.scatter(werl.pcoords[:,0], werl.pcoords[:,1], marker='.')
    # plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], alpha=0.4, s=200)
    # plt.scatter([l[0] for l in lc_centers], [l[1] for l in lc_centers], alpha=0.4, s=50, c='purple')
    # plt.title('clustering')
    # #plt.savefig('2D_TSLC')
    # plt.show()
    
