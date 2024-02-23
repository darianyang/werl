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

from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import pdist, squareform

from werl_base import WERL

class TSLC(WERL):
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    '''

    def FFT(self, l=2):
        '''
        Fastest First Traversal algorithm for pruning cluster centers
        From the implementation of TLSC by Kleiman and Shukla adapted from Buenfil et al.

        Updates: n_clusters, pcoords (shape)
        '''

        # Calculate n_clusters_prime 
        self.pcoords = self.pcoords.reshape(-1, self.pcoords.shape[2])
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

        Returns: weights - size (1, num CVs)
        '''
        
        # Get N lc clusters from the cluster id and its associated trajs 
        clusterid_counts = sorted([(i, np.unique(kmeans.labels_, return_counts=True)[1][i]) for i in range(len(kmeans.cluster_centers_))], key=lambda x: x[1])
        lc_clusters = clusterid_counts[:N]

        # Get centers for the N lc_clusters
        lc_centers = [kmeans.cluster_centers_[lc_clusters[i][0]] for i in range(len(lc_clusters))]

        # 4) PCA for points in each lc cluster -> d space
        lc_data_dict = {c[0]:[] for c in lc_clusters}
        for i in range(len(kmeans.labels_)):
            if kmeans.labels_[i] in [c[0] for c in lc_clusters]:
                lc_data_dict[kmeans.labels_[i]].append(list(self.pcoords[i]))

        Vis = []
        for key in lc_data_dict.keys():
            points = np.array(lc_data_dict[key])
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

        # Step 1: define some hyperparameters and initialize arrays --> To be provided via init_variables
        n_agents = 1  # Forced
        delta = 0.02  # Upper boundary for learning step
        n_features = 2  # Number of variables in OP space (2d pcoord for now)

        # Step 2: set initial OP weights (not WE traj weights)
        weights = [np.ones((n_features)) / n_features for _ in range(n_agents)]

        kmeans = self.tslc_clustering()
        weights = self.tslc_weighting(kmeans)

        return weights, kmeans

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

    # TSLC test
    werl = TSLC(pcoords)
    #print(pcoords)
    #print(weights)

    weights, kmeans = werl.tslc()
    print(weights)
    #test_Vis = werl.tslc_weighting(kmeans)
    # Visualize pcoord in CV space
    #plt.scatter([np.transpose(frame)[0] for frame in pcoords], [np.transpose(frame)[1] for frame in pcoords], s=40, alpha=0.3)
    #plt.show()

    #Assess distribution of cluster centers
    # N = 5 # Set number of LC clusters
    # clusterid_counts = sorted([(i, np.unique(kmeans.labels_, return_counts=True)[1][i]) for i in range(len(kmeans.cluster_centers_))], key=lambda x: x[1])
    # lc_clusters = clusterid_counts[:N]

    # # Get centers for the N lc_clusters
    # lc_centers = [kmeans.cluster_centers_[lc_clusters[i][0]] for i in range(len(lc_clusters))]

    # # Plot to see clustering
    # plt.scatter(werl.pcoords[:,0], werl.pcoords[:,1], marker='.')
    # plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], alpha=0.4, s=200)
    # plt.scatter([l[0] for l in lc_centers], [l[1] for l in lc_centers], alpha=0.4, s=50, c='purple')
    # #plt.savefig('2D_TSLC')
    # plt.show()
    
