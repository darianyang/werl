'''
WERL class containing methods to implement various adaptive sampling
methods within the context of WE.

Example Return
--------------
To split: 50 total: 11 being split 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0]
To merge: 50 total: 11 being merged 
[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [40], [], [], [36, 21, 23, 26, 24, 22, 29, 25], [], [], [20], [], [19], [], [], []]

TODO:
    - probably will need a pcoord/input data standardization step
    - implement weight thresholds
    - think about how to target the middle clusters more
'''

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

class WERL:
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    '''
    
    # pmin=1e-12, pmax=0.1,
    def __init__(self, pcoords, weights, pmin=1e-50, pmax=0.1,
                 n_clusters=None, n_split=None, 
                 b=0.07, gamma=0.6, d=2):
        '''
        Parameters
        ----------
        segments : westpa segments object (TODO)
            Each segment also has a weight attribute.
        pcoords : ndarray
            Last pcoord value of each segment for the current iteration.
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
        b : float, default 1e-4 (REAP) or 0.07 (TSLC)
            Coefficient for n_clusters heuristic.
        gamma : float, default 0.7
            Exponent for n_clusters heuristic (should theoretically be in (0.5, 1)).
        d : float, default 2 
            Intrinsic dimensionality of slow manifold for the system. Used in n_clusters heuristic.
        '''
        # TODO: make sure all of these ndims are compatible with methods (TESTS using fake data)
        # need to reshape 1d arrays from (n) to (n, 1+) for kmeans fitting
        # this is the case where you have 1 frame per walker and 1 pcoord dim
        if pcoords.ndim == 1:
            self.pcoords = pcoords.reshape(-1, 1)
        # case where you just have 1 frame per walker and 2 pcoord dims
        elif pcoords.ndim == 2:
            self.pcoords = pcoords
        # when using the full 3D pcoord array (walkers, frames, pcoords)
        elif pcoords.ndim == 3:
            # all walkers as rows but only use last frame, keep the columns as each pcoord dim
            # also need to condense to 2D for km clustering anyway so it works out
            self.pcoords = pcoords[:,-1,:].reshape(-1, pcoords.shape[2])
        else:
            raise ValueError(f"pcoord input array has {pcoords.ndim} dimensions, must be 1-3.")

        # for weight thresholds
        self.weights = weights
        self.pmin = pmin
        self.pmax = pmax

        # number of segments/walkers and pcoords during the iteration
        self.n_segments = self.pcoords.shape[0]
        self.n_pcoords = self.pcoords.shape[1]

        if n_clusters is None:
            # calculate optimal amount of n_clusters
            #self.n_clusters = np.floor(b * self.n_segments ** (gamma * d)).astype(int)
            self.n_clusters = int(b * (self.n_segments ** (gamma * d)))
        else:
            # otherwise use input n_cluster number
            self.n_clusters = n_clusters

        print("n_clusters: ", self.n_clusters)

        # use n_clusters for n walkers to split if not given
        if n_split is None:
            self.n_split = self.n_clusters
        else:
            self.n_split = n_split

        # list of walker positions to split, begin as all zeros (no splitting)
        self.to_split = [0] * self.n_segments
        # list of lists for walker positions to merge, begin as all empty
        self.to_merge = [[] for _ in range(self.n_segments)]

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
        # kmeans clustering
        #km = KMeans(n_clusters=self.n_clusters, random_state=1).fit(self.pcoords)
        km = KMeans(n_clusters=self.n_clusters).fit(self.pcoords)
        self.centers = km.cluster_centers_
        #print(km.labels_)
        self.labels = km.labels_
        #print(self.labels)

        # count each label amount (using km.labels_ since not reshaped)
        # example output: Counter({0: 21, 1: 20, 2: 4, 3: 3, 4: 2})
        # then with most_common: [(0, 21), (1, 20), (2, 4), (3, 3), (4, 2)]
        # then change to array to be able to update label values later on
        self.counts = np.array(Counter(km.labels_).most_common())
        #print(self.counts)

    def do_splitting(self):
        '''
        Updates
        -------
        self.to_split : list
        '''
        # split n_split times the lowest count cluster(s)
        splits_remaining = self.n_split

        # may need to go through multiple cluster labels if count is small
        # this will be a negative index, so start with smallest cluster (-1)
        # and decrease by 1 if needed
        lc_cluster_counter = 1

        # TODO: debugging an indexing error in line 156 (-14 index from size 13 array)
        print("COUNTS: ", self.counts)

        while splits_remaining > 0:
           
            # go through each segment index label and tag to split
            for i, seg_label in enumerate(self.labels):
                
                # if the segment is in the least count cluster
                if seg_label == self.counts[-lc_cluster_counter][0]:
                    # check if weight is above min threshold
                    if self.weights[i] > self.pmin:
                        # increase split counter on this segment by 1
                        self.to_split[i] += 1
                        # note that one of the requested splits is done
                        splits_remaining -= 1
                
                # extra precaution
                if splits_remaining == 0:
                     break
            
            # once done looping through all cluster labels, if there are still more
            # to split, then do it again using the next lowest populated cluster label count
            # to only split the lowest cluster, don't use the counter
            lc_cluster_counter += 1

    def do_merging(self):
        '''
        Updates
        -------
        self.to_merge : list
        '''
        # merge n_split times the highest count cluster(s)
        merges_remaining = self.n_split
        
        # may need to go through multiple cluster labels
        # starting with biggest cluster and +1 from there
        lc_cluster_counter = 0
        
        # make sure there isn't redundant merge pairs
        # using set since sets are unordered, unchangable, with no duplicates
        pairs_merged = set()

        # keep going until no more merges needed or until no more cluster labels avail
        #while merges_remaining > 0 and lc_cluster_counter < self.n_clusters:
        while merges_remaining > 0:
            
            # go through each segment index label and tag to merge
            for i, segi_label in enumerate(self.labels):
                
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
                
                # if the segment is in the least count cluster
                if segi_label == self.counts[lc_cluster_counter][0]:
                    
                    # now find a merge pair/partner (only merge within same cluster)
                    for j, segj_label in enumerate(self.labels):
                        
                        # if the segment is in the same least count cluster but not same seg index as i
                        if segj_label == self.counts[lc_cluster_counter][0] and i != j:
                            
                            # make sure the current merge pair hasn't already happened
                            frozen_pair = frozenset((i, j))
                            
                            # if the merge pair is new, add to set so it can't happen again
                            # also make sure not being split already
                            if frozen_pair not in pairs_merged \
                            and self.to_split[i] == 0 and self.to_split[j] == 0:
                                # check to ensure merged weight is below max threshold
                                if self.weights[i] + self.weights[j] < self.pmax:
                                    pairs_merged.add(frozen_pair)
                                    # mark these as a merge pair
                                    self.to_merge[i].append(j)
                                    merges_remaining -= 1
                                    #print(f"merged {i} and {j}")
                        
                        # extra precaution
                        if merges_remaining == 0:
                            break
                
                # extra precaution
                if merges_remaining == 0:
                    break
            
            # once done looping through all cluster labels, if there are still more
            # to merge, then do it again using the next lowest populated cluster label count
            lc_cluster_counter += 1

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

    werl = WERL(pcoords)
    print(werl.pcoords)
    werl._clustering()
