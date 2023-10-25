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
import matplotlib.pyplot as plt
from collections import Counter
#from itertools import combinations

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
        pcoords : array
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
        # need to reshape 1d arrays from (n) to (n, 1) for kmeans fitting
        if pcoords.ndim == 1:
            self.pcoords = pcoords.reshape(-1, 1)
        else:
            self.pcoords = pcoords

        # number of segments/walkers during the iteration
        self.n_segments = self.pcoords.shape[0]

        if n_clusters is None:
            # calculate optimal amount of n_clusters
            #self.n_clusters = np.floor(b * self.n_segments ** (gamma * d)).astype(int)
            self.n_clusters = int(b * (self.n_segments ** (gamma * d)))
        else:
            # otherwise use input n_cluster number
            self.n_clusters = n_clusters

        # list of walker positions to split, begin as all zeros (no splitting)
        self.to_split = [0] * self.n_segments
        # list of lists for walker positions to merge, begin as all empty
        self.to_merge = [[] for _ in range(self.n_segments)]

        # list of all possible merge pairs
        #self.merge_pairs = list(combinations([i for i in range(self.n_segments)], 2))


    def _clustering(self):
        '''
        Shared clustering method with k-means using self.n_clusters.
        '''
        # kmeans clustering
        km = KMeans(n_clusters=self.n_clusters, random_state=1).fit(self.pcoords)
        #self.centers = km.cluster_centers_
        self.labels = km.labels_

        # count each label amount
        # example output: Counter({0: 21, 1: 20, 2: 4, 3: 3, 4: 2})
        # then with most_common: [(0, 21), (1, 20), (2, 4), (3, 3), (4, 2)]
        self.counts = Counter(self.labels).most_common()
        
    def lcas(self, n_split=None):
        '''
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
        '''        
        # do clustering
        self._clustering()

        # if less than the requested amount of clusters was generated
        # then go with zero array return, currently useful for w_init
        # basically, if the clustering fails, do no split/merge operations
        if len(self.counts) < self.n_clusters:
            return self.to_split, self.to_merge
        
        # use n_clusters for n walkers to split if not given
        if n_split is None:
            n_split = self.n_clusters

        # split n_split times the lowest count cluster(s)
        splits_remaining = n_split
        
        # may need to go through multiple cluster labels if count is small
        lc_cluster_counter = 1
        
        while splits_remaining > 0:
           
            # go through each segment index label and tag to split
            for i, seg_label in enumerate(self.labels):
                
                # if the segment is in the least count cluster
                if seg_label == self.counts[-lc_cluster_counter][0]:
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

        # TODO: can this be consolidated with splitting code?
        # TODO: there is issues with merging where after the main high cluster labels
        #       there is overlap in merging, e.g. [[], [2, 10, 11, 26, 28, 33, 36, 37], [1, 10], ...]
        #       here, 2 gets merged to 1 and then 1 to 2.
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
                            if frozen_pair not in pairs_merged:
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

        # return finalized split and merge lists
        return self.to_split, self.to_merge
    
    def _calc_reward(self):
        '''
        For a set of order parameters, calculate the reward value of the segment.

        TODO: reap, tslc reward calcs.
        '''
        pass

if __name__ == "__main__":
    # test data
    pcoords = np.loadtxt('pcoords.txt')
    weights = np.loadtxt('weights.txt')

    # test init data
    # pcoords = np.array([9.5] * 50).reshape(-1,1)
    # weights = np.array([0.02] * 50)

    werl = WERL(pcoords)
    #werl._clustering()
    split, merge = werl.lcas(15)
    print(split, "\n", merge)
    print(werl.counts)

    # # test output
    # split = np.loadtxt('split.txt')
    # merge = np.load('merge.npy', allow_pickle=True)
    # print(split, merge)

