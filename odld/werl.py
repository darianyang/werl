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

    def __init__(self, pcoords, n_clusters=5):
        '''
        Parameters
        ----------
        segments : westpa segments object (TODO)
            Each segment also has a weight attribute.
        pcoords : array
            Last pcoord value of each segment for the current iteration.
        n_clusters : int
            for k-means, TODO: replace with dynamic TLSC heuristic n_cluster selection.
        '''
        # need to reshape 1d arrays from (n) to (n, 1) for kmeans fitting
        if pcoords.ndim == 1:
            self.pcoords = pcoords.reshape(-1, 1)
        else:
            self.pcoords = pcoords

        self.n_clusters = n_clusters
        
        # number of segments/walkers during the iteration
        self.n_segments = self.pcoords.shape[0]

        # list of walker positions to split, begin as all zeros (no splitting)
        self.to_split = [0] * self.n_segments
        # list of lists for walker positions to merge, begin as all empty
        self.to_merge = [[] for _ in range(self.n_segments)]
        # print(len(self.to_split))
        # print(len(self.to_merge))
        # list of all possible merge pairs
        #self.merge_pairs = list(combinations([i for i in range(self.n_segments)], 2))
        #print(self.merge_pairs)

    def LCAS(self, n_split=5):
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
            TODO: I wonder if it would be useful to be able to split the same walker multiple
                  times, right now limited to once, but the code is written so it can be multiple.
        
        Returns
        -------
        '''
        # TODO: eventually move clustering and LC to shared class method
        # kmeans clustering
        km = KMeans(n_clusters=self.n_clusters, random_state=1).fit(self.pcoords)
        centers = km.cluster_centers_
        labels = km.labels_

        # count each label amount
        # example output: Counter({0: 21, 1: 20, 2: 4, 3: 3, 4: 2})
        # then with most_common: [(0, 21), (1, 20), (2, 4), (3, 3), (4, 2)]
        counts = Counter(labels).most_common()
        #print(counts[-1][0])
        
        # split n_split times the lowest count cluster(s)
        splits_remaining = n_split
        # may need to go through multiple cluster labels if count is small
        lc_cluster_counter = 1
        while splits_remaining > 0:
            # go through each segment index label and tag to split
            for i, seg_label in enumerate(labels):
                # if the segment is in the least count cluster
                if seg_label == counts[-lc_cluster_counter][0]:
                    # increase split counter on this segment by 1
                    self.to_split[i] += 1
                    # note that one of the requested splits is done
                    splits_remaining -= 1
            # once done looping through all cluster labels, if there are still more
            # to split, then do it again using the next lowest populated cluster label count
            lc_cluster_counter += 1

        # TODO: can this be consolidated with splitting code?
        # merge (n_split * 2) times the highest count cluster(s)
        merges_remaining = n_split * 2
        # may need to go through multiple cluster labels
        lc_cluster_counter = 0
        #print(counts[lc_cluster_counter][0])
        # keep going until no more merges needed or until no more cluster labels avail
        while merges_remaining > 0 and lc_cluster_counter < self.n_clusters:
            # go through each segment index label and tag to merge
            for i, segi_label in enumerate(labels):
                #print(segi_label, lc_cluster_counter)
                # if the segment is in the least count cluster
                if segi_label == counts[lc_cluster_counter][0]:
                    # now find a merge pair/partner (only merge within same cluster)
                    for j, segj_label in enumerate(labels):
                        # if the segment is in the same least count cluster but not same seg index as i
                        if segj_label == counts[lc_cluster_counter][0] and i != j:
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

if __name__ == "__main__":
    # test data
    pcoords = np.loadtxt('pcoords.txt')
    #weights = np.loadtxt('weights.txt')
    werl = WERL(pcoords)
    print(werl.LCAS())

    # # test output
    # split = np.loadtxt('split.txt')
    # merge = np.load('merge.npy', allow_pickle=True)
    # print(split, merge)

