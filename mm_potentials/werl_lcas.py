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
from werl_base import WERL

class LCAS(WERL):
    '''
    Weighted Ensemble Reinforcement Learning (WERL).
    Using Least Counts Adaptive Sampling (LCAS).
    '''

    def lcas(self):
        '''
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

    # cut to n walkers for testing
    n = 80
    pcoords = pcoords[:n,:,:]
    weights = np.ones(n) / n

    # test init data
    #pcoords = np.array([9.5] * 50).reshape(-1,1)
    #pcoords = np.array([np.random.uniform(5, 10, 50)]).reshape(-1,1)
    #weights = np.array([0.02] * 50)

    # LCAS test
    werl = LCAS(pcoords, weights)
    split, merge = werl.lcas()
    print("SPLIT: ", split, "\nMERGE: ", merge)
    print("SPLIT INDICES: ", [i for i, e in enumerate(split) if e != 0])
    print("split sum: ", np.sum(split))
    print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))
    print("counts: ", werl.counts)