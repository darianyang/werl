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
    '''
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
    werl = LCAS(pcoords)
    split, merge = werl.lcas()
    print(split, "\n", merge)
    print([i for i, e in enumerate(split) if e != 0])
    print("split sum: ", np.sum(split))
    print("merge sum: ", np.sum([np.count_nonzero(i) for i in merge]))
    print("counts: ", werl.counts)