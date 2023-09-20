'''
Main functions for running the WESTPA implementation of REVO (WEVO).

A portion of this code is adapted from github.com/ADicksonLab/wepy
MIT License | Copyright (c) 2017, 2020 ADicksonLab


TODO: maybe eventually have something like group.py but resampler.py
      so it would take the input coordinates and output labels for split and merge
'''

import numpy as np
import itertools
import logging
import random

import os
import os.path

# only if wevo.log exists
if os.path.exists('wevo.log'):
    # remove to allow starting a fresh logger file
    os.remove ('wevo.log')
logger = logging.getLogger('wevo')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('wevo.log')
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class WEVO:
    '''
    WE resampling module based on REVO (WEVO).

    1. Input ensemble of walkers
    2. Run WEVO to decide which walkers to split and which to merge
    3. Return the decisions for splitting and merging

    Eventually incorporate into the following workflow (from REVO):
        - Calculate the pairwise all-to-all distance matrix using the distance metric
        - Decide which walkers should be merged or cloned
        - Apply the cloning and merging decisions to get the resampled walkers
        - Create the resampling data that includes:
            - distance_matrix : the calculated all-to-all distance matrix
            - n_walkers : the number of walkers. number of walkers is
                            kept constant thought the resampling.
            - variation : the final value of trajectory variation
            - images : the images of walkers that is defined by the distance object
            - image_shape : the shape of the image
    '''
    
    # TODO: may need to get segments and intercept at assign step
    # so may need WEVO mapper and sim manager
    # see binless PR for details: https://github.com/westpa/westpa/pull/240/files

    #def __init__(self, segments, pcoords, weights):
    def __init__(self, pcoords, weights, maxcycles=100,
                 merge_dist=None,
                 char_dist=None,
                 distance=None,
                 init_state=None,
                 #weights=True,
                 merge_alg='greedy',
                 pmin=1e-12,
                 pmax=0.1,
                 dist_exponent=4,
                 #seed=None,
                 **kwargs):
        '''
        Parameters
        ----------
        segments : westpa segments object (TODO)
            Each segment also has a weight attribute.
        pcoords : array
            Last pcoord value of each segment for the current iteration
        weights : array
            Weight of each segment for the iteration. 
            The sum of all weights should be 1.0.
        maxcycles : int
            Limit the amount of variance optimization iterations.
            Prevents infinite optimization with some test systems.

        # REVO
        dist_exponent : int
          The distance exponent that modifies distance and weight novelty
          relative to each other in the variation equation.
        merge_dist : float
            The merge distance threshold. Units should be the same as
            the distance metric. The distance between merged walkers should be
            less than this.
        char_dist : float
            The characteristic distance value. It is calculated by
            running a single dynamic cycle and then calculating the
            average distance between all walkers. Units should be the
            same as the distance metric.
        distance : object implementing Distance
            The distance metric to compare walkers.
        weights : bool (TODO: not used)
            Turns off or on the weight novelty in
            calculating the variation equation. When weight is
            False, the value of the novelty function is set to 1 for all
            walkers.
        merge_alg : string (TODO: implement pairs)
            Indication of which algorithm is used to find pairs to merge.
            'pairs' (default) indicates that a list of all suitable pairs is generated,
            and the pair that minimizes the expected variation loss is chosen.
            'greedy' indicates that first the lowest variation walker is 
            selected, and this is attempted to be merged with the closest 
            suitable walker
        init_state : WalkerState object
            Used for automatically determining the state image shape.
        seed : None or int, optional
            The random seed. If None, the system (random) one will be used.
        '''
        # no segs for now
        #self.segments = segments
        self.pcoords = pcoords
        self.weights = weights

        np.savetxt("pcoords.txt", pcoords)
        np.savetxt("weights.txt", weights)

        self.maxcycles = maxcycles

        self.pmin = pmin
        self.pmax = pmax

        # revo inits
        assert merge_dist is not None, "Merge distance must be given."
        #assert distance is not None,  "Distance object must be given."
        assert char_dist is not None, "Characteristic distance value (d0) must be given"
        #assert init_state is not None,  "An initial state must be given."

        # ln(probability_min)
        self.lpmin = np.log(pmin/100)
        self.dist_exponent = dist_exponent

        self.merge_dist = merge_dist
        self.merge_alg = merge_alg
        #logger.info(f"Using {self.merge_alg} merge algorithm")

        # the distance metric 
        # TODO: this may be equivalent to pcoords array (only used in dist matrix calc)
        self.distance = distance

        # the characteristic distance, char_dist
        self.char_dist = char_dist

        # setting the random seed (not used currently)
        # self.seed = seed
        # if seed is not None:
        #     random.seed(seed)

        # we do not know the shape and dtype of the images until
        # runtime so we determine them here
        #image = self.distance.image(init_state)
        #self.image_dtype = image.dtype

    def _all_to_all_distance(self):
        '''
        Calculate the pairwise all-to-all distances between segments.
        TODO: JML sort function can turn distance matrix to paired list.
        TODO: can this handle ndarrays where n > 1 ?
            I think so, if pcoord = [[0, 0, n],
                                     [0, 0, n]
                                     [0, 0, n]
                                     [...]]

        Returns
        -------
        dist_matrix : 2d array
            Distance matrix between each segment coordinate value.
        '''
        # initialize an all-to-all matrix, with 0.0 for self distances (diagonal)
        dist_matrix = np.zeros((len(self.pcoords), len(self.pcoords)))

        # build distance matrix from pcoord value distances
        for i, pcoord_i in enumerate(self.pcoords):
            for j, pcoord_j in enumerate(self.pcoords):
                # calculate Euclidean distance between two points 
                # points can be n-dimensional
                dist = np.linalg.norm(pcoord_i - pcoord_j)
                dist_matrix[i,j] = dist
        
        return dist_matrix
    
    def _novelty(self, walker_weight, num_walker_copy):
        """
        Calculates the novelty function value. (TODO)

        Parameters
        ----------
        walker_weight : float
            The weight of the walker.
        num_walker_copy : int
            The number of copies of the walker.

        Returns
        -------
        novelty : float
            The calcualted value of novelty for the given walker.
        """
        novelty = 0
        if walker_weight > 0 and num_walker_copy > 0:
            # weight based novelty always on
            # if self.weights:
            #     novelty = np.log(walker_weight/num_walker_copy) - self.lpmin
            # else:
            #     novelty = 1
            novelty = np.log(walker_weight/num_walker_copy) - self.lpmin

        if novelty < 0:
            novelty = 0

        return novelty

    def _calc_variation(self, walker_weights, num_walker_copies, distance_matrix):
        """
        Calculates the variation value. (TODO)

        Parameters
        ----------
        walker_weights : list of float
            The weights of all walkers. The sum of all weights should be 1.0.
        num_walker_copies : list of int
            The number of copies of each walker.
            0 means the walker does not exist anymore.
            1 means there is one of the this walker.
            >1 means it should be cloned to this number of walkers.
        distance_matrix : list of arraylike of shape (num_walkers)

        Returns
        -------
        variation : float
           The calculated variation value.
        walker_variations : arraylike of shape (num_walkers)
           The Vi value of each walker.
        """
        num_walkers = len(walker_weights)

        # set the novelty values
        walker_novelties = np.array([self._novelty(walker_weights[i], num_walker_copies[i])
                                     for i in range(num_walkers)])

        # the value to be optimized
        variation = 0

        # the walker variation values (Vi values)
        walker_variations = np.zeros(num_walkers)

        # calculate the variation and walker variation values
        for i in range(num_walkers - 1):
            if num_walker_copies[i] > 0:
                for j in range(i+1, num_walkers):
                    if num_walker_copies[j] > 0:

                        # partial_variation = \
                        #     ((distance_matrix[i][j] / self.char_dist) ** self.dist_exponent) \
                        #     * walker_novelties[i] * walker_novelties[j]
                        partial_variation = \
                            ((distance_matrix[i,j] / self.char_dist) ** self.dist_exponent) \
                            * walker_novelties[i] * walker_novelties[j]

                        variation += partial_variation * num_walker_copies[i] * num_walker_copies[j]
                        walker_variations[i] += partial_variation * num_walker_copies[j]
                        walker_variations[j] += partial_variation * num_walker_copies[i]

        # TODO: should variation == sum(walker_variations) ?
        return variation, walker_variations

    def _calc_variation_loss(self, walker_variation, weights, eligible_pairs):
        """
        Calculates the loss to variation through merging of eligible walkers.
        TODO:

        Parameters                                    
        ----------
        walker_variations : arraylike of shape (num_walkers)
            The Vi value of each walker.
        weights : list of float                                  
            The weights of all walkers. The sum of all weights should be 1.0.
        eligible_pairs : list of tuples
            Pairs of walker indexes that meet the criteria for merging.

        Returns
        -------                                                       
        variation_loss_list : tuple
            A tuple of the walker merge pair indicies that meet the criteria 
            for merging and minimize variation loss.
        """

        v_loss_min = np.inf
        
        min_loss_pair = ()

        for pair in eligible_pairs:
            walker_i = pair[0]
            walker_j = pair[1]

            wt_i = weights[walker_i]
            wt_j = weights[walker_j]

            v_i = walker_variation[walker_i]
            v_j = walker_variation[walker_j]

            v_loss = (wt_j * v_i + wt_i * v_j) / (wt_i + wt_j)

            if v_loss < v_loss_min:
                min_loss_pair = pair

                v_loss_min = v_loss

        return min_loss_pair

    def _find_eligible_merge_pairs(self, weights, distance_matrix, max_var_idx, num_walker_copies):
        """ 
        Find pairs of walkers that are eligible to be merged. (TODO)
        
        Parameters
        ----------
        weights : list of floats                                                  
            The weights of all walkers. The sum of all weights should be 1.0.
        distance_matrix : list of arraylike of shape (num_walkers)
            The distance between every walker according to the distance metric.
        max_var_idx : float 
            The index of the walker that had the highest walker variance
            and is a candidate for cloning.
        num_walker_copies : list of int
            The number of copies of each walker.
                0 means the walker is not exists anymore.
                1 means there is one of the this walker.
                >1 means it should be cloned to this number of walkers.                        

        Returns
        -------
        eligible_pairs : list of tuples                                
            Pairs of walker indexes that meet the criteria for merging.
        """

        eligible_pairs = []

        for i in range(len(weights) - 1):
            for j in range(i + 1, len(weights)):
                if i != max_var_idx and j != max_var_idx:
                    if num_walker_copies[i] == 1 and num_walker_copies[j] == 1:
                        if weights[i] + weights[j] < self.pmax:
                            if distance_matrix[i,j] < self.merge_dist:
                                eligible_pairs.append((i,j))

        return eligible_pairs

    def decide(self, num_walker_copies, distance_matrix):
        """
        Optimize the trajectory variation by making decisions for resampling.
        # TODO

        Parameters
        ----------
        walker_weights : list of float
            The weights of all walkers. 
        num_walker_copies : list of int
            The number of copies of each walker.
            0 means the walker is not exists anymore.
            1 means there is one of the this walker.
            >1 means it should be cloned to this number of walkers.
        distance_matrix : list of arraylike of shape (num_walkers)

        Returns
        -------
        resampling_data : list of dict of {str : value}
            The resampling records resulting from the decisions.
        variation : float
            The optimized value of the trajectory variation.
        """
        num_walkers = len(self.weights)

        variations = []
        merge_groups = [[] for i in range(num_walkers)]
        walker_clone_nums = [0 for i in range(num_walkers)]

        # make copy of walkers properties
        new_walker_weights = self.weights.copy()
        # num_walker_copies from decide is just np.ones array for input
        new_num_walker_copies = num_walker_copies.copy()


        # calculate the initial variation which will be optimized
        variation, walker_variations = self._calc_variation(self.weights, 
                                                            new_num_walker_copies,
                                                            distance_matrix)
        variations.append(variation)

        # maximize the variance through cloning and merging
        #logging.info("Starting variance optimization: {}".format(variation))
        logger.info("Starting variance optimization: {}".format(variation))

        productive = True
        self.count = 0
        # limit the amount of optimization cycles (count) with max cycles arg
        while productive and self.count <= self.maxcycles:
            self.count += 1
            productive = False
            # find min and max walker_variationss, alter new_amp

            # initialize to None, we may not find one of each
            min_idx = None
            max_idx = None

            # selects a walker with minimum walker_variations and a walker with
            # maximum walker_variations walker (distance to other walkers) will be
            # tagged for cloning (stored in maxwind), except if it is
            # already a keep merge target
            max_tups = []
            for i, value in enumerate(walker_variations):
                # 1. must have an amp >=1 which gives the number of clones to be made of it
                # 2. clones for the given amplitude must not be smaller than the minimum probability
                # 3. must not already be a keep merge target
                if (new_num_walker_copies[i] >= 1) and \
                   (new_walker_weights[i]/(new_num_walker_copies[i] + 1) > self.pmin) and \
                   (len(merge_groups[i]) == 0):
                    max_tups.append((value, i))


            if len(max_tups) > 0:
                max_value, max_idx = max(max_tups)

            #logger.info(f"Running cycle {self.count}")
            merge_pair = []
            if self.merge_alg == 'pairs':
                # default use greedy for now (original REVO implementation)

                # a list of all suitable pairs is generated and the pair that 
                # minimizes the expected variation loss is chosen.
                pot_merge_pairs = self._find_eligible_merge_pairs(new_walker_weights,
                                                                  distance_matrix, 
                                                                  max_idx, 
                                                                  new_num_walker_copies)

                merge_pair = self._calc_variation_loss(walker_variations, 
                                                       new_walker_weights, 
                                                       pot_merge_pairs)
                
            elif self.merge_alg == 'greedy':
                # walker with the lowest walker_variations (distance to other walkers)
                # will be tagged for merging (stored in min_idx)
                min_tups = [(value, i) for i,value in enumerate(walker_variations)
			    if new_num_walker_copies[i] == 1 and (new_walker_weights[i]  < self.pmax)]

                if len(min_tups) > 0:
                    min_value, min_idx = min(min_tups)

                # does min_idx have an eligible merging partner?
                closewalk = None
                condition_list = np.array([i is not None for i in [min_idx, max_idx]])
                if condition_list.all() and min_idx != max_idx:
                    # get the walkers that aren't the minimum and the max
                    # walker_variations walkers, as candidates for merging
                    closewalks = set(range(num_walkers)).difference([min_idx, max_idx])

                    # remove those walkers that if they were merged with
                    # the min walker_variations walker would violate the pmax
                    closewalks = [idx for idx in closewalks
                                  if (new_num_walker_copies[idx]==1) and
                                  (new_walker_weights[idx] + new_walker_weights[min_idx] < self.pmax)]

                    # if there are any walkers left, get the distances of
                    # the close walkers to the min walker_variations walker if that
                    # distance is less than the maximum merge distance
                    if len(closewalks) > 0:
                        # closewalks_dists = [(distance_matrix[min_idx][i], i) for i in closewalks
                        #                     if distance_matrix[min_idx][i] < (self.merge_dist)]
                        closewalks_dists = [(distance_matrix[min_idx, i], i) for i in closewalks
                                            if distance_matrix[min_idx, i] < (self.merge_dist)]

                    # first check if it even exists as local variable
                    if 'closewalks_dists' in locals():
                        # if any were found set this as the closewalk
                        if len(closewalks_dists) > 0:
                            closedist, closewalk = min(closewalks_dists)
                            merge_pair = [min_idx, closewalk]

            else:
                raise ValueError('Unrecognized value for merge_alg in WEVO')

            # did we find a suitable pair to merge?
            if len(merge_pair) != 0:
                min_idx = merge_pair[0]
                closewalk = merge_pair[1]
                
                # change new_amp
                tempsum = new_walker_weights[min_idx] + new_walker_weights[closewalk]
                new_num_walker_copies[min_idx] = new_walker_weights[min_idx]/tempsum
                new_num_walker_copies[closewalk] = new_walker_weights[closewalk]/tempsum
                new_num_walker_copies[max_idx] += 1

                # re-determine variation function, and walker_variations values
                new_variation, walker_variations = self._calc_variation(new_walker_weights,
                                                                        new_num_walker_copies,
                                                                        distance_matrix)

                # variation increaces and walker to squash not already present 
                # TODO: should it be min_idx or closewalk or both?
                # this works with both but might be better to check after the random step
                # just not sure if that would take care of the infinite opt increase loop
                # I think this current implemetation makes sense, since you won't add to 
                # the merge group walker position if either of the merge pair values are
                # already appended to be merged into each other

                if new_variation > variation and \
                merge_pair[0] not in merge_groups[merge_pair[1]] and \
                merge_pair[1] not in merge_groups[merge_pair[0]]:
        
                #if new_variation > variation:

                    variations.append(new_variation)

                    #logging.info("Variance move to {} accepted".format(new_variation))
                    #logger.info("Variance move to {} accepted".format(new_variation))

                    productive = True
                    variation = new_variation

                    # make a decision on which walker to keep
                    # (min_idx, or closewalk), equivalent to:
                    # `random.choices([closewalk, min_idx],
                    #                 weights=[new_walker_weights[closewalk], new_walker_weights[min_idx])`
                    r = random.uniform(0.0, new_walker_weights[closewalk] + new_walker_weights[min_idx])

                     # keeps closewalk and gets rid of min_idx
                    if r < new_walker_weights[closewalk]:
                        keep_idx = closewalk
                        squash_idx = min_idx

                    # keep min_idx, get rid of closewalk
                    else:
                        keep_idx = min_idx
                        squash_idx = closewalk

                    # update weight
                    new_walker_weights[keep_idx] += new_walker_weights[squash_idx]
                    new_walker_weights[squash_idx] = 0.0

                    # update new_num_walker_copies
                    new_num_walker_copies[squash_idx] = 0
                    new_num_walker_copies[keep_idx] = 1

                    # add the squash index to the merge group
                    merge_groups[keep_idx].append(squash_idx)
                    #print(f"Added squash index {squash_idx} to merge group {keep_idx}:\n", merge_groups)

                    # add the indices of the walkers that were already
                    # in the merge group that was just squashed
                    merge_groups[keep_idx].extend(merge_groups[squash_idx])

                    # reset the merge group that was just squashed to empty
                    merge_groups[squash_idx] = []

                    # increase the number of clones that the cloned walker has
                    walker_clone_nums[max_idx] += 1

                    # new variation for starting new stage
                    new_variation, walker_variations = self._calc_variation(new_walker_weights,
                                                                            new_num_walker_copies,
                                                                            distance_matrix)
                    variations.append(new_variation)

                    #logging.info("variance after selection: {}".format(new_variation))
                    #logger.info("variance after selection: {}".format(new_variation))

                # if not productive
                else:
                    #logger.info("Not productive")
                    new_num_walker_copies[min_idx] = 1
                    new_num_walker_copies[closewalk] = 1
                    new_num_walker_copies[max_idx] -= 1


        #return walker_actions, variations[-1]
        return walker_clone_nums, merge_groups, variations[-1], walker_variations

    def resample(self):
        """
        Resamples walkers based on REVO algorithm (TODO)

        Returns
        -------
        resampled_walkers : list of resampled_walkers
        resampling_data : list of dict of str: value
            The resampling records resulting from the decisions.
        resampler_data :list of dict of str: value
            The resampler records resulting from the resampler actions.
        """ 

        #initialize the parameters
        #num_walkers = len(walkers)
        num_walkers = len(self.pcoords)

        # I'm using self.weights
        #walker_weights = [walker.weight for walker in walkers]
        
        # Needs to be floats to do partial amps during second variation calculations
        num_walker_copies = np.ones(num_walkers)

        # calculate distance matrix
        #distance_matrix, images = self._all_to_all_distance()
        distance_matrix = self._all_to_all_distance()

        logger.info("distance_matrix")
        logger.info("\n{}".format(str(np.array(distance_matrix))))

        # determine cloning and merging actions to be performed, by
        # maximizing the variation, i.e. the Decider
        # resampling_data, variation = self.decide(num_walker_copies, distance_matrix)

        split, merge, variation, walker_variations= self.decide(num_walker_copies, distance_matrix)
        logger.info(f"WEVO ran for {self.count} cycles")
        logger.info(f"\nTo split: {len(split)} total: {np.sum(split)} being split \n {split}")
        logger.info(f"\nTo merge: {len(merge)} total: {np.sum([len(i) for i in merge])} being merged \n {merge}")
        #logger.info(f"Final variation = {variation}\n")
        #logger.info(f"Final variation from walkers len = {len(walker_variations)}\n")

        return split, merge, variation, walker_variations


if __name__ == '__main__':
    
    # using example pcoords and weights from ODLD simulation with default H&K resampler
    # TODO: test 2D and 3D pcoords (should work since distance matrix is Euclidean-based)
    iteration = 75
    print(f"\nTesting ODLD WE Iteration {iteration}")
    import h5py
    f = h5py.File("west_default100.h5", "r")
    pcoords = f[f"iterations/iter_{iteration:08d}"]["pcoord"][:,-1]
    pcoords = pcoords.reshape(-1)
    weights = f[f"iterations/iter_{iteration:08d}"]["seg_index"]["weight"]

    print(f"\nPCOORDS: {len(pcoords)} total \n {pcoords}")
    print(f"\nWEIGHTS: {len(weights)} total \n {weights}\n")

    # initialize wevo
    #resample = WEVO(pcoords, weights, merge_dist=1, char_dist=0.608)
    resample = WEVO(pcoords, weights, merge_dist=1, char_dist=1.13, merge_alg="pairs")

    # got char dist of 0.608
    # dist_matrix = resample._all_to_all_distance()
    # print(np.mean(dist_matrix))

    # run wevo
    split, merge, variation = resample.resample()

    # TODO: no split merge decisions at low ODLD iterations (1-4)
    # seems like initial splitting needed for wevo to start working
    # because: needs larger amount of initial walkers (needs more knobs to optimize)

    # TODO: the total segments are not conserved, should be constant but increases
    # sometimes will split more than merge (for greedy)
    # for pairs, seems to be constant split and merge amounts but the final segments
    # may increase or decrease

    # TODO: check greedy and pairs parents for merging
    # I already checked splitting parents and it all looked okay (so may be a splitting issue)

    """ WEVO
    1. get distance matrix (_all_to_all_distance)
    2. determine split-merge decisions from variance opt (decide)
        - _calc_variation : calcs total V and per-walker v
    """
