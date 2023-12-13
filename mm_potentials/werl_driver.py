import logging

import numpy as np
import operator
from westpa.core.we_driver import WEDriver

from werl import WERL
#from split_test import LCAS_split

############################# SET WERL PARAMETERS #############################


###############################################################################

log = logging.getLogger(__name__)

class WERLDriver(WEDriver):
    '''
    This replaces the Huber & Kim based WE algorithm with a resampling scheme based 
    on reinforcement learning.
    '''

    def _split_decision(self, bin, to_split, split_into):
        '''
        TODO: this removes an extra walker
        '''
        # remove the walker being split
        bin.remove(to_split)
        # get the n split walker children
        new_segments_list = self._split_walker(to_split, split_into, bin)
        # add all new split walkers back into bin, maintaining history
        bin.update(new_segments_list)


    def _merge_decision(self, bin, to_merge, cumul_weight=None):
        '''
        TODO: this adds an extra walker
        '''
        # removes every walker in to_merge 
        # TODO: I think I need to remove the current walker, which isn't in the to_merge list
        bin.difference_update(to_merge)
        #new_segment, parent = self._merge_walkers(to_merge, None, bin)
        new_segment, parent = self._merge_walkers(to_merge, cumul_weight, bin)
        # add in new merged walker
        bin.add(new_segment)


    def _adjust_count(self, ibin):
        '''
        TODO: adjust to sort/adjust by variance, not weight.
        '''
        bin = self.next_iter_binning[ibin]
        target_count = self.bin_target_counts[ibin]
        weight_getter = operator.attrgetter('weight')

        #print("PRINT ATTRS: ", dir(bin))
        #for b in bin:
        #    print(weight_getter(b))

        # split
        while len(bin) < target_count:
            log.debug('adjusting counts by splitting')
            # always split the highest variance walker into two
            segments = sorted(bin, key=weight_getter)
            bin.remove(segments[-1])
            new_segments_list = self._split_walker(segments[-1], 2, bin)
            bin.update(new_segments_list)

        # merge
        while len(bin) > target_count:
            log.debug('adjusting counts by merging')
            # always merge the two lowest variance walkers
            segments = sorted(bin, key=weight_getter)
            bin.difference_update(segments[:2])
            merged_segment, parent = self._merge_walkers(segments[:2], cumul_weight=None, bin=bin)
            bin.add(merged_segment)


    def _run_we(self):
        '''
        Run recycle/split/merge. Do not call this function directly; instead, use
        populate_initial(), rebin_current(), or construct_next().
        '''
        self._recycle_walkers()

        # sanity check
        self._check_pre()

        # dummy resampling block
        # TODO: wevo is really only using one bin
        # ibin only needed right now for temp split merge option (TODO)
        for ibin, bin in enumerate(self.next_iter_binning):
            
            # TODO: is this needed? skips empty bins probably
            if len(bin) == 0:
                continue

            else:
                # this will just get you the final pcoord for each segment... which may not be enough
                segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
                pcoords = np.array(list(map(operator.attrgetter('pcoord'), segments)))
                weights = np.array(list(map(operator.attrgetter('weight'), segments)))
 
                # TODO: how to handle nd arrays > n = 1 ?
                #print("PCOORD SHAPE: ", pcoords.shape)

                # make the pcoord input uniform and >=3d
                # because with more than 1 pcoord, all pcoord arrays will be 3d
                # I think it is already the case where even 1d pcoords are 3d arrays
                # so this might not be necessary
                pcoords = np.atleast_3d(pcoords)

                #nsegs = pcoords.shape[0]
                #nframes = pcoords.shape[1]
                #ndims = pcoords.shape[2]

                # TODO: this will allow you to get the pcoords for all frames
                current_iter_segments = self.current_iter_segments

                curr_segments = np.array(sorted(current_iter_segments, key=operator.attrgetter('weight')), dtype=np.object_)
                curr_pcoords = np.array(list(map(operator.attrgetter('pcoord'), curr_segments)))
                curr_weights = np.array(list(map(operator.attrgetter('weight'), curr_segments)))

                #np.save("pcoords_full.npy", curr_pcoords)
                # pcoord for the last frame of previous iteration
                # or the first frame of current iteration
                pcoords = pcoords[:,0,:]

                #print("weights", weights)
                #print("before pcoords", pcoords)
                #pcoords = pcoords.reshape(nsegs,nframes)
                #print("after pcoords", pcoords)
                
                # print for sanity check
                print("pcoords", pcoords)
                print("weights", weights)
                print("Initial weight sum: ", np.sum(weights))

                # run wevo and do split merge based on wevo decisions
                # TODO: get merge_dista/char_dist/merge_alg/other wevo parameters from west.cfg
                # TODO: pairs seems to not fail with assertion error while greedy tends to fail more often
                # the reason 'greedy' fails is because extra walkers are sometimes created
                # but happens at random so it's hard to track down the root cause
                
                ### 
                # resample = WEVO(pcoords, weights, merge_dist=merge_dist, char_dist=char_dist,
                #                 merge_alg=merge_alg, pmin=pmin, pmax=pmax, dist_exponent=dist_exponent)
                # split, merge, variation, walker_variations = resample.resample()
                #resample = WERL(pcoords, n_clusters=5)
                resample = WERL(curr_pcoords, n_clusters=None)
                split, merge = resample.lcas()
                #split, merge = LCAS_split(pcoords, weights)

                print(f"LCAS split: {split}\nLCAS merge: {merge}")
                #print(f"Final variation value after {resample.count} wevo cycles: ", variation)

                # count each operation and segment as a check
                segs = 0
                splitting = 0
                merging = 0

                # go through each seg and split merge
                for i, seg in enumerate(segments):
                    
                    # split or merge on a segment-by-segment basis
                    if split[i] != 0:
                        #self._split_by_wevo(bin, seg, split[i])
                        # need an extra walker since split operation reduces total walkers by 1
                        # I think revo doesn't count the current seg
                        self._split_decision(bin, seg, split[i] + 1)
                        splitting += split[i] + 1
                        #splitting += split[i]
                    if len(merge[i]) != 0:
                        # list of all segs objects in the current merge list element
                        to_merge = [segment for num, segment in enumerate(segments) if num in merge[i]]
                        # adding current segment to to_merge list
                        # I think revo doesn't count the current seg
                        to_merge.append(seg)
                        
                        # cumul_weight should be the total weights of all the segments being merged
                        # cumul_weight is calculated automatically if not given
                        self._merge_decision(bin, to_merge)
                        merging += len(to_merge)
                    
                    segs += 1

                print("Bin attrs post WERL: ", self.next_iter_binning[ibin])
                #print(f"Final variation from walkers len = {len(walker_variations)}\n")
                # make bin target count consistent via splitting high weight and merging low weight
                # TODO: maybe do this with variance sorting instead of weight sorting?
                # this shouldn't be required now that wevo is consistent in total walkers
                if self.do_adjust_counts:
                    self._adjust_count(ibin)

                print(f"Total = {segs}, splitting = {splitting}, merging = {merging}")

                # TODO: print to check that min and max prob are being controlled by wevo
                # west.log has this but I should have another check printed

                # TODO: temp fix for no initial splitting needed for wevo to start working
                # i.e. no wevo cycles were able to run
                # if resample.count == 1:
                #     # running H&K resampling
                #     target_count = self.bin_target_counts[ibin]
                #     ideal_weight = weights.sum() / target_count
                #     self._split_by_weight(bin, target_count, ideal_weight)
                #     self._merge_by_weight(bin, target_count, ideal_weight)
                #     # if self.do_adjust_counts:
                #     #     self._adjust_count(bin)
                

        # another sanity check
        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))
