# WERL - Bowman FAST algorithm
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import pairwise_distances_argmin_min
from werl.werl_base import WERL
from math import ceil


class FAST(WERL):
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
    
    def directed_reward(self, minimize_val=True):
        '''
        Get the scaled values for the directed component of the reward function.

        minimize_val: Boolean for assigning whether the pcoord is to be minimized or maximized
        Example: The target for the 1D ODLD is a small x, so here we are minimizing

        Returns a tuple (reward, cluster id, centroid pcoord id)
        '''
        closest, _ = pairwise_distances_argmin_min(self.centers, self.pcoords)
        state_vals = []
        for close in closest:
            centroid_val = self.pcoords[close]
            state_vals.append(centroid_val) # Get the centroids for each state
        
        # Scaling values - requires the program to know if the value should be minimized or maximized
        # These values are the directed component of the reward for each state
        # Scales from 0 (closest to target) to 1 (furthest from target)
        dim_state_vals = np.array(state_vals).T  # state centroid vals per pcoord
        state_mins, state_maxs = [np.min(d) for d in dim_state_vals], [np.max(d) for d in dim_state_vals]

        # Reward calculation - for each pcoord, calc the scaled rewards for each state
        _scaled_rewards = []
        for i in range(len(dim_state_vals)):
            state_min, state_max = state_mins[i], state_maxs[i] # min and max for each pcoord across all state centroids
            if minimize_val:
                _scaled_rewards.append([(state_max-val)/(state_max-state_min) for val in dim_state_vals[i]])
            else:
                _scaled_rewards.append([(val-state_min)/(state_max-state_min) for val in dim_state_vals[i]])
        
        # Need to have a single reward value per state
        # Here, we simply sum the individual rewards per pcoord per state
        # Then we normalize the rewards in the same way as before
        if self.pcoords.shape[-1] > 1:
            scaled_rewards = [np.sum(t) for t in np.array(_scaled_rewards).T]
            state_min, state_max = np.min(scaled_rewards), np.max(scaled_rewards) # The state values used are the centroid values

            if minimize_val:
                scaled_rewards = [(state_max-val)/(state_max-state_min) for val in scaled_rewards]
            else:
                scaled_rewards = [(val-state_min)/(state_max-state_min) for val in scaled_rewards]
        
        # However, if we don't have multiple pcoords, we should not do this - we already have the dir reward component
        else:
            scaled_rewards = _scaled_rewards[0] # only one list to get, but stored as list of lists for higher dim cases
            print(scaled_rewards)

        # Traceback to the centroid and label of the cluster (state)
        state_reps = []
        for i in range(len(closest)):
            centroid_val = self.pcoords[closest[i]]
            centroid_index = np.where(self.pcoords == centroid_val)[0][0] # Gets the index for the n-dim centroid value in pcoords
            label = self.labels[centroid_index]
            state_reps.append((float(scaled_rewards[i]), label, centroid_index))
        
        return state_reps 
    
    def undirected_reward(self):
        '''
        Get the scaled values for the undirected component of the reward function.

        Returns a tuple (reward, cluster id)
        '''
        ## Get min and max instances across all states, as well as each state's instances
        ## TODO: What to do in the case of a tie? 
        ## TODO: Undirected reward only gives 10-11 clusters when 20 requested
        ### This is because having high number of clusters generates non-unique clusters
        ### Does this add artificial reward to these clusters? They all will be 1.0 undirected reward
        ### It may be the best case to set these rewards to zero, since they are technically already represented
        ### Something to revisit later, because it really only happens in the first iteration

        clust_and_counts = np.unique(self.labels, return_counts=True)

        clusts = list(clust_and_counts[0])
        clustcounts = list(clust_and_counts[1])

        # If there are not enough unique cluster centers, "make" unique empty clustere
        # They are still there, but np.unique misses them because walkers will not be assigned to the non-unique clusters
        if len(clust_and_counts[0]) != self.n_clusters:
            for i in range(clusts[-1]+1, self.n_clusters):
                clusts.append(i)
                clustcounts.append(0)

        clust_and_counts = [clusts, clustcounts]
        counts = [c for c in clust_and_counts[1]]
        min_count, max_count = np.min(counts), np.max(counts)

        ## Scaling values - need to put on same scale as the directed compounent
        ## Scales from 0 (most sampled) to 1 (least sampled)
        scaled_rewards = [(max_count-c)/(max_count-min_count) for c in counts]

        state_reps = []
        for i, reward in enumerate(scaled_rewards):
            state_reps.append((reward, clust_and_counts[0][i]))

        return state_reps

    def compute_reward(self, directed, undirected, alpha=1):
        '''
        Computes the reward for each state using scaled directed (phi) and undirected (psi) components
        R(i) = phi(i) + alpha * psi(i)

        alpha: float for scaling the value of undirected reward component
        Here, alpha=1 is the default - no scaling/balancing performed for undiredcted component
        '''

        print('LABELS:', self.labels)
        dir_rewards = [r[0] for r in directed]
        undir_rewards = [r[0] for r in undirected]

        rewards = []

        for i in range(len(dir_rewards)):
            rewards.append(dir_rewards[i] + alpha*undir_rewards[i])

        return rewards

    def do_prop_splitting(self, rewards):
        '''
        Do splitting based on the reward prop for each state to total reward

        Updates self.to_split
        '''

        ## Get each state's reward proportion
        total_reward = np.sum(rewards)
        reward_props = [float("{:.3f}".format(r/total_reward)) for r in rewards]
        num_to_split_frac = [r*self.n_split for r in reward_props]

        ## Decide num to split from each cluster based on reward proportion
        # Here, just round up or down
        num_to_split = []
        for n in num_to_split_frac:
            if (n - int(n)) >= 0.5:
                n = ceil(n)
            else:
                n = int(n)
            num_to_split.append(n)

        ## Get and sort the reward indices
        reward_w_ind = [(i, r) for i, r in enumerate(reward_props)]
        sorted_reward_indices = [r[0] for r in sorted(reward_w_ind, key=lambda x:x[1], reverse=True)]
        # print('SORTED REWARD INDICES:', sorted_reward_indices)

        # print('TOTAL NUM TO SPLIT:', self.n_split)
        # print('SUM OF NUM TO SPLIT:', np.sum(num_to_split), '\n')
        # print('CALCULATED NUM TO SPLIT:', num_to_split) 
        

        ## Make necessary corrections to num_to_split
        # If more splits are calculated for a cluster than members, 
        # redistribute splits to the next highest reward cluster
        print()
        print("CORRECTING SPLITS PER CLUSTER AFTER REWARD CALCULATION")
        print("======================================================")
        _, label_counts = np.unique(self.labels, return_counts=True)
        for i, n in enumerate(sorted_reward_indices):
            while num_to_split[n] > label_counts[n]:
                print('Reducing num_to_split for Cluster', str(n)+':', num_to_split[n], '->', num_to_split[n]-1)
                num_to_split[n] -= 1
                # Prioritize redistributing to high reward clusters
                for i in range(len(sorted_reward_indices)):
                    r_ind = sorted_reward_indices[i]
                    if num_to_split[r_ind] < label_counts[r_ind]:
                        print('Increasing num_to_split for Cluster', str(r_ind)+':', num_to_split[r_ind], '->', num_to_split[r_ind]+1)
                        num_to_split[r_ind] += 1 
                        break

        ## Then correct when sum(num_to_split) is not equal to self.n_split, again prioritize adding to high reward clusters
        while np.sum(num_to_split) != self.n_split:
            if np.sum(num_to_split) > self.n_split:
                # Remove a split from the lowest reward cluster
                for num in sorted_reward_indices[::-1]:
                    if num_to_split[num] > 0:
                        num_to_split[num] -= 1
                        print("Removed a split from Low Reward Cluster", num)
                        break
            elif np.sum(num_to_split) < self.n_split:
                # Add a split to the highest reward cluster
                for num in sorted_reward_indices:
                    if num_to_split[num] < np.unique(self.labels, return_counts=True)[1][num]:
                        num_to_split[num] += 1
                        print("Added a split to High Reward Cluster", num)
                        break

        ## Select splits randomly from each cluster n times, for each n in num_to_split
        splits_remaining = self.n_split

        # This is a dictionary that relates state labels to the number of segments with weights > pmin
        # I use this to redistribute the splits and avoid errors - can surely be improved
        avail_splits = {i:0 for i in np.unique(self.labels)}
        for i, seg_label in enumerate(self.labels):
            if self.weights[i] > self.pmin:
                avail_splits[self.labels[i]] += 1

        # Here, we can then control for weight threshold by modifying num_to_split before splitting
        print()
        print("CORRECTING SPLITS PER CLUSTER IF WEIGHT THRESHOLD IS SET")
        print("========================================================")
        diff_counter = []
        for i in range(len(num_to_split)):
            diff_counter.append(list(avail_splits.values())[i] - num_to_split[i])

        # diff counter tells us if we are splitting too many from each cluster after correcting for weight
        # diff counter is sorted by ascending cluster ids, same as num_to_split, so the comparison is done pairwise
        print("Per cluster difference between number of segments in cluster and splits demanded from cluster:")
        print(diff_counter)
        if np.sum(diff_counter) == np.sum([np.abs(i) for i in diff_counter]):
            print("No changes are needed in this iteration.")

        for i, diff in enumerate(diff_counter):
            avail_vals = [val for val in avail_splits.values()]
            counter = 0 # This will track the high reward cluster that we try to redistribute splits into, reset for each
            while diff < 0:
                this_high_cluster = sorted_reward_indices[counter] 

                # If a split can be made from this high(est) reward cluster, move the split to that cluster and update all
                if diff_counter[this_high_cluster] > 0 and i != this_high_cluster: 
                    num_to_split[i] -= 1 # remove a split from the problematic cluster
                    num_to_split[this_high_cluster] += 1 # Add a split to this high reward cluster
                    diff += 1 
                    avail_vals[this_high_cluster] = avail_vals[this_high_cluster] - 1 
                    diff_counter[this_high_cluster] = diff_counter[this_high_cluster] - 1 
                    print("Moving a split from cluster "+ str(i+1)+" to high reward cluster "+str(this_high_cluster+1)+" (rank "+str(counter+1)+")")
                    print("Updated num_to_split:", num_to_split)
                else:
                    counter += 1 # If a split cannot be made in this high(est) reward cluster, move to next highest
                    print("No additional splits possible in cluster "+str(this_high_cluster+1)+"(rank "+str(counter+1)+") - Attempt next cluster")
            
        print('CORRECTED NUM TO SPLIT:', num_to_split) 
        print('CORRECTED SUM OF NUM TO SPLIT:', np.sum(num_to_split))
        print("========================================================")
        while splits_remaining > 0:
            # go through each segment index label and tag to split
            # no need to "randomize" selection - there is no specific ordering relating to the cluster labels
            for i, seg_label in enumerate(self.labels):
                # if there are still splits to make in the cluster
                if num_to_split[seg_label] > 0:
                    # check if weight is above min threshold 
                    if self.weights[i] > self.pmin and self.to_split[i] == 0:
                        # increase split counter on this segment by 1
                        self.to_split[i] += 1
                        # note that one of the requested splits is done
                        splits_remaining -= 1
                        # update num_to_split
                        num_to_split[seg_label] -= 1
                        # update avail_splits dict
                        avail_splits[self.labels[i]] -= 1    

                # extra precaution
                if splits_remaining == 0:
                     break

    def do_prop_merging(self, rewards):
        '''
        Merge segments, prioritizing merging starting from the lowest reward clusters
        TODO: implement pmax that disallows merging segments with p > pmax

        Updates self.to_merge
        '''

        total_reward = np.sum(rewards)
        reward_props = [r/total_reward for r in rewards]
        reward_w_ind = [(i, r) for i, r in enumerate(reward_props)]
        sorted_reward_indices = [r[0] for r in sorted(reward_w_ind, key=lambda x:x[1], reverse=True)]

        merges_remaining = self.n_split
        low_reward_cluster_counter = -1  # Start looking for merge pairs in the lowest reward cluster

        # make sure there isn't redundant merge pairs
        # using set since sets are unordered, unchangable, with no duplicates
        pairs_merged = set()

        # keep going until no more merges needed or until no more cluster labels avail
        # while merges_remaining > 0 and lc_cluster_counter < self.n_clusters:
        while merges_remaining > 0:
            
            # go through each segment index label and tag to merge
            for i, segi_label in enumerate(self.labels):
                skip_i_seg = False
                for pair in pairs_merged:
                    if i in pair:
                        skip_i_seg = True
                        break
                if skip_i_seg:
                    continue

                # if the segment is in the least count cluster
                if segi_label == sorted_reward_indices[low_reward_cluster_counter]:
                    
                    # now find a merge pair/partner (only merge within same cluster)
                    for j, segj_label in enumerate(self.labels):
                        
                        # if the segment is in the same least count cluster but not same seg index as i
                        if segj_label == sorted_reward_indices[low_reward_cluster_counter] and i != j:
                            
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
            # to merge, then do it again using the next lowest reward cluster label count
            low_reward_cluster_counter -= 1

    def plot_clustering(self):
        # This is mainly for testing the clustering, I do not see this being used in final version
        if self.pcoords.shape[-1] != 2:
            return
        import matplotlib.pyplot as plt
        colors = ['black', 'green']
        sizes = [5, 60]
        color_points = []
        size_points = []
        for i in range(len(self.to_split)):
            color_points.append(colors[self.to_split[i]])
            size_points.append(sizes[self.to_split[i]])
       
        # Really dirty but this is only for visualizing    
        for i in range(len(self.to_split)):
            if self.to_merge[i] != []:
                to_merge = self.to_merge[i]
                for m in to_merge:
                    color_points[m] = 'purple'
                    size_points[m] = 30

        plt.figure()
        plt.scatter(self.pcoords.T[0], self.pcoords.T[1], c=color_points, s=size_points, alpha=0.5)
        plt.scatter(self.pcoords.T[0], self.pcoords.T[1], s=5, c='k')
        plt.show()

    def fast(self):
        '''
        Currently, this only works with a 1D pccord, i.e. 1D ODLD
        '''
        
        if len(np.unique(self.pcoords)) == 1:
            return self.to_split, self.to_merge

        ## Kmeans clustering
        self._clustering()
        
        ## Get rewards
        dir_state_rewards = self.directed_reward()
        undir_state_rewards = self.undirected_reward()

        ## Calculate overall reward per state
        rewards = self.compute_reward(dir_state_rewards, undir_state_rewards, alpha=1)
        print('TOT REWARD:', rewards)

        # if less than the requested amount of clusters was generated
        # then go with zero array return, currently useful for w_init
        # basically, if the clustering fails, do no split/merge operations
        if len(self.counts) < self.n_clusters:
            return self.to_split, self.to_merge


        # run split merge methods, updating split/merge lists
        self.do_prop_splitting(rewards) 
        self.do_prop_merging(rewards)

        # can plot the clusters with split and merge decisions - ONLY 2D
        # self.plot_clustering()

        return self.to_split, self.to_merge



if __name__ == "__main__":

    ## 2D TEST SET
    pcoords = np.load("test_data/pcoords_full.npy")
    weights = np.ones(len(pcoords)) / len(pcoords)
    
    print(f"pcoord full shape: {pcoords.shape}")

    werl = FAST(pcoords, weights)
    fast = werl.fast()


