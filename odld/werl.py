## test.py
import numpy as np
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors

class WERL:

    def get_data(file):
        f = open(file, 'r')
        lines = f.readlines()
        f.close()
        return [float(line.rstrip()) for line in lines]

    def get_split(pcoords, pcoords_to_split):
        split=[]
        for i in range(len(pcoords)):
            pcoord = pcoords[i]
            if pcoord in pcoords_to_split:
                split.append(1)
            else:
                split.append(0)
        return split

    def get_merge(pcoords, merge_indices):
        merge=[]
        merge_into = list(merge_indices.keys())
        for i in range(len(pcoords)):
            if i in merge_into:
                merge.append([merge_indices[i]])
            else:
                merge.append([])
        return merge

    def merge_decision(pcoords, pcoords_to_merge, weights):
        merge_id = []
        for i in range(len(pcoords)):

            ## Get weight for each traj to be merged
            pcoord = pcoords[i]
            if pcoord in pcoords_to_merge:
                weight = weights[i]
                merge_id.append((i, weight))

        ## Randomly gen pairs of trajs to be merged
        ## each pair is a tuple of tuples
        pairs = []
        while merge_id:
            print(merge_id)
            traj1 = merge_id.pop(random.randrange(0, len(merge_id)))
            traj2 = merge_id.pop(random.randrange(0, len(merge_id)))
            pairs.append((traj1, traj2))

        ## Normalize each weight
        merge_indices = {}
        for pair in pairs:
            traj1, traj2 = pair[0], pair[1]
            w1, w2 = traj1[1], traj2[1]
            w1_norm = w1/(w1+w2)
            w2_norm = w2/(w1+w2)

            ## Get merge decision probabilistically
            rand = np.random.uniform(0, 1.0)
            if w1_norm < w2_norm:
                if rand <= w1_norm:
                    to_merge = traj2
                    merged_into = traj1
                else:
                    to_merge = traj1
                    merged_into = traj2
            else:
                if rand <= w2_norm:
                    to_merge = traj1
                    merged_into = traj2
                else:
                    to_merge = traj2
                    merged_into = traj1
            
            to_merge_ind = to_merge[0]
            merged_into_ind = merged_into[0]

            ## x merged to y
            merge_indices[merged_into_ind] = to_merge_ind

        return merge_indices

    def print_cycle(split_arr, merge_arr, num_traj):
        print('-----------------------------------------------------------------------------------')
        print("To split:", num_traj, "total:", np.sum(split_arr), "being split.")
        print(split_arr)
        print("To merge:", num_traj, "total:", np.sum([len(m) for m in merge_arr]), "being merged.")
        print(merge_arr)

    def test_function(pcoord_txt, weight_txt):
        pcoords = get_data(pcoord_txt)
        weights = get_data(weight_txt)
        split, merge = xdim_split(pcoords, weights, printout=True)
        return split, merge

    def plot_on_odld(pcoords, labels, centers, A=2, B=10, C=0.5, x0=1, show=False, savename=""):
        twopi_by_A = 2 * np.pi / A
        half_B = B / 2
        sigma = 0.001 ** (0.5)
        gradfactor = sigma * sigma / 2
        reflect_at = 10

        x = np.arange(0.75,10.1,0.05)
        xarg = twopi_by_A * (x - x0)

        eCx = np.exp(C * x)
        eCx_less_one = eCx - 1.0
        y = half_B / (eCx_less_one * eCx_less_one) * (twopi_by_A * eCx_less_one * np.sin(xarg) + C * eCx * np.cos(xarg))

        plt.rcParams['figure.figsize'] = [20,10]
        plt.rcParams['font.size'] = 20
        plt.plot(x, y, color='k', alpha=0.5, label='ODLD potential')
        #plt.plot(9.5,0,marker='*',color='green',markersize=15)
        plt.axvline(reflect_at,color='k', ls='--', label='Reflecting barrier')
        #plt.axvline(9.5,color='green',alpha=0.5)
        plt.ylabel('gradient')
        plt.xlabel('x-coordinate')
        #plt.text(8.1,10,'reflecting barrier ->', color='red')
        #plt.text(8.9,-3,'basis state', color='green')

        pcoords = [float(p) for p in pcoords]
        colors = ['blue','red','green','magenta','orange']

        plt.scatter(pcoords, [0 for p in range(len(pcoords))], c=labels, cmap=matplotlib.colors.ListedColormap(colors), marker='x')
        for i in range(len(centers)):
            plt.axvline(centers[i], color=colors[i])
        plt.legend()
        if savename != "":
            plt.savefig(savename)
        if show:
            plt.show()

    def get_max_min_cluster(clust_dict, labels):
        max_cluster, max_id = 0, 0
        min_cluster, min_id = len(labels), 0
        for key in clust_dict.keys():
            if len(clust_dict[key]) > max_cluster:
                max_cluster = len(clust_dict[key])
                max_id = key
            elif len(clust_dict[key]) < min_cluster:
                min_cluster = len(clust_dict[key])
                min_id = key
        return max_id, min_id

    ### Splitting and merging functions ###
    def xdim_split(pcoords, weights, max_to_split=5, printout=False):

        num_to_merge = 2*max_to_split
        num_traj = len(pcoords)

        ## Split by pcoord value
        pcoords_to_split = sorted(pcoords)[:max_to_split]
        pcoords_to_merge = sorted(pcoords)[-num_to_merge:]

        split_arr = get_split(pcoords, pcoords_to_split)
        merge_indices = merge_decision(pcoords, pcoords_to_merge, weights)

        merge_arr = get_merge(pcoords, merge_indices)
        if printout:
            print(merge_indices)
            print_cycle(split_arr, merge_arr, num_traj)

        return split_arr, merge_arr

    def LCAS_split(pcoords, weights, max_to_split=5, printout=False, saveplots=False):
        
        MAX_TO_SPLIT = 5
        NUM_TO_MERGE = 2*MAX_TO_SPLIT
        NUM_TRAJ = len(pcoords)
        N_CLUSTERS = 5

        ## Perform Kmeans clustering on pcoords
        pcoords = pcoords.reshape(-1,1)
        k_clust = KMeans(n_clusters=N_CLUSTERS).fit(pcoords)
        centers = k_clust.cluster_centers_
        labels = k_clust.labels_

        num_to_merge = 2*max_to_split
        num_traj = len(pcoords)

        ## Initialize dictionary
        clusters = {}
        for i in range(N_CLUSTERS):
            clusters[i] = []

        for i in range(len(pcoords)):
            clusters[labels[i]].append(float(pcoords[i]))
        
        ## Evaluate clustering success - plot clusters on ODLD
        if saveplots:
            plot_on_odld(pcoords, labels, centers)

        ## Get the size of max and min clusters and their respective IDs
        max_id, min_id = get_max_min_cluster(clusters, labels)
        max_c_size, min_c_size = len(clusters[max_id]), len(clusters[min_id])

        print('Total number of trajectories:', len(pcoords))
        print('Most populated:', max_id, 'with size', len(clusters[max_id]))
        print('Least populated:', min_id, 'with size', len(clusters[min_id]))


        ### Get split and merge arrays ###

        ## Get pcoords to split from smallest cluster
        _pcoords_to_split = clusters[min_id]
        if min_c_size > max_to_split:
            pcoords_to_split = []
            for i in range(max_to_split):
                pcoord = _pcoords_to_split.pop(random.randrange(0, len(_pcoords_to_split)))
                pcoords_to_split.append(pcoord)
        else:
            pcoords_to_split = _pcoords_to_split

        ## Get pcoords to merge from largest cluster
        _pcoords_to_merge = clusters[max_id]
        
        # Only set (n_split*2) trajs to be merged
        # unless only n<N-split trajs to be split
        # in which case we only merge n*2
        if min_c_size < max_to_split:
            pcoords_to_merge = []
            for i in range(min_c_size*2):
                pcoord = _pcoords_to_merge.pop(random.randrange(0, len(_pcoords_to_merge)))
                pcoords_to_merge.append(pcoord)
        else:
            pcoords_to_merge = []
            for i in range(max_to_split*2):
                pcoord = _pcoords_to_merge.pop(random.randrange(0, len(_pcoords_to_merge)))
                pcoords_to_merge.append(pcoord)
            
        split_arr = get_split(pcoords, pcoords_to_split)
        merge_indices = merge_decision(pcoords, pcoords_to_merge, weights)

        merge_arr = get_merge(pcoords, merge_indices)
        if printout:
            print(merge_indices)
            print_cycle(split_arr, merge_arr, num_traj)

        return split_arr, merge_arr

if __name__ == "__main__":
    
    ### 1 iteration using test data ###

    ## Get pcoords, weights from test data
    pcoords = np.array(get_data('pcoords.txt'))
    weights = np.array(get_data('weights.txt'))

    MAX_TO_SPLIT = 5
    NUM_TO_MERGE = 2*MAX_TO_SPLIT
    NUM_TRAJ = len(pcoords)
    N_CLUSTERS = 5

    ### Test xdim_split with test data ###
    split_arr, merge_arr = xdim_split(pcoords, weights, max_to_split=MAX_TO_SPLIT, printout=True)


    ### Test LCAS with test data ###
    split_arr, merge_arr = LCAS_split(pcoords, weights, max_to_split=MAX_TO_SPLIT, printout=True)


    ### Plot 100th LCAS iteration to see the clustering
    f = open('LCAS_i100_pcoords.txt', 'r')
    lines = f.readlines()
    f.close()
    pcoords = []
    for i in range(len(lines)):
        pcoords.append(float(lines[i].rstrip()))

    pcoords=np.array(pcoords)
    pcoords = pcoords.reshape(-1,1)
    k_clust = KMeans(n_clusters=N_CLUSTERS).fit(pcoords)
    centers = k_clust.cluster_centers_
    labels = k_clust.labels_
    plot_on_odld(pcoords, labels, centers, show=True, savename='kmeans_ODLD_i100')
