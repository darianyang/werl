import numpy as np

lc_clusters = np.array([3, 1, 4, 0, 2])
rewards = np.array([1.12667526, 1.23873339, 0.64157923, 1.22793741, 0.61519596])
cluster_counts = np.array([[2, 15], [0, 8], [4, 7], [1, 5], [3, 5]])

# Sort lc_clusters and rewards in descending order based on the values in rewards
sorted_indices = np.argsort(rewards)
lc_clusters = lc_clusters[sorted_indices]
print(lc_clusters)
rewards = rewards[sorted_indices]
print(rewards)

# Insert lc_clusters as a new column before the first column of cluster_counts
cluster_counts = np.insert(cluster_counts, 0, lc_clusters, axis=1)

print(cluster_counts)
