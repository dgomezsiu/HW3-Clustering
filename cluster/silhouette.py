import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # find the number of clusters (largest number in y + 1)
        count_clusters = np.amax(y) + 1
        # labels are the numbers of clusters
        labels = np.arange(0,count_clusters)

        # find the intra cluster distance
        count_observations, count_labels = X.shape
        # initialize an array to store intra cluster disatnces for each observation
        intra_dist = np.zeros(count_observations)

        # for every label calculate cdist
        for cluster in np.unique(labels):
            # calculate euclidean distances for this cluster
            this_cluster = X[cluster == y]
            distances = cdist(this_cluster, this_cluster, metric = 'euclidean')
            # assign average distance to intra distance by cluster
            indices = np.where(cluster == y)
            intra_dist[indices] = np.sum(distances, axis = 1) / (distances.shape[0] - 1)

        # find inter cluster distances from a point to other clusters
        inter_dist = np.zeros(count_observations)
        for cluster in np.unique(labels):
            # only look at non cluster clusters
            this_cluster = X[cluster == y]
            indices = np.where(cluster == y)
            other_clusters = np.delete(y, indices)
            # initiate an array to store distances
            distances = np.full(this_cluster.shape[0], np.inf)
            #take the smallest average distance for each point to other cluster
            for inter_cluster in np.unique(other_clusters):
                that_cluster = X[inter_cluster == y]
                inter_distance = cdist(this_cluster, that_cluster, metric = 'euclidean')
                average_distances_per_cluster = np.mean(inter_distance, axis = 1)
                distances = np.minimum(average_distances_per_cluster, distances)
            
            inter_dist[indices] = distances
        return (inter_dist - intra_dist) / (np.maximum(inter_dist, intra_dist))