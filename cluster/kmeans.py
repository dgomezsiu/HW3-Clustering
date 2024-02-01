import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        #doing the basic error handling

        # if k is not positive, error
        if k <= 0: raise Exception("k must be positive")

        # if k is not an integer, error
        if type(k) != int: raise Exception("k must be an integer")

        # initialize attributes for the class
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # assign number of observations and features from shape of input matrix
        self.observations, self.features = mat.shape

        # if the number of clusters is greater than number of observations, error
        if self.k > self.observations: raise Exception("number of clusters exceeds number of observations")

        # start at k random points in observations for centroid
        self.centroids = mat[np.random.choice(self.observations, self.k, replace = False)]

        # initialize prediction matrix
        self.predicted_labels = np.zeros((self.observations, 1))

        # initialize iterator tracker at 0 and error at inf
        iter = 0
        error = np.inf

        # loop through to minimize error while number of iterations is below max iter and error is above tolerance
        while iter < self.max_iter and error > self.tol:
            # assign predicted labels
            self.predicted_labels = self.predict(mat)
            # store previous centroids and update centroids
            self.previous_centroids = self.centroids
            self.centroids = self.get_centroids(mat)
            # update the error and iterator
            error = self.get_error()
            iter += 1



    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        # use Euclidian distance between a point and the closest centroid. take the minimum
        return np.argmin(cdist(mat, self.centroids, metric = 'euclidean'), axis = 1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        # returns the sum of the square mean error of the previous centroids and current centroids
        return np.sum(np.square(self.previous_centroids - self.centroids))

    def get_centroids(self, mat: np.ndarray) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        # initialize centroid matrix shape k, features
        fit_centroids = np.zeros((self.k, self.features))

        # for each cluster, return the average of the poitns at each label
        for cluster in range(self.k):
            fit_centroids[cluster, :] = np.mean(mat[cluster == self.predicted_labels, :], axis = 0)

        return fit_centroids