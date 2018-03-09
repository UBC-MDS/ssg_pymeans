import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ssg_pymeans.data import default_data

class InvalidInput(Exception):
    pass

class Pymeans:
    def __init__(self, data=None):
        if (data is None):
            self.data = default_data
        else:
            if (isinstance(data, pd.DataFrame)):
                self.data = data
            else:
                raise InvalidInput('Input must be pandas data frame.')

    def init_cent(self, k=2, method='random'):
        """Centroids initialization using random or kmeans++ method.

        Args:
            k (int): Number of clusters.
            method (str): Name of the initialization method.
        Returns:
            list: k tuples of centroids
        """
        if (not isinstance(k, int)):
            raise InvalidInput('k must be an integer.')

        if (k <= 0 or k > self.data.shape[0]):
            raise InvalidInput('Out of bound error: k must be larger than \
                zero and smaller than the number of rows')

        if method == 'random':
            # ramdomly pick initial points
            centroids = np.random.randint(self.data.shape[0], size=k)
            return centroids
        elif method == 'kmpp':
            # use kmeans++ to pick initial points
            pass
        else:
            raise InvalidInput('Invalid initialization method.')

    def euc_dist(self, p1, p2):
        """Euclidean distance between any two points"""
        return np.linalg.norm(p1 - p2)

    def should_stop(self, centroids, centroids_new, eps):
        """Check if converged."""
        c0 = centroids.reset_index(drop=True)
        c1 = centroids_new.reset_index(drop=True)
        diff = c0.subtract(c1)
        diff = diff.abs()
        if np.max(diff.max(axis=1).values) <= eps:
            return True
        return False

    def tot_wss(self, data):
        """Calculate the total within cluster sum of square error."""
        tot_wss = 0
        def ed(row, centroid):
            return np.linalg.norm(row.values[0:1] - centroid.values)
        for name, cluster in data:
            centroid = cluster.mean()
            centroid = centroid[['x1', 'x2']]
            dist = cluster.apply(ed, centroid=centroid, axis=1)
            tot_wss = tot_wss + dist.sum()
        return tot_wss

    def fit(self, K, method='random'):
        """Compute k-means clustering
        Returns:
            dictionary: Contains
                        1. pandas data frame of the attributes and clustering for each data point
                        2. total within cluster sum of square and
                        3. pandas data frame of k centroids
        """
        data = self.data[['x1', 'x2']]  # raw
        nobs = data.shape[0]  # number of samples

        # get centroids
        cent_init = self.init_cent(K, method)
        centroids = data.iloc[cent_init, [0,1]]
        centroids = centroids.reset_index(drop=True)

        dist_mat = np.zeros((nobs, K))  # for calculation
        labels = []
        eps = 0
        n_iter = 0
        max_iter = 20
        stop = False

        while (not stop):
            labels = []
            for row in range(nobs):
                for k in range(K):
                    dist_mat[row, k] = self.euc_dist(data.iloc[row,[0,1]],
                                                     centroids.iloc[k,[0,1]])
                idx_min = np.argmin(dist_mat[row,])
                labels.append(str(int(idx_min) + 1))

            # group data based on labels
            data['cluster'] = labels

            centroids_new = data.groupby('cluster').mean()

            if (self.should_stop(centroids, centroids_new, eps) or (n_iter > max_iter)):
                stop = True

            centroids = centroids_new
            n_iter = n_iter + 1

        print('kmeans converged in %i runs' % n_iter)

        tot_withinss = self.tot_wss(data.groupby('cluster'))

        res = {
            'data': data,
            'tot_withinss': tot_withinss,
            'centroids': centroids
        }

        return res

    def predict(self, data_new, centroids):
        """Predict k-means clustering for new data frame
        Returns:
            dataframe: Contains
                       1. new data
                       2. clustering label for each data point
        """
        data = data_new
        def calc_dist(row, dat): # row is centroids
            return np.linalg.norm(row - dat)

        def assign_cluster(row): # row is data here
            dist = centroids.apply(calc_dist, dat=row, axis=1)
            label = np.argmax(dist.values) + 1
            return str(label)

        data['cluster'] = data.apply(assign_cluster, axis=1)
        return data

    def input_shape_validation(self):
        """Utility function checking input data shape for kmplot.

        Returns
            bool: True if shape is valid. False otherwise.
        """
        if (not self.data.shape or
                len(self.data.shape) <= 1 or
                self.data.shape[0] <= 0 or
                self.data.shape[1] < 3):
            return False
        return True

    def input_label_validation(self):
        """Utility function checking if the cluster column exists for kmplot.

        Returns:
            bool: True if the cluster column exists. False otherwise.
        """
        if (not 'cluster' in self.data.columns):
            return False
        return True

    def kmplot(self):
        """Visualize kmeans results in a scatter plot.

        Returns:
            matplotlib.lines.Line2D: plot showing the scatter plot of kmeans
                                     results, colored by clusters.

        Raises:
            InvalidInput: If self.data has zero row, less than three columns,
                          or no cluster column.
        """
        if not self.input_shape_validation():
            raise InvalidInput('Input must have at least one row and \
                three columns.')
        if not self.input_label_validation():
            raise InvalidInput('No cluster labels. Run fit first before plot.')

        fig = plt.plot(self.data.iloc[:,0], self.data.iloc[:,1], '.')

        return fig
