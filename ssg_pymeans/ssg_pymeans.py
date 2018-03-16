import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from ssg_pymeans.data import default_data

class InvalidInput(Exception):
    pass

class Pymeans:
    def __init__(self, data=None):
        if (data is None):
            self.data = default_data[['x1', 'x2']]
        else:
            if (isinstance(data, pd.DataFrame)):
                if ((not data.shape[1] == 2) or
                        (data.shape[0] < 1)):
                    raise InvalidInput('Input error: Input must be a data frame and \
                        have at least one row and two columns.')
                else:
                    try:
                        self.data = data
                        self.data.columns = ['x1', 'x2']
                    except:
                        print('Failed to convert column names')
            else:
                raise InvalidInput('Input must be pandas data frame.')
        self.tot_withinss = 0  # total within cluster sum of squared error
        self.centroids = None  # coordinates of cluster centroids

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
            # centroids = np.random.randint(self.data.shape[0], size=k)
            centroids = np.random.choice(self.data.shape[0], k, replace=False)
            return centroids
        elif method == 'kmpp':
            # use kmeans++ to pick initial points
            print('kmpp not support yet. Using random instead...')
            # centroids = np.random.randint(self.data.shape[0], size=k)
            centroids = np.random.choice(self.data.shape[0], k, replace=False)
            return centroids
            pass
        else:
            raise InvalidInput('Invalid initialization method.')

    def euc_dist(self, p1, p2):
        """Euclidean distance between any two points

        Args:
            p1 (array-like): coordinates (x1, x2) of the first point.
            p2 (array-like): coordinates (x1, x2) of the second point.
        Return:
            float: the Euclidean distance between the two points.
        """
        try:
            dist = np.linalg.norm(p1 - p2)
            return dist
        except:
            print('Error in calculating the Euclidean distance.')

    def should_stop(self, centroids, centroids_new, eps=0):
        """Check if kmeans converged.

        Args:
            centroids (Pandas DataFrame): previous cluster centroids.
            centroids_new (Pandas DataFrame): new cluster centroids.
            eps (float): tolerance

        Return:
            bool: indicates whether kmeans has converged.
        """
        c0 = centroids.reset_index(drop=True)
        c1 = centroids_new.reset_index(drop=True)
        diff = c0.subtract(c1)
        diff = diff.abs()
        # check the difference between old and new centroids
        if np.max(diff.max(axis=1).values) <= eps:
            return True
        return False

    def tot_wss(self, data):
        """Calculate the total within cluster sum of square error.

        Args:
            data (Pandas DataFrame): Clustered data grouped by cluster label.

        Return:
            float
        """
        tot_wss = 0
        def ed(row, centroid):
            return np.linalg.norm(row.values[0:1] - centroid.values)
        for name, cluster in data:
            centroid = cluster.mean()
            centroid = centroid[['x1', 'x2']]
            # apply the Euclidean distance function to each row of the data in the cluster
            dist = cluster.apply(ed, centroid=centroid, axis=1)
            tot_wss = tot_wss + dist.sum()
        assert tot_wss >= 0, 'Error in tot_wss. tot_wss cannot be less than 0'
        return tot_wss

    def fit(self, K, method='random'):
        """Compute k-means clustering

        Args:
            K (int): number of clusters.

        Returns:
            dictionary: Contains
                        1. pandas data frame of the attributes and clustering for each data point
                        2. total within cluster sum of square and
                        3. pandas data frame of k centroids
        """
        if ((not isinstance(self.data, pd.DataFrame)) or
                (not self.data.shape[1] == 2) or
                (not self.data.shape[0] >= 1)):
            raise InvalidInput('Input error: Input must be a data frame and \
                have at least one row and two columns.')

        data = self.data[['x1', 'x2']]  # raw
        self.data = data
        nobs = data.shape[0]  # number of samples

        # get initial centroids
        cent_init = self.init_cent(K, method)
        centroids = data.iloc[cent_init, [0,1]]
        centroids = centroids.reset_index(drop=True)

        dist_mat = np.zeros((nobs, K))  # for calculation
        labels = []
        eps = 0
        n_iter = 0
        max_iter = 100
        stop = False

        while (not stop):
            labels = []
            for row in range(nobs):
                for k in range(K):
                    # each column in dist_mat represents the distance from a
                    # point to the centroid of its cluster
                    dist_mat[row, k] = self.euc_dist(data.iloc[row,[0,1]],
                                                     centroids.iloc[k,[0,1]])
                idx_min = np.argmin(dist_mat[row,])
                labels.append(str(int(idx_min) + 1))  # labels are string

            # group data based on labels
            data['cluster'] = labels

            # new centroids in the iteration
            centroids_new = data.groupby('cluster').mean()

            # stop if new centroids = old centroids or exceeds max iteration (100)
            if (self.should_stop(centroids, centroids_new, eps) or (n_iter > max_iter)):
                stop = True

            centroids = centroids_new
            n_iter = n_iter + 1

        print('kmeans converged in %i runs' % n_iter)

        tot_withinss = self.tot_wss(data.groupby('cluster'))

        # save to class attributes
        self.data = data
        self.tot_withinss = tot_withinss
        self.centroids = centroids

        res = {
            'data': data,
            'tot_withinss': tot_withinss,
            'centroids': pd.DataFrame(centroids)
        }

        return res

    def predict(self, data_new, centroids):
        """Predict k-means clustering for new data frame
        Returns:
            dataframe: Contains
                       1. new data
                       2. clustering label for each data point
        """
        # Check inputs
        if ((not isinstance(data_new, pd.DataFrame)) or
                (not data_new.shape[1] == 2) or
                (not data_new.shape[0] >= 1)):
            raise InvalidInput('Input error: Input must be a data frame and \
                have at least one row and two columns.')

        if ((not isinstance(centroids, pd.DataFrame)) or
                (not data_new.shape[1] == 2) or
                (not data_new.shape[0] >= 1)):
            raise InvalidInput('Input error: centroids must be a data frame and \
                have at least one row and two columns.')

        data = data_new
        def calc_dist(row, dat): # row is centroids
            return np.linalg.norm(row - dat)

        def assign_cluster(row): # row is data here
            # distance from a point to all the centroids
            dist = centroids.apply(calc_dist, dat=row, axis=1)
            # cluster label is the index of the centroid closest to the point
            label = np.argmin(dist.values) + 1
            return str(label)

        # iterate over each data point, i.e. row in data
        data['cluster'] = data.apply(assign_cluster, axis=1)
        return data

    def input_shape_validation(self, data_to_valid=None):
        """Utility function checking input data shape for kmplot.

        Args:
            data_to_valid (DataFrame): Data to be validated.

        Returns
            bool: True if shape is valid. False otherwise.
        """
        # if data_to_valid not provided, use self.data instead
        if data_to_valid is None:
            data = self.data
        else:
            data = data_to_valid

        if (not data.shape or
                len(data.shape) <= 1 or
                data.shape[0] <= 0 or
                data.shape[1] < 3):
            return False
        return True

    def input_label_validation(self, data_to_valid=None):
        """Utility function checking if the cluster column exists for kmplot.

        Args:
            data_to_valid (DataFrame): Data to be validated.

        Returns:
            bool: True if the cluster column exists. False otherwise.
        """
        # if data_to_valid not provided, use self.data instead
        if data_to_valid is None:
            data = self.data
        else:
            data = data_to_valid

        if (not 'cluster' in data.columns):
            return False
        return True

    def kmplot(self, pred_data=None):
        """Visualize kmeans results in a scatter plot.

        Args:
            pred_data (DataFrame): prediction data with cluster labels.

        Returns:
            matplotlib figure: plot showing the scatter plot of kmeans
                                     results, colored by clusters.
            matplotlib axes: figure details.

        Raises:
            InvalidInput: If self.data has zero row, less than three columns,
                          or no cluster column.
        """
        # if pred_data not provided, use self.data instead, and in this case,
        # kmplot will plot the training results. If pred_data is provided,
        # it will plot the prediction results
        if pred_data is None:
            data = self.data
        else:
            data = pred_data

        if not self.input_shape_validation(data):
            raise InvalidInput('Input must have at least one row and \
                three columns.')
        if not self.input_label_validation(data):
            raise InvalidInput('No cluster labels. Run fit first before plot.')

        clusters = data.groupby('cluster')
        fig, ax = plt.subplots()
        for name, cluster in clusters:
            ax.plot(cluster.x1, cluster.x2, marker='.', linestyle='', label=cluster)
        # fig = plt.plot(data.iloc[:,0], data.iloc[:,1], '.')
        plt.xlabel('x1')
        plt.ylabel('x2')

        return fig, ax
