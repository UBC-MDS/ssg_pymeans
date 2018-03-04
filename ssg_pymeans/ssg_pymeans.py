import pandas as pd
import matplotlib.pyplot as plt
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
            pass
        elif method == 'kmpp':
            # use kmeans++ to pick initial points
            pass
        else:
            raise InvalidInput('Invalid initialization method.')

    def fit(self):
        pass

    def predict(self):
        pass

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
