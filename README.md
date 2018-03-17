[![Build Status](https://travis-ci.org/UBC-MDS/ssg_pymeans.svg?branch=master)](https://travis-ci.org/UBC-MDS/ssg_pymeans)

# ssg_pymeans

A Python package for k-means clustering.

## Contributors

Sophia Wang, Susan Fung, Guanchen Zhang

## Description

This is the repository for the Python version of the `ssg_pymeans` package. The R version is available [here](https://github.com/UBC-MDS/ssg_kmeansr).

This package implements the classical unsupervised clustering method, [k-means](https://en.wikipedia.org/wiki/K-means_clustering) for two-dimensional datasets, with options for choosing the initial centroids (e.g. random and kmeans++). Users will be able to find clusters in their data, label new data, and observe the clustering results.

## Functions

The package implements in an OOP fashion the following class functions:

- initial points selection:
  -  basic k-means: initial centroids are picked randomly.
  -  k-means++: initial centroids are picked based on distance. More details can be found [here](https://en.wikipedia.org/wiki/K-means%2B%2B). (Note: this feature is not yet implemented.)
- clustering: build clusters and save cluster attributes
- prediction: predict the label of new data based on the cluster attributes
- plotting: the package will provide plotting functions to visualize the results and performance

Outputs related to performance (within cluster sum of squared distance) is part of the output from clustering.

The package includes two datasets for testing and demonstration.

## Installing the Package

Run the following in your command line:

`pip install git+https://github.com/UBC-MDS/ssg_pymeans.git`

## Examples

```python
from ssg_pymeans import Pymeans
import pandas as pd

# sample toy data sets, you can find a bigger sample data set in the data folder
train_data = pd.DataFrame({
    'x1': pd.Series([1, 2, 3]),
    'x2': pd.Series([4, 5, 6])
})

test_data = pd.DataFrame({
    'x1': pd.Series([1, 2, 3]),
    'x2': pd.Series([4, 5, 6])
})

# Load the included data in the package
pymeans = Pymeans()
# Alternatively, load your own data
# pymeans = Pymeans(data = train_data)

# Train the model
model = pymeans.fit(K=3) # three clusters

# See the training results
pymeans.kmplot()

# Make prediction on new data set
pred_results = pymeans.predict(test_data, model['centroids'])
pymeans.kmplot(pred_results)
```

## Ecosystem

Similar package:

- [k-means in sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

`ssg_pymeans` is intended to help understand the fundamentals of k-means and variants. Contributors are encouraged to build advanced features on top of this base k-means package.
