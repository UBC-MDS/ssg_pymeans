# ssg_pymeans

A Python package for k-means clustering.

## Contributors:

Sophia Wang, Susan Fung, Guanchen Zhang

## Description

This is the repository for the Python version of the `ssg_pymeans` package. The R version is available [here](https://github.com/UBC-MDS/ssg_kmeansr).

This package implements the classical unsupervised clustering method, [k-means](https://en.wikipedia.org/wiki/K-means_clustering), with options for choosing the initial centroids (e.g. random and kmeans++). Users will be able to find clusters in their data, label new data, and observe the clustering results.

Depending on the progress and time constraint, we also plan to cover some other clustering methods in this package. See details in the section below.

## Functions

The package will be implemented in an OOP fashion with the following class functions:

- initial points selection:
  -  basic k-means: initial centroids are picked randomly.
  -  k-means++: initial centroids are picked based on distance. More details can be found [here](https://en.wikipedia.org/wiki/K-means%2B%2B).
- clustering: build clusters and save cluster attributes
- prediction: predict the label of new data based on the cluster attributes
- plotting: the package will provide plotting functions to visualize the results and performance

Outputs related to performance will be denoted by class attributes, e.g. within cluster sum of squared distance.

The package will include one or two common datasets for testing and demonstration.

Some miscellenous functions include:

- input validation: check for input validity, e.g. whether the dimensions of X match with the label vector (if provided by user).
- input scaling: scale/normalize the input data if necessary

Optional functions depending on the progress:

- EM clustering
- Multi-dimensional scaling
- Transductive/inductive SSL

## Ecosystem

Similar package:

- [k-means in sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

`ssg_pymeans` is intended to help understand the fundamentals of k-means and variants. Contributors are encouraged to build advacned features on top of this base k-means package.
