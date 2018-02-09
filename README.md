# ssg_pymeans

A Python package for k-means clustering.

## Contributors:

Sophia Wang, Susan Fung, Guanchen Zhang

## Description

This is the repository for the Python version of the `ssg_pymeans` package. The R version is available [here](https://github.com/UBC-MDS/ssg_kmeansr).

This package implements the classical unsupervised clustering method, [k-means](https://en.wikipedia.org/wiki/K-means_clustering).

## Functions

The package provides the following functions:

- basic k-means: initial points are picked randomly.
- k-means ++: initial points are picked based on distance. k-means++ is explained [here](https://en.wikipedia.org/wiki/K-means%2B%2B).
- plotting: the package will provide plotting functions to visualize the results and performance.

Some miscellenous functions include:

- input validation
- input scaling: scale/normalize the input data if necessary

Optinal functions depending on progress:

- EM clustering
- Multi-dimensional scaling

## Ecosystem

Similar package:

- [k-means in sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

`ssg_pymeans` is intended to help understand the fundamentals of k-means and variants. Contributors are encouraged to build advacned features on top of this base k-means package.
