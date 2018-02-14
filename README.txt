===========
ssg_pymeans
===========

ssg_pymeans provides functions to implement k-means, a classical unsupervised
clustering method. ssg_pymeans provides options for different initial centroid
selection, including random and kmeans++. You might find the package useful
if you try to understand the basics of kmeans or to build more advanced features
on top this basic kmeans implementation. Typical usage often looks like this:

    #!/usr/bin/env python

    from ssg_pymeans import Pymeans

    kmeans_job = Pymeans()
    kmeans_job.init_cent('kmpp')
    kmeans_job.kmeans()
    kmeans_job.kmplot()

Dependencies
============

ssg_pymeans requires matlplotlib and numpy.

Functions
=========

* init_cent: initial points selection
* kmeans: build clusters
* predict: predict the label of new data based on the cluster attributes
* kmplot: visualize the results and performance

The package will include one or two common datasets for testing and
demonstration.

Some utility functions include:

* input_validation: check for input validity, e.g. whether the dimensions of
  X match with the label vector (if provided by user)
* input scaling: scale/normalize the input data if necessary

Optional functions on the roadmap:

* EM clustering
* Multi-dimensional scaling
* Transductive/inductive SSL

Outputs related to performance will be denoted by class attributes, e.g. within
cluster sum of squared distance.

Ecosystem
=========

Similar package:

* k-means in sklearn:
  http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

ssg_pymeans is intended to help understand the fundamentals of k-means and
variants. Contributors are encouraged to build advanced features on top of this
base k-means package.
