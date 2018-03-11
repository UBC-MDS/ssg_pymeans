===========
ssg_pymeans
===========

ssg_pymeans implements kmeans for two-dimensional datasets. It is not a wrapper around other kmeans packages.

ssg_pymeans provides options for different initial centroid
selections, including random and kmeans++. You might find the package useful
if you try to understand the basics of kmeans or to build more advanced features
on top this basic kmeans implementation. Typical usage often looks like this:

    #!/usr/bin/env python

    from ssg_pymeans import Pymeans

    pymeans = Pymeans(my_data_frame)
    model = pymeans.fit(k = 3, method = "random")
    results = pymenas.predict(new_data  = new_data_frame, centroids = model['centroids'])
    pymenas.kmplot(results)

Home Page
=========

https://github.com/UBC-MDS/ssg_pymeans

Dependencies
============

ssg_pymeans requires matlplotlib, pandas and numpy.

Functions
=========

* fit: build clusters
* predict: predict the label of new data based on the cluster attributes
* kmplot: visualize the results and performance

The package includes two datasets for testing and demonstration.

Outputs related to performance (within cluster sum of squared distance) is part of the output from clustering.

Ecosystem
=========

Similar package:

* k-means in sklearn:
  http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

ssg_pymeans is intended to help understand the fundamentals of k-means and
variants. Contributors are encouraged to build advanced features on top of this
base k-means package.
