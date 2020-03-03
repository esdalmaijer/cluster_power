CLUSTER POWERRRRRRRR
====================

This is the code for data simulation, dimensionality reduction, and clustering used in the following manuscript:

- Dalmaijer, E.S., Nord, C.L., & Astle, D.E. (2020). Statistical power for cluster analysis. [arXiv:2003.00381](https://arxiv.org/abs/2003.00381)

Please cite the manuscript in any work that you publish on the basis of this code.


![Bunny amid science eggs and Easter graphs](https://github.com/esdalmaijer/cluster_power/raw/master/github_social_image.jpg "FUZZY CLUSTER POWERRRRR")



How do I use this?
------------------

1) Install Python, and install the required dependencies listed in the "dependencies.txt" file.

2) Adjust whatever you'd like in the code.

3) Run `cluster_power.py` to produce simulated covariance matrices, datasets, dimensionality reduction, clustering, and simulations for power estimation.

4) Run `c-means_k-means_comparison` to create the example k-means and c-means graphs, and run the simulation that compares c-means and k-means silhouette coefficients.

5) Helper functions are included in `cluster.py` and `data_generation_playground.py`, and can be imported in the standard Python way.


How do I use your code for \[METHOD X\]?
----------------------------------------

A great many analysis options for dimensionality reduction and clustering have already been implemented, and you can use this to compute power for the specific pipeline that you have in mind. (You're totes welcome.)

To run our simulations, change the script `cluster_power.py`, specifically where it says `DIM_REDUCTION_ALGORITHMS` and `CLUSTER_ALGORITHMS`. The available options are listed below.

**Dimensionality reduction**
- None
- "PCA" principal component analysis
- "ICA" independent component analysis
- "FA" factor analysis
- "TSNE" t-stochastic neighbour embedding
- "UMAP" uniform manifold approximation and embedding
- "RANDOMPROJECTION"
- "FEATUREAGGLOMERATION"
- "ISOMAP"
- "LLE" local linear embedding
- "HESSIAN" Hessian eigenmaps
- "MLLE" modified local linear embedding
- "LTSA" local tangent space alignment
- "MDS" multi-dimensional scaling
- "DICTIONARY" dictionary learning
- "TSVD" truncated SVD (also known as "LSE")

**Cluster algorithms**
- None
- "KMEANS"
- "WARD" (agglomerative clustering, Ward linkage, Euclidean distance)
- "COSINE" (agglomerative clustering, average linkage, cosine distance)
- "EUCLIDEAN" (agglomerative clustering, average linkage, Euclidean distance)
- "CITYBLOCK" or "MANHATTAN" (agglomerative clustering, average linkage, city block distance)
- "SPECTRAL"
- "AVERAGELINKAGE"
- "BIRCH"
- "DBSCAN"
- "HDBSCAN"
- "AFFINITYPROPAGATION"
- "MEANSHIFT"


Dependencies
------------

I've included a `pip freeze` output, but I'm sure that not all listed dependencies are actually required. You basically only need the following packages (and their respective dependencies):

- HDBSCAN, hdbscan==0.8.12
- Matplotlib, matplotlib==2.1.2
- NumPy, numpy==1.16.5
- scikit-fuzzy, scikit-fuzzy==0.4.2
- scikit-learn, scikit-learn==0.20.4
- SciPy, scipy==1.2.2
- UMAP, umap-learn==0.2.1


Do I smell Python 2?
--------------------

Yes, sorry. One thousand written apologies and a Chelsea bun for everyone who is negatively impacted by this, and willing to come claim them.

This started as what seemed to be a really quick thing in 2019. It then rapidly spiralled out of control, as complexities were becoming apparent. That said, the code likely only needs a few changes to run smoothly on Python 3, and all dependencies should be compatible too.

I can't update the code myself, as I need to block out enough time in my days to play with my demanding cat. But I'll happily accept pull requests of people who heroically take on the challenge.


License
-------

GPLv3. The full legal document is enclosed, but basically: Use as you please, but credit Dr Edwin Dalmaijer (and cite the above paper where appropriate). Also don't blame me if using this code somehow causes you harm.

The bunny photo used here is by Tim Reckmann, who originally posted it to Flickr, after which it was uploaded to Wikimedia Commons as [Bunny_with_eggs_in_Easter](https://commons.wikimedia.org/wiki/File:Bunny_with_eggs_in_Easter.jpg). It is licensed under the terms of the cc-by-2.0.

