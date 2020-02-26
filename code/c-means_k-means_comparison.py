import os
import copy
import time
import itertools

import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn import datasets
from sklearn.metrics import adjusted_rand_score, calinski_harabaz_score, \
    silhouette_samples, silhouette_score
import skfuzzy

from cluster import clustering, cluster_comparison, convenience_clustering, \
    correlation_matrix, dim_reduction, plot_averages, plot_clusters, \
    plot_samples, plot_silhouette, preprocess

from data_generation_playground import create_sample, create_equidistant_sample, \
    dD_plot, fuzzy_silhouette_coefficient, predict_D
    

def compute_silhouette_coefficient(X, y, u=None, alpha=1.0):
    
    # Count the unique labels.
    labels = numpy.unique(y)
    # Create a vector for easy Boolean-vector creation.
    arr = numpy.zeros(X.shape[0], dtype=int)
    arr[:,] = range(0,X.shape[0])
    # Create an empty vector to hold silhouette values in.
    s = numpy.zeros(y.shape, dtype=float) * numpy.NaN
    # Loop through all samples.
    for i in range(X.shape[0]):
        # Select all non-i samples.
        sel = arr != i
        # Compute the distance between the current sample and all other samples.
        d = numpy.sqrt(numpy.sum((X[sel,:] - X[i,:])**2, axis=1))
        # Compute the distance between the current sample and its cluster
        # members.
        a = numpy.mean(d[y[sel]==y[i]])
        # Compute the distance between the current sample and all other
        # clusters' members.
        b_min = numpy.inf
        for lbl in labels:
            if y[i] != lbl:
                b = numpy.mean(d[y[sel]==lbl])
                if b < b_min:
                    b_min = copy.copy(b)
        # Compute the cluster silhouette for this sample.
        s[i] = (b_min - a) / max(a, b_min)
    
    if u is None:
        s_m = numpy.mean(s)
    else:
        # Find largest and second-largest elements for all samples.
        u_sorted = numpy.sort(u, axis=1)
        u_p = u_sorted[:,-1]
        u_q = u_sorted[:,-2]
        # Compute the fuzzy silhouette score.
        s_m = numpy.sum( ((u_p - u_q)**alpha) * s) / numpy.sum((u_p - u_q)**alpha)

    return s_m, s


# # # # #
# CONSTANTS

# Colours from the "points of view" set.
COLS = { \
    "orange": "#e69f00", \
    "skyblue": "#56b4e9", \
    "blueishgreen": "#009e74", \
    "yellow": "#f0e442", \
    "blue": "#0072b2", \
    "vermillion": "#d55e00", \
    "reddishpurple": "#cc79a7", \
    }
# Colours for plots.
PLOTCOLS = { \
    0: COLS["orange"], \
    1: COLS["skyblue"], \
    2: COLS["reddishpurple"], \
    }

# FILES AND FOLDERS
# Auto-detect the current directory.
DIR = os.path.dirname(os.path.abspath(__file__))
# Construct paths to output directories.
OUTDIR = os.path.join(DIR, "output", "example_plots")
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


# # # # #
# EXAMPLE CLUSTERS

# This is a figure with 6 panels. The top rows contain ground truths, and
# the bottom k-means results. The left column is idealised data (make_blobs),
# the middle is the Iris dataset, and the right is more realistic data with
# closer centroids.

# Dataset names.
dataset_names = ["blobs", "iris", "realistic"]

# Make data.
k = 3
n_observations = 150
n_features = 4
realistic_centroid_separation = 3.3
within_feature_difference = numpy.sqrt((realistic_centroid_separation**2) \
    / n_features)
dims = (n_observations, n_features, len(dataset_names))
X = numpy.zeros(dims, dtype=float)
y_truth = numpy.zeros((dims[0],dims[2]), dtype=int)
for di, data_name in enumerate(dataset_names):

    if data_name == "blobs":
        # Make a few blobs.
        X[:,:,di], y_truth[:,di] = datasets.make_blobs(n_samples=X.shape[0], \
            n_features=X.shape[1], centers=k, cluster_std=1.0, random_state=1)

    elif data_name == "iris":
        # Load the Iris dataset.
        iris = datasets.load_iris()
        # Project into 2 dimensions using MDS.
        X[:,:,di] = iris.data
        y_truth[:,di] = iris.target
    
    elif data_name == "multivariate normal":
        # Seed random generator for reproducibility.
        numpy.random.seed(1)
        # Make a fake dataset.
        p = numpy.ones(k) / float(k)
        X[:,:,di], y_truth[:,di], cov = create_sample(n_observations, \
            n_features, n_features, within_feature_difference, p, \
            min_cov=-0.3, max_cov=0.3, cov_matrix=None)

    elif data_name == "realistic":
        # Seed random generator for reproducibility.
        numpy.random.seed(1)
        # Define p and d.
        p = numpy.ones(k) / float(k)
        d = realistic_centroid_separation
        # Determine the size of each cluster.
        sample_n = []
        for p_ in p:
            sample_n.append(int(numpy.round(n_observations*p_, decimals=0)))
        # Recompute the actual sample size.
        n = sum(sample_n)
        # Create the ground truth cluster membership.
        y_truth[:,di] = numpy.ones(n, dtype=int) * -1
        si = 0
        for i in range(len(sample_n)):
            y_truth[:,di][si:si+sample_n[i]] = i
            si += sample_n[i]
        # Create covariance matrix.
        min_cov = -0.3
        max_cov = 0.3
        cov_matrix = numpy.zeros((n_features, n_features), dtype=float)
        for i in range(cov_matrix.shape[0]):
            for j in range(i+1, cov_matrix.shape[1]):
                cov_matrix[i,j] = min_cov + numpy.random.rand() * max_cov
                cov_matrix[j,i] = cov_matrix[i,j]
        # Set diagonal to 1.0
        cov_matrix[numpy.diag_indices(n_features, ndim=2)] = 1.0
        # Correct wrong covariance matrices.
        min_eig = numpy.min(numpy.real(numpy.linalg.eigvals(cov_matrix)))
        if min_eig < 0:
            # Correct the covariance matrix.
            cov_matrix -= 10*min_eig * numpy.eye(*cov_matrix.shape)
            # Transform back to a maximum of 1.0
            cov_matrix /= numpy.max(cov_matrix)
        # Create means in two dimensions.
        m = numpy.zeros((n_features, k), dtype=float)
        # Move the means to the left and the right.
        if k == 2:
            # Move left.
            m[0,0] -= d/2.0
            # Move right.
            m[0,1] += d/2.0
        # Move the means away from the middle so that the points come
        # to lie on an equilateral triangle. The basis of this is
        # cut in half by (0,0), and the third point is directly above
        # (0,0) at distance sqrt(d**2 - (d/2)**2)
        elif k == 3:
            # Move left.
            m[0,0] -= d/2.0
            # Move right.
            m[0,1] += d/2.0
            # Move up.
            m[1,2] += numpy.sqrt(d**2 - (d/2)**2)
        # Move means to four corners, so that the corners form two
        # stacked equilateral triangles. This means the distance
        # between all points is d. If the left and right points are at
        # distance d/2 from the centre (0,0), then the top and bottom
        # points are at distance sqrt(d**2 - (d/2)**2).
        elif k == 4:
            # Move left.
            m[0,0] -= d/2.0
            # Move right.
            m[0,1] += d/2.0
            # Move up.
            m[1,2] += numpy.sqrt(d**2 - (d/2)**2)
            # Move down.
            m[1,3] -= numpy.sqrt(d**2 - (d/2)**2)
        # Simulate data.
        X[:,:,di] = numpy.zeros((n_observations,m.shape[0]), dtype=float)
        for i in range(m.shape[1]):
            X[:,:,di][y_truth[:,di]==i,:] = numpy.random.multivariate_normal( \
                m[:,i], cov_matrix, size=sample_n[i])


# PLOT
# Set random seed for reproducibility (MDS and k-means are stochastic).
numpy.random.seed(1)
# Create a new figure.
fig, ax = pyplot.subplots(nrows=2, ncols=3, figsize=(20.0, 12.0), dpi=600.0)
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, \
    wspace=0.1, hspace=0.1)
# Loop through all datasets.
for i in range(X.shape[2]):
    
    # Set the axis label.
    ax[0,i].set_title(dataset_names[i].title(), fontsize=32)

    # Compute original centroids.
    m = numpy.zeros((X[:,:,i].shape[1], k), dtype=float)
    for j in range(k):
        m[:,j] = numpy.mean(X[:,:,i][y_truth[:,i]==j], axis=0)

    # Compute all centroid distances.
    combs = list(itertools.combinations(range(k), 2))
    d = numpy.zeros(len(combs))
    for j, comb in enumerate(combs):
        d[j] = numpy.sqrt(numpy.sum((m[:,comb[0]]-m[:,comb[1]])**2))

    # Compute a 2-dimensional projection.
    X_projected = dim_reduction(X[:,:,i], n_components=2, mode="MDS")
    # Transform projection in space between (0.1, 0.9)
    for j in range(X_projected.shape[1]):
        _min = numpy.min(X_projected[:,j])
        _max = numpy.max(X_projected[:,j])
        if _min < 0:
            X_projected[:,j] += numpy.abs(_min)
            _max += numpy.abs(_min)
        X_projected[:,j] = 0.1 + 0.8* (X_projected[:,j] / _max)
    
    # Plot the original data.
    for lbl in numpy.unique(y_truth[:,i]):
        ax[0,i].plot(X_projected[:,0][y_truth[:,i]==lbl], \
            X_projected[:,1][y_truth[:,i]==lbl], 'o', color=PLOTCOLS[lbl])
    
    # Cluster the data.
    y = clustering(X[:,:,i], mode="KMEANS", n_clusters=k)
    
    # Compute the silhouette score.
    sil = silhouette_score(X[:,:,i], y, metric="euclidean")

    # Compute overlap. Note that clusters might 
    # overlap, but be labelled with a different value.
    # Values are arbitrary, so this should be 
    # corrected.
    # Run through all possible permutations of labels.
    perms = list(itertools.permutations(range(k)))
    overlap = numpy.zeros(y.shape[0], dtype=bool)
    max_overlap = 0
    closest_perm = list(numpy.unique(y))
    for perm in perms:
        # Create permutated array.
        y_perm = numpy.copy(y)
        for j in range(k):
            y_perm[y==j] = perm[j]
        # Compute the overlap for this permutation.
        o = y_truth[:,i] == y_perm
        # Save the overlap if this is the current best.
        if numpy.sum(o.astype(int)) > max_overlap:
            overlap = copy.deepcopy(o)
            max_overlap = numpy.sum(overlap.astype(int))
            closest_perm = copy.deepcopy(perm)
    
    # Plot the clustered data.
    for lbl in numpy.unique(y_truth[:,i]):
        ax[1,i].plot(X_projected[:,0][y==lbl], X_projected[:,1][y==lbl], \
            'o', color=PLOTCOLS[closest_perm[lbl]])
    
    # Annotate overlap and silhouette coefficient.
    ax[1,i].annotate("Silhouette score = {}".format(round(sil, ndigits=2)), \
        (0.05, 0.9), fontsize=18)
    s = "Accuracy = {}%".format(int(round(100*max_overlap/float(y.shape[0]))))
    ax[1,i].annotate(s, (0.05, 0.8), fontsize=18)
    
# Remove ticks from all plots (space is arbitrary).
for i in range(ax.shape[1]):
    for j in range(ax.shape[0]):
        ax[j,i].set_xticks([])
        ax[j,i].set_yticks([])
        ax[j,i].set_xlim(0,1)
        ax[j,i].set_ylim(0,1)
# Set axis labels.
ax[0,0].set_ylabel("Ground truth", fontsize=32)
ax[1,0].set_ylabel("Cluster solution", fontsize=32)

# Save figure.
fig.savefig(os.path.join(OUTDIR, "fig_01_cluster_examples.png"))
pyplot.close(fig)


# # # # #
# FUZZY C-MEANS

# Six panels, two rows and three columns.
# Left column:  top:    original data (three clusters)
#               bottom: empty (for bunny)
# Mid column:   top:    k-means solution
#               bottom: c-means solution
# Right column: top:    silhouettes for k-means
#               bottom: silhouettes for c-means

# Create an equidistant 3-cluster multivariate normal distribution.
numpy.random.seed(1)
X_, y_, cov_ = create_equidistant_sample(n_observations=500, d=3.0, \
    p=(0.33,0.34,0.33))
labels = list(numpy.unique(y_))
k = len(labels)

# Compute all centroid distances.
combs = list(itertools.combinations(range(k), 2))
d = numpy.zeros(len(combs))
for j, comb in enumerate(combs):
    d[j] = numpy.sqrt(numpy.sum((m[:,comb[0]]-m[:,comb[1]])**2))

# Transform projection in space between (0.1, 0.9)
for j in range(X_.shape[1]):
    _min = numpy.min(X_[:,j])
    _max = numpy.max(X_[:,j])
    if _min < 0:
        X_[:,j] += numpy.abs(_min)
        _max += numpy.abs(_min)
    X_[:,j] = 0.1 + 0.8* (X_[:,j] / _max)

# Compute original centroids.
m = numpy.zeros((X_.shape[1], k), dtype=float)
for j in range(k):
    m[:,j] = numpy.mean(X_[y_==j], axis=0)

# Create a new figure.
fig, ax = pyplot.subplots(nrows=2, ncols=3, figsize=(20.0, 12.0), dpi=600.0)
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, \
    wspace=0.1, hspace=0.1)

# Plot original distribution in top-left.
for lbl in labels:
    ax[0,0].plot(X_[:,0][y_==lbl], X_[:,1][y_==lbl], 'o', color=PLOTCOLS[lbl])
# Annotate the cluster separation.
ax[0,0].annotate(r"$\Delta=${}".format(round(numpy.min(d), ndigits=1)), \
    (0.05, 0.9), fontsize=24)

# Run through k-means and c-means.
for i, cluster_method in enumerate(["k-means", "c-means"]):
    
    # Employ cluster algorithm.
    if cluster_method == "k-means":
        y = clustering(X_, mode="KMEANS", n_clusters=k)
        u = None
    elif cluster_method == "c-means":
        _, u, _, _, _, _, _ = skfuzzy.cluster.cmeans( \
            numpy.transpose(X_), k, 2, error=0.005, maxiter=1000, init=None)
        # Get hard cluster membership.
        y = numpy.argmax(u, axis=0)
        # Transpose u to use it later.
        u = numpy.transpose(u)
    
    # Compute silhouette scores.
    sil_mean, sil = fuzzy_silhouette_coefficient(X_, y, u, alpha=1.0)
    
    # Compute overlap. Note that clusters might 
    # overlap, but be labelled with a different value.
    # Values are arbitrary, so this should be 
    # corrected.
    # Run through all possible permutations of labels.
    perms = list(itertools.permutations(range(k)))
    overlap = numpy.zeros(y.shape[0], dtype=bool)
    max_overlap = 0
    closest_perm = list(numpy.unique(y))
    for perm in perms:
        # Create permutated array.
        y_perm = numpy.copy(y)
        for j in range(k):
            y_perm[y==j] = perm[j]
        # Compute the overlap for this permutation.
        o = y_ == y_perm
        # Save the overlap if this is the current best.
        if numpy.sum(o.astype(int)) > max_overlap:
            overlap = copy.deepcopy(o)
            max_overlap = numpy.sum(overlap.astype(int))
            closest_perm = copy.deepcopy(perm)

    # Plot cluster outcome.
    for lbl in numpy.unique(y):
        # In k-means, draw fully opaque points.
        if cluster_method == "k-means":
            sel = y==lbl
            ax[i,1].plot(X_[:,0][sel], X_[:,1][sel], 'o', \
                color=PLOTCOLS[closest_perm[lbl]])
        # For c-means, create a colour array with varrying transparency.
        elif cluster_method == "c-means":
            # Create the colour vector.
            rgb = matplotlib.colors.to_rgb(PLOTCOLS[closest_perm[lbl]])
            col = numpy.zeros((X_.shape[0],4), dtype=float)
            col[:,0:3] = rgb
            col[:,3] = u[:,lbl]
            # Draw the points.
            ax[i,1].scatter(X_[:,0], X_[:,1], color=col)

    # Compute the opacity levels for silhouettes.
    if cluster_method == "k-means":
        # Simply use full opacity in k-means.
        opacity = numpy.ones(y.shape)
    if cluster_method == "c-means":
        # For fuzzy clustering, sort the u matrix to find the largest two values.
        u_sorted = numpy.sort(u, axis=1)
        u_p = u_sorted[:,-1]
        u_q = u_sorted[:,-2]
        opacity = (u_p-u_q)**1.0
        # Scale opacity to reflect normalised contribution.
        opacity /= numpy.max(opacity)
        # Computed corrected silhouette coefficients.
        sil_cor = (opacity*sil) / (numpy.sum(opacity)/sil.shape[0])

    # Plot silhouette values.
    y_pos = 0
    y_ax_upper = 0
    y_ax_lower = 0
    y_ticks = []
    # Go through all unique labels.
    for lbl in closest_perm:
        # Create a Boolean vector to select the observations assigned to the
        # current cluster.
        sel = y==lbl
        # Compute distances between samples and their assigned centroid.
        #d = numpy.sqrt(numpy.sum((X_[sel,:] - m[:,lbl])**2, axis=1))
        # Create a sorting based on centroid distance.
        sort_indices = numpy.argsort(sil[sel])
        # Find all silhouette values in this cluster.
        if cluster_method == "k-means":
            cluster_silhouette_vals = sil[sel]
        elif cluster_method == "c-means":
            cluster_silhouette_vals = sil_cor[sel]
        # Sort the silhouette values by centroid distance.
        cluster_silhouette_vals = cluster_silhouette_vals[sort_indices]
        # Loop through all values.
        y_ax_upper += len(cluster_silhouette_vals)
        # Plot the shadows of the k-means silhouettes.
        ax[i,2].barh(range(y_ax_lower, y_ax_upper), \
            sil[sel][sort_indices], height=1.0, edgecolor=None, alpha=0.2, \
            color="#000000")
        # Create a horizontal bar graph for all samples in this cluster.
        ax[i,2].barh(range(y_ax_lower, y_ax_upper), cluster_silhouette_vals, \
            height=1.0, edgecolor=None, alpha=0.6, \
            color=PLOTCOLS[closest_perm.index(lbl)])
        # Update the y ticks.
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.0)
        y_ax_lower += len(cluster_silhouette_vals)
    # Draw the cluster coefficient into the graph.
    ax[i,2].axvline(sil_mean, color='#ff69b4', linestyle='--')
    # Annotate the cluster score.
    ax[i,2].annotate("score = {}".format(round(sil_mean, ndigits=2)), \
        (sil_mean+0.03, 0), color="#ff69b4", fontsize=18)
    # Annotate the name of the algorithm.
    ax[i,1].annotate("{}".format(cluster_method), (0.05, 0.9), fontsize=24)

# Remove or set ticks for all plots.
for i in range(ax.shape[1]):
    for j in range(ax.shape[0]):
        ax[j,i].set_yticks([])
        if i == 2:
            ax[j,i].set_xticks([0.0, 0.25, 0.5, 0.7])
            ax[j,i].set_xlim([-0.2, 1.0])
            ax[j,i].tick_params(labelsize=14)
        else:
            ax[j,i].set_xticks([])
            ax[j,i].set_xlim(0,1)
            ax[j,i].set_ylim(0,1)
# Set axis titles.
ax[0,0].set_title("Ground truth", fontsize=32)
ax[0,1].set_title("Cluster solution", fontsize=32)
ax[0,2].set_title("Silhouette coefficient", fontsize=32)

# Save figure.
fig.savefig(os.path.join(OUTDIR, "fig_02_fuzzy_example.png"))
pyplot.close(fig)


# # # # #
# HARD AND FUZZY SILHOUETTES

# Define ranges for parameters.
d_range = numpy.arange(0.5, 10.5, 0.5) #[2, 3, 4, 6, 8]
#d_range = [0.0, 0.3, 0.8, 1.8, 3.8, 7.8]
k_range = [1, 2, 3, 4]
k_guess_range = range(2,8)
n_iterations = 100
# Define an uncorrelated covariance matrix.
cov = [[1.0, 0.0],[0.0,1.0]]

# Create a new figure.
fig, ax = pyplot.subplots(nrows=2, ncols=len(k_range), \
    figsize=(len(k_range)*6.0, 10.0), dpi=600.0)
fig.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.90, \
    wspace=0.1, hspace=0.1)

# Create a normed colour.
cmap = matplotlib.cm.get_cmap("viridis")
norm = matplotlib.colors.Normalize(vmin=0, vmax=10.0)

# Loop through all combinations of k and d.
for ki, k in enumerate(k_range):

    # Set the title for this axis.
    ax[0,ki].set_title(r"Ground truth $k$="+"{}".format(k), fontsize=32)
    # Draw silhouette thresholds in the plots.
    for i in range(ax.shape[0]):
        ax[i,ki].plot([0,numpy.max(k_guess_range)+1], numpy.ones(2)*0.5, \
            "--", color="#666666", alpha=0.5) #, label="Reliable clustering")
        ax[i,ki].plot([0,numpy.max(k_guess_range)+1], numpy.ones(2)*0.7, \
            "--", color="#000000", alpha=0.5) #, label="Strong clustering")
    
    # Loop through all difference values.
    for di, d in enumerate(d_range):
        
        # For k==1, only test d=0. There is no separation between 1 cluster.
        if k == 1 and di == 0:
            # Set the distance to 0 on the first iteration.
            d = 0
        elif k == 1 and di > 0:
            # Skip all next iterations.
            continue
        
        # Create vectors to hold silhouette scores in.
        dims = (len(k_guess_range), n_iterations)
        k_sil = numpy.zeros(dims, dtype=float)
        c_sil = numpy.zeros(dims, dtype=float)
        k_acc = numpy.zeros(n_iterations, dtype=float)
        c_acc = numpy.zeros(n_iterations, dtype=float)
        k_sen = numpy.zeros(n_iterations, dtype=float)
        c_sen = numpy.zeros(n_iterations, dtype=float)
        
        print("Running through {} iterations of k={}, d={}".format( \
            n_iterations, k, d))
        t0 = time.time()

        for ii in range(n_iterations):

            # Generate random data.
            p = numpy.ones(k, dtype=float) / float(k)
            X_, y_truth, _ = create_equidistant_sample(n_observations=120, \
                d=d, p=p, cov_matrix=cov)
            labels = list(numpy.unique(y_truth))
            
            # Apply k-means and c-means for different numbers of clusters.
            for i, k_guess in enumerate(k_guess_range):
                
                # Apply k-means.
                y = clustering(X_, mode="KMEANS", n_clusters=k_guess)
                u = None
                # Compute silhouette score.
                k_sil[i,ii], sil = fuzzy_silhouette_coefficient(X_, y, u, \
                    alpha=1.0)
                
                # Apply c-means.
                _, u, _, _, _, _, _ = skfuzzy.cluster.cmeans( \
                    numpy.transpose(X_), k_guess, 2, error=0.005, \
                    maxiter=1000, init=None)
                # Get hard cluster membership.
                y = numpy.argmax(u, axis=0)
                # Transpose u to use it later.
                u = numpy.transpose(u)
                # Compute silhouette scores.
                c_sil[i,ii], sil = fuzzy_silhouette_coefficient(X_, y, u, \
                    alpha=1.0)
            
            # Record if clustering was detected.
            if k == 1:
                k_acc[ii] = int(numpy.max(k_sil[:,ii]) >= 0.5)
                c_acc[ii] = int(numpy.max(c_sil[:,ii]) >= 0.5)
            else:
                k_acc[ii] = int(k_sil[k_guess_range.index(k),ii] >= 0.5)
                c_acc[ii] = int(c_sil[k_guess_range.index(k),ii] >= 0.5)
            # Record if the correct value for k was chosen.
            k_sen[ii] = int(k_guess_range[numpy.argmax(k_sil[:,ii])] == k)
            c_sen[ii] = int(k_guess_range[numpy.argmax(c_sil[:,ii])] == k)

        print("\tDone in {} seconds".format(time.time()-t0))

        # Set colour for this level of d.
        col = cmap(norm(d))
        # Plot silhouette scores.
        for i, var in enumerate([k_sil, c_sil]):
            # Skip some stuff if we only have 1 iteration.
            if var.shape[1] == 1:
                m = var[:,0]
                s = None
            else:
                # Compute the average silhouette score.
                m = numpy.nanmean(var, axis=1)
                # Compute the 95% confidence interval.
                sd = numpy.nanstd(var, axis=1)
                sem = sd / numpy.sqrt(var.shape[1])
                ci = 1.96 * sem
                # Compute how often the analysis would identify clustering, and
                # how often it identified k.
                if i == 0:
                    acc = numpy.nanmean(k_acc)
                    sen = numpy.nanmean(k_sen)
                elif i == 1:
                    acc = numpy.nanmean(c_acc)
                    sen = numpy.nanmean(c_sen)
                # Annotate the clustering identification scores, but only
                # for differences that are in the interesting range
                s = None
                if (d == 0) or (1.5 < d < 4.5):
                    s = r"$\Delta$={}; $p(k>1)$={}".format(\
                        round(d, ndigits=1), round(acc, ndigits=2))
                    if k > 1:
                        s += r"; $p(k)$={}".format(round(sen, ndigits=2))
            # Plot the mean and confidence interval.
            ax[i,ki].plot(k_guess_range, m, '-', lw=3, color=col, alpha=0.5, label=s)
            if var.shape[1] > 1:
                ax[i,ki].fill_between(k_guess_range, m-ci, m+ci, color=col, alpha=0.3)

    # Add legends.
    ax[0,ki].legend(loc="lower right")
    ax[1,ki].legend(loc="lower right")

# Set axis titles.
ax[0,0].set_ylabel("K-means silhouette", fontsize=24)
ax[1,0].set_ylabel("C-means silhouette", fontsize=24)
# Set axis limits and ticks.
for i in range(ax.shape[0]):
    for j in range(ax.shape[1]):
        if i == 1:
            ax[i,j].set_xlabel(r"$k_{guess}$", fontsize=24)
        ax[i,j].tick_params(labelsize=14)
        ax[i,j].set_xticks(k_guess_range)
        ax[i,j].set_xlim(1, numpy.max(k_guess_range)+1)
        ax[i,j].set_ylim(-0.2, 1.1)
        ax[i,j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# Set a colour bar on the right-most axes.
for i in range(ax.shape[0]):
    divider = make_axes_locatable(ax[i,-1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, \
        orientation="vertical")
    cbar.set_ticks([0.0, 10.0])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(r"Cluster separation $\Delta$", fontsize=24)

# Save figure.
fig.savefig(os.path.join(OUTDIR, "fig_03_silhouette_comparison.png"))
pyplot.close(fig)


#
## Apply c-means.
#cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(numpy.transpose(X_), k, \
#    2, error=0.005, maxiter=1000, init=None)
## Get hard cluster membership.
#y = numpy.argmax(u, axis=0)
## Compute silhouette scores.
#s_fuzzy, sil = fuzzy_silhouette_coefficient(X_, y, numpy.transpose(u), alpha=1.0)
#s_hard = numpy.mean(sil)
#
#pyplot.figure()
#for lbl in range(u.shape[0]):
#    # Create a colour array with varrying transparency.
#    rgb = matplotlib.colors.to_rgb(PLOTCOLS[lbl])
#    col = numpy.zeros((X_.shape[0],4), dtype=float)
#    col[:,0:3] = rgb
#    col[:,3] = u[lbl,:]
#    # Draw the points.
#    pyplot.scatter(X_[:,0], X_[:,1], color=col)
#
#
## Try a few different c-means options.
#k_guess_range = range(2,11)
#sil_fuzz = numpy.zeros(len(k_guess_range))
#sil_hard = numpy.zeros(len(k_guess_range))
#fpc = numpy.zeros(len(k_guess_range))
#for i, k_guess in enumerate(k_guess_range):
#    # Apply c-means.
#    cntr, u, u0, d, jm, p, fpc[i] = skfuzzy.cluster.cmeans(numpy.transpose(X_), \
#        k_guess, 2, error=0.005, maxiter=1000, init=None)
#    # Get hard cluster membership.
#    y = numpy.argmax(u, axis=0)
#    # Compute silhouette scores.
#    sil_fuzz[i], sil = fuzzy_silhouette_coefficient(X_, y, numpy.transpose(u), \
#        alpha=1.0)
#    sil_hard[i] = numpy.mean(sil)
#
#pyplot.figure()
#pyplot.plot(k_guess_range, sil_fuzz, 'o-', label="Fuzzy")
#pyplot.plot(k_guess_range, sil_hard, 'o-', label="Hard")
#pyplot.plot(k_guess_range, fpc, 'o-', label="FPC?")
#pyplot.legend(loc="upper right")
