import copy

import numpy
from scipy.stats import pearsonr
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_sample(n_observations, n_features, n_different, d_different, \
    p_group, min_cov=-0.5, max_cov=0.5, cov_matrix=None):
    
    """
    desc:
        Generates a sample of (n_observations, n_features) in which n_different
        features show differences between two equal-sized clusters. The
        difference will be of a magnitude of a Cohen's d of d_different.
    
    Arguments:
        n_observations:
            desc:
                Number of observations in the sample.
            type:
                int

        n_features:
            desc:
                Number of features for each observation.
            type:
                int
        
        n_different:
            desc:
                Number of features in which two clusters will be different
                from each other. Value should be in range [0, n_observations]
            type:
                int

        d_different:
            desc:
                Effect size (Cohen's d) of the difference within each sample
                that shows a difference between the two groups.
            type:
                float
        
        p_group:
            desc:
                Proportion of the total sample in one cluster. The other
                cluster will contain a proportion of 1-p_group of the total
                sample size. Can also be a list of floats, with one entry for
                each sample, e.g. (0.5,0.5) or (0.33, 0.34, 0.33). Currently
                up to three clusters are supported.
            type:
                float or list
    
    Returns:
        [X, y, cov_matrix]:
            desc:
                X is a numpy array with shape (n_observations, n_features)
                that contains normally distributed data within each
                non-different features, and bimodally distributed data for
                each different feature.
                y is a numpy array with shape (n_observations) that is 0 for
                the members of one cluster, and 1 for the members of the other
                cluster. This is the ground truth for comparing clustering
                results against.
                cov_matrix is a numpy array with shape (n_features, n_features)
                that contains the covariance matrix of the data before the
                group differences were introduced.
            type:
                list
    """
    
    # Compute sizes for each of the clusters.
    if type(p_group) is float:
        sample_n = [int(numpy.round(n_observations*p_group, decimals=0))]
        sample_n.append(int(n_observations-sample_n[0]))
    else:
        sample_n = []
        for p in p_group:
            sample_n.append(int(numpy.round(n_observations*p, decimals=0)))
    # Correct number of observations.
    n_observations = numpy.nansum(sample_n)
    # Create ground truth cluster labels.
    y_truth = numpy.ones(n_observations, dtype=int) * -1
    si = 0
    for i in range(len(sample_n)):
        y_truth[si:si+sample_n[i]] = i
        si += sample_n[i]
    
    # Create a covariance matrix if none was provided.
    if cov_matrix is None:
        cov_matrix = numpy.zeros((n_features, n_features), dtype=float)
        for i in range(cov_matrix.shape[0]):
            for j in range(i+1, cov_matrix.shape[1]):
                cov_matrix[i,j] = min_cov + numpy.random.rand() * max_cov
                cov_matrix[j,i] = cov_matrix[i,j]
        # Set diagonal to 1.0
        di = numpy.diag_indices(n_features, ndim=2)
        cov_matrix[di] = 1.0

    # If the lowest eigenvalue is negative, the covariance matrix isn't
    # symmetric positive semi-definite. This needs to be corrected.
    if len(cov_matrix.shape) > 2:
        for i in range(cov_matrix.shape[2]):
            min_eig = numpy.min(numpy.real(numpy.linalg.eigvals(cov_matrix[:,:,i])))
            if min_eig < 0:
                # Correct the covariance matrix.
                cov_matrix[:,:,i] -= 10*min_eig \
                    * numpy.eye(*cov_matrix[:,:,i].shape)
                # Transform back to a maximum of 1.0
                cov_matrix /= numpy.max(cov_matrix[:,:,i])
    else:
        min_eig = numpy.min(numpy.real(numpy.linalg.eigvals(cov_matrix)))
        if min_eig < 0:
            # Correct the covariance matrix.
            cov_matrix -= 10*min_eig * numpy.eye(*cov_matrix.shape)
            # Transform back to a maximum of 1.0
            cov_matrix /= numpy.max(cov_matrix)

    # Determine what features will be different.
    diff_i = range(n_features)
    numpy.random.shuffle(diff_i)
    diff_i = diff_i[:n_different]
    
    # Create a vector of feature means for each cluster.
    if len(sample_n) == 1:
        # Create the means for this one cluster at 0.
        m = numpy.zeros((n_features,1), dtype=float)
    elif len(sample_n) == 2:
        # Start with means for boths cluster at 0.
        m = numpy.zeros((n_features,2), dtype=float)
        # Randomly generate directions.
        direction = numpy.random.rand(len(diff_i))
        direction[direction<0.5] = -1
        direction[direction>0] = 1
        # Change the cluster means away from each other by d_different.
        m[diff_i,0] += (d_different/2.0) * direction
        m[diff_i,1] -= (d_different/2.0) * direction
    elif len(sample_n) == 3:
        # Start with means for all cluster at 0.
        m = numpy.zeros((n_features, 3), dtype=float)
        # Randomly generate directions.
        direction = numpy.random.rand(len(diff_i))
        direction[direction<0.5] = -1
        direction[direction>0] = 1
        # Keep one cluster at 0, and move the others to d_different in
        # opposite directions.
        m[diff_i,0] += d_different * direction
        m[diff_i,2] -= d_different * direction
    
    # Simulate the dataset.
    X_raw = numpy.zeros((n_observations,n_features), dtype=float)
    for i in range(m.shape[1]):
        cm = numpy.copy(cov_matrix)
        if len(cov_matrix.shape) > 2:
            if cov_matrix.shape[2] > 1:
                cm = numpy.copy(cov_matrix[:,:,i])
        X_raw[y_truth==i,:] = numpy.random.multivariate_normal(m[:,i], cm, \
            size=sample_n[i])

    return X_raw, y_truth, cov_matrix


def create_equidistant_sample(n_observations=100, d=3.0, p=(0.33,0.34,0.33), cov_matrix=None):
    
    # Compute k.
    k = len(p)
    
    # Set the number of features.
    n_features = 2
    
    # Determine the size of each cluster.
    sample_n = []
    for p_ in p:
        sample_n.append(int(numpy.round(n_observations*p_, decimals=0)))
    # Recompute the actual sample size.
    n = sum(sample_n)
    # Create the ground truth cluster membership.
    y_truth = numpy.ones(n, dtype=int) * -1
    si = 0
    for i in range(len(sample_n)):
        y_truth[si:si+sample_n[i]] = i
        si += sample_n[i]
    # Create covariance matrix.
    if cov_matrix is None:
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
    X = numpy.zeros((n,m.shape[0]), dtype=float)
    for i in range(m.shape[1]):
        X[y_truth==i,:] = numpy.random.multivariate_normal(m[:,i], \
            cov_matrix, size=sample_n[i])
    
    return X, y_truth, cov_matrix


def predict_D(d, n):
    
    """Predicts cluster separation D on the basis of within-feature effect
    size d and number of different features n.
    """
    
    return numpy.sqrt(numpy.nansum((numpy.ones(n, dtype=float) * d)**2))


def corr_matrix(X):
    r = numpy.zeros((X.shape[1],X.shape[1]), dtype=float)
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            r[i,j] = pearsonr(X[:,i], X[:,j])[0]
    return r


def dD_plot(fig, ax, m, xticklabels, yticklabels):

    # Plot the heatmap.
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=10)
    # Add colourbar.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label(r"Centroid distance $\Delta$", fontsize=16)
    # Add axis ticks.
    ax.set_xticks(numpy.arange(0, m.shape[1]+1))
    ax.set_xticklabels(xticklabels, fontsize=12)
    ax.set_yticks(numpy.arange(0, m.shape[0]+1))
    ax.set_yticklabels(yticklabels, fontsize=12)
    # Set axis limits.
    ax.set_xlim(-0.5, m.shape[1]-0.5)
    ax.set_ylim(-0.5, m.shape[0]-0.5)
    # Add axis labels.
    ax.set_xlabel(r"Effect size $\delta$", fontsize=16)
    ax.set_ylabel(r"Proportion of different features", fontsize=16)
    # Annotate individual values.
    for col in range(m.shape[1]):
        for row in range(m.shape[0]):
            ax.annotate(str(numpy.round(m[row,col], decimals=1)), \
                (col-0.1,row-0.05), color="#FFFFFF", fontsize=12)
    
    return ax


def fuzzy_silhouette_coefficient(X, y, u=None, alpha=1.0):
    
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
        d = numpy.sqrt(numpy.nansum((X[sel,:] - X[i,:])**2, axis=1))
        # Compute the distance between the current sample and its cluster
        # members.
        a = numpy.nanmean(d[y[sel]==y[i]])
        # Compute the distance between the current sample and all other
        # clusters' members.
        b_min = numpy.inf
        for lbl in labels:
            if y[i] != lbl:
                b = numpy.nanmean(d[y[sel]==lbl])
                if b < b_min:
                    b_min = copy.copy(b)
        # Compute the cluster silhouette for this sample.
        s[i] = (b_min - a) / max(a, b_min)
    
    if u is None:
        s_m = numpy.nanmean(s)
    else:
        # Find largest and second-largest elements for all samples.
        u_sorted = numpy.sort(u, axis=1)
        u_p = u_sorted[:,-1]
        u_q = u_sorted[:,-2]
        # Compute the fuzzy silhouette score.
        s_m = numpy.nansum(((u_p-u_q)**alpha) * s) / numpy.nansum((u_p-u_q)**alpha)

    return s_m, s