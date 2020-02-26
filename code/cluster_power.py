import os
import copy
import time
import itertools

import numpy
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.metrics import adjusted_rand_score, calinski_harabaz_score, \
    silhouette_samples, silhouette_score

import skfuzzy

from cluster import clustering, cluster_comparison, convenience_clustering, \
    correlation_matrix, dim_reduction, plot_averages, plot_clusters, \
    plot_samples, plot_silhouette, preprocess

from data_generation_playground import create_sample, dD_plot, \
    fuzzy_silhouette_coefficient, predict_D


# # # # #
# CONSTANTS

# SWITCHES
# Redo covariance matrix generation.
OVERWRITE_COVARIANCES = False
# Redo population data generation.
OVERWRITE_DATA = False
# Redo dimensionality reduction.
OVERWRITE_REDUCED_DATA = False
# Redo d-D plots.
OVERWRITE_dDPLOTS = False
# Redo cluster analysis.
OVERWRITE_CLUSTER_DATA = False
# Redo cluster coefficient overview plots.
OVERWRITE_CLUSTER_OVERVIEW_PLOTS = False
# Redo power and accuracy simulation.
OVERWRITE_POWER = False


# POPULATION DATA
# Number of observations in each "population" file. (Sub-samples will be
# pulled from these files.)
N_OBSERVATIONS = 1000
# Number of features in each "population" file.
N_FEATURES = 15
# Number of different features in each population file. This should be a list
# of integer numbers between 0 and N_FEATURES (inclusive on both ends).
N_DIFFERENT_FEATURES = [1, 5, 10, 15]
# Difference between cluster means within each different feature. Because
# we're using z-scored space, this is Cohen's d. Should be a list of floats.
M_DIFFERENCE = [0.3, 0.5, 0.8, 1.3, 2.1]
# Proportion of observations in each cluster. List of tuples, each of which
# with up to three proportions that add up to 1.
P_CLUSTER = [(0.1,0.9), (0.5,0.5), (0.33, 0.34, 0.33)]

# COVARIANCE MATRICES
COVARIANCE_TYPES = ["none", "random-01", "random-02", "3-factor", "4-factor", \
    "mixed-01", "mixed-02"]
COVARIANCE_MIXES = { \
    "mixed-01": ["3-factor", "4-factor", "random-01"], \
    "mixed-02": ["random-01", "random-02", "none"], \
    }
# Limits for correlations between random features.
COV_MIN = -0.3
COV_MAX = 0.3
# Limits for correlations between features within the same factor.
COV_HIGH_MIN = 0.4
COV_HIGH_MAX = 0.9
# Probability of correlations between features within the same factor to be
# negative.
COV_HIGH_NEGATIVE_P = 0.5

# DIMENSIONALITY REDUCTION
DIM_REDUCTION_ALGORITHMS = ["MDS", "UMAP"]
DIM_REDUCTION_N_COMPONENTS = 2

# CLUSTERING
# Specify for what files cluster output should be generated, or None to run
# clustering on all dim-reduced files.
FILES_FOR_CLUSTERING = None
CLUSTER_ALGORITHMS = ["KMEANS", "HDBSCAN", "WARD", "COSINE"]
CLUSTER_K = { \
    "KMEANS":   range(2,6), \
    "WARD":     range(2,6), \
    "COSINE":   range(2,6), \
    "CMEANS":   range(2,6), \
    "HDBSCAN":  [2], \
    }

# POWER AND ACCURACY
# The number of resampling iterations to compute the standard error of the
# mean of cluster outcome measures.
SIM_N_ITERATIONS_PER_CELL = 100
# The number of observations to be sampled from simulated populations.
SIM_N = [10, 20, 40, 80, 160]
# Number of clusters and their proportional sizes in simulated populations.
SIM_P = [(0.1,0.9), (0.5,0.5), (0.33,0.34,0.33), (0.25, 0.25, 0.25, 0.25)]
# Distances between simulated populations.
SIM_D = range(1,11)
# Set the simulation covariance type.
SIM_COV_TYPE = "3-factor"
# Cluster algorithms to be run on simulations for power.
SIM_CLUSTER_METHODS = ["CMEANS"] #["HDBSCAN", "KMEANS", "CMEANS"]

# FILES AND FOLDERS
# Auto-detect the current directory.
DIR = os.path.dirname(os.path.abspath(__file__))
# Construct paths to data directories.
DATADIR = os.path.join(DIR, "data")
COV_DIR = os.path.join(DATADIR, "covariance")
DAT_DIR = os.path.join(DATADIR, "data")
DIM_DIR = os.path.join(DATADIR, "dim-reduced_data")
CLU_DIR = os.path.join(DATADIR, "cluster_output")
# Construct paths to output directories.
OUTDIR = os.path.join(DIR, "output")
dD_DIR = os.path.join(OUTDIR, "d-D_plots")
D_DISCREP_DIR = os.path.join(OUTDIR, "centroid_separation_discrepancy_plots")
CPLOT_DIR = os.path.join(OUTDIR, "cluster_plots")
SIM_DIR = os.path.join(OUTDIR, "power_accuracy_simulation")

# Create directories if necessary.
for dir_path in [DATADIR, COV_DIR, DAT_DIR, DIM_DIR, CLU_DIR, OUTDIR, dD_DIR, \
    D_DISCREP_DIR, CPLOT_DIR, SIM_DIR]:

    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


# # # # #
# COVARIANCE CONSTRUCTION

# Create covariance matrices.
for ci, cov_type in enumerate(COVARIANCE_TYPES):
    
    # Construct the file path.
    fpath = os.path.join(COV_DIR, "{}.csv".format(cov_type))

    # Only continue if the file does not exist, or should be overwritten.
    if not os.path.isfile(fpath) or OVERWRITE_COVARIANCES:
        
        # Skip the "mixed" covariance types (they'll be recombinations).
        if "mixed" in cov_type:
            continue
        
        print("Constructing covariance matrix {} ({}/{})".format( \
            cov_type, ci+1, len(COVARIANCE_TYPES)))
        
        # Create an empty matrix.
        cov_matrix = numpy.zeros((N_FEATURES, N_FEATURES), dtype=float)
        
        # Create a covariance structure.
        if cov_type in ["none", "independent", "zero"]:
            # Set covariance to 0.
            cov_matrix *= 0.0

        elif "random" in cov_type:
            # Loop through all pairs of variables.
            for i in range(cov_matrix.shape[0]):
                for j in range(i+1, cov_matrix.shape[1]):
                    # Set this variable pair to a random covariance.
                    cov_matrix[i,j] = numpy.random.rand() * (COV_MAX-COV_MIN) \
                        + COV_MIN
                    cov_matrix[j,i] = cov_matrix[i,j]

        elif "factor" in cov_type:
            # Compute the number of features per factor.
            n_factors = int(cov_type[:cov_type.find("-")])
            features_per_factor = int(numpy.floor( \
                float(N_FEATURES) / float(n_factors)))
            # Construct a vector to store which features are related to what
            # factor.
            factor_membership = numpy.zeros(N_FEATURES)
            for i in range(n_factors):
                si = i * int(features_per_factor)
                if i == n_factors-1:
                    factor_membership[si:] = i
                else:
                    factor_membership[si:si+features_per_factor] = i
                
            for i in range(cov_matrix.shape[0]):
                for j in range(i+1, cov_matrix.shape[1]):
                    # If these features are in the same factor, choose a high
                    # correlation.
                    if factor_membership[i] == factor_membership[j]:
                        r = numpy.random.rand() * (COV_HIGH_MAX-COV_HIGH_MIN) \
                            + COV_HIGH_MIN
                        # Flip the sign of the correlation for some features.
                        if numpy.random.rand() < COV_HIGH_NEGATIVE_P:
                            r *= -1
                    # If these features are not in the same factor, choose a
                    # random correlation value.
                    else:
                        r = numpy.random.rand() * (COV_MAX-COV_MIN) + COV_MIN
                    # Save the chosen correlation value.
                    cov_matrix[i,j] = r
                    cov_matrix[j,i] = cov_matrix[i,j]

        # Set diagonal to 1.0
        di = numpy.diag_indices(N_FEATURES, ndim=2)
        cov_matrix[di] = 1.0
        
#        pyplot.figure()
#        pyplot.imshow(cov_matrix, vmin=-1, vmax=1, cmap="coolwarm")
#        pyplot.title(cov_type)
        
        # Save the covariance matrix in a CSV file.
        with open(fpath, "w") as f:
            for i in range(cov_matrix.shape[0]):
                if i == 0:
                    line = ""
                else:
                    line = "\n"
                line += ",".join(map(str, list(cov_matrix[i,:])))
                f.write(line)

    # Do nothing if the matrix already exists and shouldn't be overwritten.
    else:
        print("Covariance matrix {} ({}/{}) already exists".format( \
            cov_type, ci+1, len(COVARIANCE_TYPES)))


# # # # #
# DATA GENERATION

# Loop through all datasets that should be generated.
current_iteration = 0
total_iterations = len(COVARIANCE_TYPES) * len(N_DIFFERENT_FEATURES) \
    * len(M_DIFFERENCE) * len(P_CLUSTER)
for ci, cov_type in enumerate(COVARIANCE_TYPES):
    for ni, n_diff in enumerate(N_DIFFERENT_FEATURES):
        for di, m_diff in enumerate(M_DIFFERENCE):
            for pi, p in enumerate(P_CLUSTER):
                
                # Construct the file path.
                fname = "{}_{}-features_{}-diff".format(cov_type, \
                    n_diff, str(round(m_diff, ndigits=2)).replace(".",""))
                for i in range(len(p)):
                    fname += "_{}-p{}".format(int(round(100*p[i])),i)
                fpath = os.path.join(DAT_DIR, "{}.csv".format(fname))
                
                # Check if the file exists and needs to be overwritten.
                if not os.path.isfile(fpath) or OVERWRITE_DATA:
        
                    print("Constructing data file {} ({}/{})".format( \
                        fname, current_iteration+1, total_iterations))
                    
                    # Load the covariance matrix.
                    if "mixed" in cov_type:
                        cov = numpy.zeros((N_FEATURES,N_FEATURES,len(p)), \
                            dtype=float) * numpy.NaN
                        for i in range(len(p)):
                            cov_path = os.path.join(COV_DIR, "{}.csv".format( \
                                COVARIANCE_MIXES[cov_type][i]))
                            cov[:,:,i] = numpy.loadtxt(cov_path, dtype=float, \
                                delimiter=",", unpack=True)
                    else:
                        cov_path = os.path.join(COV_DIR, "{}.csv".format(cov_type))
                        cov = numpy.loadtxt(cov_path, dtype=float, delimiter=",", \
                            unpack=True)
                    
                    # Generate the data.
                    X, y_truth, _ = create_sample(N_OBSERVATIONS, \
                        N_FEATURES, n_diff, m_diff, p, cov_matrix=cov)
                    
                    # Write sample to file.
                    with open(fpath, "w") as f:
                        header = ["cluster"]
                        header.extend(["feature_{}".format(i+1) for i in range(N_FEATURES)])
                        f.write(",".join(map(str, header)))
                        for i in range(X.shape[0]):
                            line = [y_truth[i]]
                            line.extend(X[i,:])
                            f.write("\n" + ",".join(map(str, line)))
                    
                    # Unreference data.
                    del X, y_truth
                
                # Don't overwrite existing files if not specifically requested.
                else:
                    print("Data file {} ({}/{}) already exists".format( \
                        fname, current_iteration+1, total_iterations))
                
                # Update the iteration number.
                current_iteration += 1


# # # # #
# DIMENSIONALITY REDUCTION

# Auto-detect all datasets.
all_files = os.listdir(DAT_DIR)
all_files.sort()

# Loop through all datasets.
current_iteration = 0
total_iterations = len(all_files) * len(DIM_REDUCTION_ALGORITHMS)
for fi, fname in enumerate(all_files):
    # Loop through all dimensionality reduction algorithms.
    for di, dim_reduction_method in enumerate(DIM_REDUCTION_ALGORITHMS):
        
        # Construct the data file path.
        fpath_in = os.path.join(DAT_DIR, fname)

        # Construct the file path for the reduced data.
        fpath_out = os.path.join(DIM_DIR, "{}_{}.csv".format(\
            os.path.splitext(fname)[0], dim_reduction_method))
        
        # Create new or overwrite existing data.
        if not os.path.isfile(fpath_out) or OVERWRITE_REDUCED_DATA:
        
            print("Reducing data using {} in file {} ({}/{})".format( \
                dim_reduction_method, fname, current_iteration+1, \
                total_iterations))
            t0 = time.time()
            
            # Load data.
            X = numpy.loadtxt(fpath_in, dtype=float, delimiter=",", \
                skiprows=1, unpack=False)
            y_truth = X[:,0].astype(int)
            X = X[:,1:]
            
            # Reduce data dimensionality.
            X_red = dim_reduction(X, n_components=DIM_REDUCTION_N_COMPONENTS, \
                mode=dim_reduction_method)
            
            # Store data.
            with open(fpath_out, "w") as f:
                header = ["cluster"]
                header.extend(["dim_{}".format(i+1) for i in range(X_red.shape[1])])
                f.write(",".join(map(str, header)))
                for i in range(X_red.shape[0]):
                    line = [y_truth[i]]
                    line.extend(X_red[i,:])
                    f.write("\n" + ",".join(map(str, line)))
            
            print("\tDone in {} seconds".format( \
                round(time.time()-t0, ndigits=2)))

        # Don't overwrite existing files if not specifically requested.
        else:
            print("Data in file {} ({}/{}) already reduced".format( \
                fname, current_iteration+1, total_iterations))
        
        current_iteration += 1


# # # # #
# d-D PLOTS

# Create a dict to record cluster centroid separation.
centroid_D = {}

# Loop through all dimensionality reduction algorithms.
dim_reduction_methods = [None]
dim_reduction_methods.extend(DIM_REDUCTION_ALGORITHMS)
current_iteration = 0
total_iterations = len(dim_reduction_methods) * len(COVARIANCE_TYPES) \
    * len(P_CLUSTER)
for di, dim_reduction_method in enumerate(dim_reduction_methods):
    # Loop through all covariance types.
    for ci, cov_type in enumerate(COVARIANCE_TYPES):
        # Loop through all cluster numbers.
        for pi, p in enumerate(P_CLUSTER):
            
            current_iteration += 1
            
            # Construct the file path for the reduced data.
            fname_out = "{}".format(cov_type)
            for i in range(len(p)):
                fname_out += "_{}-p{}".format(int(round(100*p[i])),i)
            if dim_reduction_method is None:
                fname_out += "_d-D_plot.png"
            else:
                fname_out += "_{}_d-D_plot.png".format(dim_reduction_method)
            fpath_out = os.path.join(dD_DIR, fname_out)
            
            # Only continue if the file needs to be created or overwritten.
            if os.path.isfile(fpath_out) and not OVERWRITE_dDPLOTS:
                print("d-D plot {} ({}/{}) already created".format( \
                    fname_out, current_iteration, total_iterations))
                continue

            print("Creating d-D plot {} ({}/{})".format( \
                fname_out, current_iteration, total_iterations))

            # Create an empty d-D matrix.
            dims = (len(N_DIFFERENT_FEATURES), len(M_DIFFERENCE))
            m = numpy.zeros(dims, dtype=float) * numpy.NaN

            # Loop through all datasets.
            for ni, n_diff in enumerate(N_DIFFERENT_FEATURES):
                for di, m_diff in enumerate(M_DIFFERENCE):

                    # Construct the file name.
                    fname = "{}_{}-features_{}-diff".format(cov_type, \
                        n_diff, str(round(m_diff, ndigits=2)).replace(".",""))
                    for i in range(len(p)):
                        fname += "_{}-p{}".format(int(round(100*p[i])),i)
                    if dim_reduction_method is not None:
                        fname += "_{}".format(dim_reduction_method)
                    
                    # Construct the data file path.
                    if dim_reduction_method is None:
                        fpath_in = os.path.join(DAT_DIR, "{}.csv".format(fname))
                    else:
                        fpath_in = os.path.join(DIM_DIR, "{}.csv".format(fname))

                    # Load data.
                    X = numpy.loadtxt(fpath_in, dtype=float, delimiter=",", \
                        skiprows=1, unpack=False)
                    y_truth = X[:,0].astype(int)
                    X = X[:,1:]
                
                    # Compute the ground-truth distance between two clusters.
                    # (In the case of k=3, the y_truth==1 cluster will be the
                    # middle cluster with m=0 for all features. It will is the
                    # nearest to the other clusters, and thus the current
                    # calculation represents the shortest centroid distance,
                    # analogous to the logic of silhouette scores.)
                    n_features = X.shape[1]
                    d = numpy.arange(n_features, dtype=float)
                    for i in range(n_features):
                        d[i] = numpy.nanmean(X[y_truth==0,i]) \
                            - numpy.nanmean(X[y_truth==1,i])
                    m[ni,di] = numpy.sqrt(numpy.nansum(d**2))
            
            # Create a new entry in the centroid separation dict.
            centroid_D[fname_out.replace("_d-D_plot.png", "")] \
                = numpy.copy(m)

            # Plot d-D matrix.
            # Create a new figure.
            fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0, 6.0), \
                dpi=300.0)
            # Create ticklabels.
            xticklabels = map(str, M_DIFFERENCE)
            yticklabels = map(str, numpy.round( \
                numpy.array(N_DIFFERENT_FEATURES, dtype=float) \
                / numpy.max(N_DIFFERENT_FEATURES), decimals=2))
            ax = dD_plot(fig, ax, m, xticklabels, yticklabels)
            # Save and close the figure.
            fig.savefig(fpath_out)
            pyplot.close(fig)


# # # # #
# d-D DISCREPANCY

PLOTCOLS = { \
    "None":     "#4e9a06", \
    "MDS":      "#204a87", \
    "UMAP":     "#5c3566", \
    }

# Only do this if the dD plot info has been generated.
if len(centroid_D.keys()) > 0:

    # Create a new figure.
    all_fig, all_ax = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0, 6.0), \
        dpi=900.0)

    # Auto-detect all datasets.
    all_names = []
    for name in centroid_D.keys():
        if ("MDS" not in name) and ("UMAP" not in name) and (name is not "truth"):
            all_names.append(name)
    all_names.sort()
    
    # Create an d-D matrix for the population centroid distances.
    dims = (len(N_DIFFERENT_FEATURES), len(M_DIFFERENCE))
    centroid_D["truth"] = numpy.zeros(dims, dtype=float) * numpy.NaN
    # Loop through all used numbers of different features and effect sizes.
    for ni, n_diff in enumerate(N_DIFFERENT_FEATURES):
        for di, m_diff in enumerate(M_DIFFERENCE):
            centroid_D["truth"][ni,di] = predict_D(m_diff, n_diff)
    # Create a variable with the same data reshaped into a vector.
    x = centroid_D["truth"].reshape(centroid_D["truth"].size)
    
    # Loop through all datasets.
    current_iteration = 0
    total_iterations = len(all_names)
    for fi, name in enumerate(all_names):
        
        print("Plotting centroid discrepancy for {} ({}/{})".format( \
            name, current_iteration+1, total_iterations))
        current_iteration += 1

        # Create a new figure.
        fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0, 6.0), \
            dpi=300.0)
        
        # Loop through all dimensionality reduction algorithms.
        for di, dim_reduction_method in enumerate(dim_reduction_methods):
            
            # Construct the key for the current data.
            key = copy.deepcopy(name)
            if dim_reduction_method is not None:
                key += "_{}".format(dim_reduction_method)
                
            # Grab the data.
            y = centroid_D[key].reshape(centroid_D[key].size)
            
            # Plot the data against the theoretical ground-truth centroid
            # distance in the population.
            ax.plot(x, y, 'o', color=PLOTCOLS[str(dim_reduction_method)], \
                label=str(dim_reduction_method))
            
            # Plot the data points in the overall figure.
            if fi == 0:
                lbl = str(dim_reduction_method)
            else:
                lbl = None
            all_ax.plot(x, y, 'o', color=PLOTCOLS[str(dim_reduction_method)], \
                label=lbl, alpha=0.3)
        
        # Set limits and draw a line through x=y.
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        lim = max(xlim[1], ylim[1])
        ax.plot(numpy.arange(0.0, lim+0.1, 0.1), \
            numpy.arange(0.0, lim+0.1, 0.1), '--', color="#000000", alpha=0.3)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Finish the plot.
        ax.legend(loc="lower right", fontsize=14)
        ax.set_xlabel(r"Population centroid distance $\Delta$", fontsize=16)
        ax.set_ylabel(r"Sample centroid distance $\Delta$", fontsize=16)
        # Save the plot.
        fpath_out = os.path.join(D_DISCREP_DIR, "{}.png".format(name))
        fig.savefig(fpath_out)
        pyplot.close(fig)

    # Set limits and draw a line through x=y.
    xlim = all_ax.get_xlim()
    ylim = all_ax.get_ylim()
    lim = max(xlim[1], ylim[1])
    all_ax.plot(numpy.arange(0.0, lim+0.1, 0.1), \
        numpy.arange(0.0, lim+0.1, 0.1), '--', color="#000000", alpha=0.3)
    all_ax.set_xlim(xlim)
    all_ax.set_ylim(ylim)
    # Finish the plot.
    all_ax.legend(loc="upper left", fontsize=14)
    all_ax.set_xlabel(r"Population centroid distance $\Delta$", fontsize=16)
    all_ax.set_ylabel(r"Sample centroid distance $\Delta$", fontsize=16)
    # Save the plot.
    fpath_out = os.path.join(D_DISCREP_DIR, "ZOMG_ALL_DISCREPANCIES.png")
    all_fig.savefig(fpath_out)
    pyplot.close(all_fig)


# # # # #
# CLUSTERED DATASETS

# Auto-detect all dimensionality-reduced files.
if FILES_FOR_CLUSTERING is None:
    all_files = os.listdir(DIM_DIR)
    all_files.extend(os.listdir(DAT_DIR))
    all_files.sort()
else:
    all_files = FILES_FOR_CLUSTERING

# Run through all cluster methods.
current_iteration = 0
total_iterations = len(CLUSTER_ALGORITHMS) * len(all_files)
for ci, cluster_method in enumerate(CLUSTER_ALGORITHMS):
    # Run through all files.
    for fi, fname in enumerate(all_files):
        
        current_iteration += 1
        
        # Construct output file path.
        fname_out = "{}_{}.csv".format( \
            os.path.splitext(fname)[0], cluster_method)
        fpath_out = os.path.join(CLU_DIR, fname_out)
        
        # Only continue if the output file does not exist yet, or should be
        # overwritten.
        if os.path.isfile(fpath_out) and not OVERWRITE_CLUSTER_DATA:
            print("Clustering {} output for {} ({}/{}) already created".format( \
                cluster_method, fname_out, current_iteration, total_iterations))
            continue
        print("Clustering using {} for {} ({}/{})".format( \
            cluster_method, fname_out, current_iteration, total_iterations))
        t0 = time.time()

        # Construct data file path (make sure there's a CSV extension).
        fname_in = "{}.csv".format(os.path.splitext(fname)[0])
        # Make sure to load from the correct directory (either dim-reduced or
        # raw).
        dim_reduced = False
        for dim_reduction_method in DIM_REDUCTION_ALGORITHMS:
            if dim_reduction_method in fname_in:
                dim_reduced = True
        if dim_reduced:
            fpath_in = os.path.join(DIM_DIR, fname_in)
        else:
            fpath_in = os.path.join(DAT_DIR, fname_in)

        # Load data.
        X = numpy.loadtxt(fpath_in, dtype=float, delimiter=",", \
            skiprows=1, unpack=False)
        y_truth = X[:,0].astype(int)
        X = X[:,1:]
        
        # Create empty matrices to hold the outcomes in.
        y = numpy.zeros((X.shape[0],len(CLUSTER_K[cluster_method])), dtype=int) \
            * numpy.NaN
        s = numpy.zeros((X.shape[0],len(CLUSTER_K[cluster_method])), dtype=float) \
            * numpy.NaN

        # Run through the requested numbers of clusters.
        for ki, k in enumerate(CLUSTER_K[cluster_method]):
            # Run cluster analysis.
            y[:,ki] = clustering(X, mode=cluster_method, n_clusters=k)
            # Compute a silhouette coefficient for each sample, but only if
            # more than one cluster was detected.
            if len(numpy.unique(y[:,ki])) == 1:
                s[:,ki] = 0.0
            else:
                s[:,ki] = silhouette_samples(X, y[:,ki], metric='euclidean')
        
        # Write to file.
        with open(fpath_out, "w") as f:
            header = ["cluster"]
            for ki, k in enumerate(CLUSTER_K[cluster_method]):
                header.append("y_k={}".format(k))
                header.append("s_k={}".format(k))
            f.write(",".join(map(str,header)))
            for i in range(X.shape[0]):
                line = [y_truth[i]]
                for ki, k in enumerate(CLUSTER_K[cluster_method]):
                    line.append(y[i,ki])
                    line.append(s[i,ki])
                f.write("\n" + ",".join(map(str,line)))

        print("\tDone in {} seconds".format( \
            round(time.time()-t0, ndigits=2)))


# # # # #
# CLUSTER PLOTS

# Loop through all cluster and dimensionality reduction algorithms.
current_iteration = 0
total_iterations = len(CLUSTER_ALGORITHMS) * (len(DIM_REDUCTION_ALGORITHMS)+1) \
    * len(COVARIANCE_TYPES) * len(P_CLUSTER)
for cmi, cluster_method in enumerate(CLUSTER_ALGORITHMS):
    dim_reduction_methods = [None]
    dim_reduction_methods.extend(DIM_REDUCTION_ALGORITHMS)
    for di, dim_reduction_method in enumerate(dim_reduction_methods):
        # Loop through all covariance types.
        for ci, cov_type in enumerate(COVARIANCE_TYPES):
            # Loop through all cluster numbers.
            for pi, p in enumerate(P_CLUSTER):
                
                current_iteration += 1
                
                # Construct the file path for the plots.
                fname_out = "{}".format(cov_type)
                for i in range(len(p)):
                    fname_out += "_{}-p{}".format(int(round(100*p[i])),i)
                if dim_reduction_method is None:
                    fname_out += "_{}".format(cluster_method)
                else:
                    fname_out += "_{}_{}".format(dim_reduction_method, cluster_method)
                fname_out_acc = fname_out + "_accuracy_plot.png"
                fpath_out_acc = os.path.join(CPLOT_DIR, fname_out_acc)
                fname_out_coef = fname_out + "_silhouette_plot.png"
                fpath_out_coef = os.path.join(CPLOT_DIR, fname_out_coef)
                fname_out_rand = fname_out + "_Rand_plot.png"
                fpath_out_rand = os.path.join(CPLOT_DIR, fname_out_rand)
                
                # Only continue if the file needs to be created or overwritten.
                if os.path.isfile(fpath_out_acc) and not OVERWRITE_CLUSTER_OVERVIEW_PLOTS:
                    print("Plots {} ({}/{}) already created".format( \
                        fname_out, current_iteration, total_iterations))
                    continue
    
                print("Creating plots {} ({}/{})".format( \
                    fname_out, current_iteration, total_iterations))
    
                # Create an empty d-D matrix.
                dims = (len(N_DIFFERENT_FEATURES), len(M_DIFFERENCE), \
                    len(CLUSTER_K[cluster_method]))
                p_overlap = numpy.zeros(dims, dtype=float) * numpy.NaN
                s_avg = numpy.zeros(dims, dtype=float) * numpy.NaN
                r_scr = numpy.zeros(dims, dtype=float) * numpy.NaN
    
                # Loop through all datasets.
                for ni, n_diff in enumerate(N_DIFFERENT_FEATURES):
                    for di, m_diff in enumerate(M_DIFFERENCE):
        
                        # Construct the file name.
                        fname = "{}_{}-features_{}-diff".format(cov_type, \
                            n_diff, str(round(m_diff, ndigits=2)).replace(".",""))
                        for i in range(len(p)):
                            fname += "_{}-p{}".format(int(round(100*p[i])),i)
                        if dim_reduction_method is None:
                            fname += "_{}".format(cluster_method)
                        else:
                            fname += "_{}_{}".format(dim_reduction_method, \
                                cluster_method)
                        
                        # Construct the data file path.
                        fpath_in = os.path.join(CLU_DIR, "{}.csv".format(fname))
    
                        # Load data.
                        raw = numpy.loadtxt(fpath_in, dtype=float, \
                            delimiter=",", skiprows=1, unpack=False)
                        y_truth = raw[:,0].astype(int)
                        y = numpy.zeros((raw.shape[0],len(CLUSTER_K[cluster_method])), \
                            dtype=int) * numpy.NaN
                        s = numpy.zeros((raw.shape[0],len(CLUSTER_K[cluster_method])), \
                            dtype=float) * numpy.NaN
                        for i, k in enumerate(CLUSTER_K[cluster_method]):
                            y[:,i] = raw[:,1+2*i]
                            s[:,i] = raw[:,2+2*i]
                        
                        # Compute overlap and average silhouette scores.
                        for i, k in enumerate(CLUSTER_K[cluster_method]):

                            # Compute the average silhouette score, but only
                            # for assigned datapoints. Unassigned observations
                            # are labelled -1.
                            if numpy.nansum((y[:,i] >= 0).astype(int)) == 0:
                                s_avg[ni,di,i] = 0.0
                            else:
                                s_avg[ni,di,i] = numpy.nanmean(s[:,i][y[:,i] >= 0])
                            
                            # Compute the adjusted Rand index.
                            r_scr[ni,di,i] = adjusted_rand_score(y_truth, y[:,i])

                            # Compute overlap. Note that clusters might 
                            # overlap, but be labelled with a different value.
                            # Values are arbitrary, so this should be 
                            # corrected.
                            # Run through all possible permutations of labels.
                            perms = list(itertools.permutations(range(k)))
                            overlap = numpy.zeros(y.shape[0], dtype=bool)
                            max_overlap = 0
                            for perm in perms:
                                # Create permutated array.
                                y_perm = numpy.copy(y[:,i])
                                for j in range(k):
                                    y_perm[y[:,i]==j] = perm[j]
                                # Compute the overlap for this permutation.
                                o = y_truth == y_perm
                                # Save the overlap if this is the current best.
                                if numpy.nansum(o.astype(int)) > max_overlap:
                                    overlap = copy.deepcopy(o)
                                    max_overlap = numpy.nansum(overlap.astype(int))
                            # Compute accuracy.
                            p_overlap[ni,di,i] = float(numpy.nansum(overlap.astype(int))) \
                                / float(overlap.shape[0])
                
                # Plot the matrices for the real number of clusters.
                for i, var in enumerate(["Classification accuracy", "Adjusted Rand index", "Silhouette coefficient"]):
                    # Choose the index for the correct number of clusters.
                    if len(p) in CLUSTER_K[cluster_method]:
                        ki = CLUSTER_K[cluster_method].index(len(p))
                    else:
                        ki = 0
                    # Choose the right variable and output path.
                    if var == "Classification accuracy":
                        m = p_overlap[:,:,ki]
                        if cluster_method == "HDBSCAN":
                            vmin = 0.0
                        else:
                            vmin = numpy.max(p)
                        fpath_out = fpath_out_acc
                    elif var == "Adjusted Rand index":
                        m = r_scr[:,:,ki]
                        vmin = 0.0
                        vmax = 1.0
                        fpath_out = fpath_out_rand
                    elif var == "Silhouette coefficient":
                        m = s_avg[:,:,ki]
                        vmin = 0.0
                        fpath_out = fpath_out_coef

                    # Create a new figure.
                    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0, 6.0), \
                        dpi=300.0)
                    # Plot the heatmap.
                    im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=1)
                    # Add colourbar.
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
                    cbar.set_label(var, fontsize=16)
                    # Add axis ticks.
                    ax.set_xticks(numpy.arange(0, m.shape[1]+1))
                    ax.set_xticklabels(map(str, M_DIFFERENCE), fontsize=12)
                    ax.set_yticks(numpy.arange(0, m.shape[0]+1))
                    ax.set_yticklabels(map(str, numpy.round( \
                        numpy.array(N_DIFFERENT_FEATURES, dtype=float) \
                        / numpy.max(N_DIFFERENT_FEATURES), \
                        decimals=2)), fontsize=12)
                    # Set axis limits.
                    ax.set_xlim(-0.5, m.shape[1]-0.5)
                    ax.set_ylim(-0.5, m.shape[0]-0.5)
                    # Add axis labels.
                    ax.set_xlabel(r"Effect size $\delta$", fontsize=16)
                    ax.set_ylabel(r"Proportion of different features", fontsize=16)
                    # Annotate individual values.
                    for col in range(m.shape[1]):
                        for row in range(m.shape[0]):
                            ax.annotate(str(numpy.round(m[row,col], decimals=2)).ljust(4, "0"), \
                                (col-0.15,row-0.05), color="#FFFFFF", fontsize=12)
                    # Save and close the figure.
                    fig.savefig(fpath_out)
                    pyplot.close(fig)

# # # # #
# POWER AND ACCURACY

# Run through all cluster methods.
for ci, cluster_method in enumerate(SIM_CLUSTER_METHODS):
    
    # Construct the figure path.
    outpath = os.path.join(SIM_DIR, "power_accuracy_{}.png".format(cluster_method))
    
    # Check if the current figure needs to be overwritten.
    if os.path.isfile(outpath) and not OVERWRITE_POWER:
        print("Power and accuracy outcomes for {} ({}/{}) already exist".format( \
            cluster_method, ci+1, len(SIM_CLUSTER_METHODS)))
        continue
    
    print("Simulating data and computing power and accuracy outcomes for {} ({}/{})".format( \
        cluster_method, ci+1, len(SIM_CLUSTER_METHODS)))

    # Create a new figure.
    fig, ax = pyplot.subplots(nrows=4, ncols=len(SIM_P), \
        figsize=(4*8.0, len(SIM_P)*6.0), dpi=300)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, \
        wspace=0.2, hspace=0.2)

    # Create a normed colourmap.
    cmap = matplotlib.cm.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=max(SIM_D))
    col = cmap(norm(0.5))
    
    # Add colour maps to the right-most plots.
    for row in range(ax.shape[0]):
        divider = make_axes_locatable(ax[row,-1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        #cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, \
            orientation="vertical")
        cbar.set_ticks([0.0, max(SIM_D)])
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(r"Cluster separation $\Delta$", fontsize=24)

    # Create a 2x2 covariance matrix.
    cov = [[1.0, 0.0], [0.0, 1.0]]
    
    # Loop through all datasets that need to be simulated.
    for pi, p in enumerate(SIM_P):
        
        print("\tRunning {} simulations for {} effect sizes ({}/{})".format( \
              SIM_N_ITERATIONS_PER_CELL, len(SIM_D), pi+1, len(SIM_P)))
        t0 = time.time()
    
        # Determine k.
        k = len(p)

        # Open a new text file.
        pwr_f = open(os.path.join(SIM_DIR, "power_{}_pop-{}.csv".format( \
            cluster_method, pi)), "w")
        header = ["DELTA"]
        header.extend(SIM_N)
        pwr_f.write(",".join(map(str, header)))
        
        # Loop through all simulated effect sizes.
        for di, d in enumerate(SIM_D):
    
            # Create vectors to hold outcomes in.
            dim = (len(SIM_N), SIM_N_ITERATIONS_PER_CELL)
            sil = numpy.zeros(dim, dtype=float) * numpy.NaN
            pwr = numpy.zeros(dim, dtype=float) * numpy.NaN
            p_k = numpy.zeros(dim, dtype=float) * numpy.NaN
            ran = numpy.zeros(dim, dtype=float) * numpy.NaN
            acc = numpy.zeros(dim, dtype=float) * numpy.NaN
            
            # Loop through all simulations.
            for ni, n in enumerate(SIM_N):
                for ii in range(SIM_N_ITERATIONS_PER_CELL):
                    
                    # Determine the size of each cluster.
                    sample_n = []
                    for p_ in p:
                        sample_n.append(int(numpy.round(n*p_, decimals=0)))
                    # Recompute the actual sample size.
                    n = sum(sample_n)
                    # Create the ground truth cluster membership.
                    y_truth = numpy.ones(n, dtype=int) * -1
                    si = 0
                    for i in range(len(sample_n)):
                        y_truth[si:si+sample_n[i]] = i
                        si += sample_n[i]
    
                    # Create means in two dimensions.
                    m = numpy.zeros((2, k), dtype=float)
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
                            cov, size=sample_n[i])
                    
                    # Create empty matrices to hold the outcomes in.
                    y = numpy.zeros((X.shape[0],len(CLUSTER_K[cluster_method])), \
                        dtype=int) * numpy.NaN
                    s = numpy.zeros((X.shape[0],len(CLUSTER_K[cluster_method])), \
                        dtype=float) * numpy.NaN
                    if cluster_method == "CMEANS":
                        s_fuzz = numpy.zeros(len(CLUSTER_K[cluster_method]), \
                            dtype=float) * numpy.NaN

                    # Run cluster analyses.
                    for ki, k_guess in enumerate(CLUSTER_K[cluster_method]):
                        # Cluster the data.
                        if cluster_method == "CMEANS":
                            _, u, _, _, _, _, _ = skfuzzy.cluster.cmeans( \
                                numpy.transpose(X), k_guess, 2, error=0.005, \
                                maxiter=1000, init=None)
                            y[:,ki] = numpy.argmax(u, axis=0)
                            s_fuzz[ki], _ = fuzzy_silhouette_coefficient( \
                                X, y[:,ki], numpy.transpose(u), alpha=1.0)
                        else:
                            y[:,ki] = clustering(X, mode=cluster_method, \
                                n_clusters=k_guess)
                        # Compute a silhouette coefficient for each sample,
                        # but only if more than one cluster was detected.
                        if len(numpy.unique(y[:,ki])) == 1:
                            s[:,ki] = 0.0
                        else:
                            s[:,ki] = silhouette_samples(X, y[:,ki], \
                                metric='euclidean')
                    
                    # Compute the silhouette score.
                    if cluster_method == "HDBSCAN":
                        ki = 0
                    else:
                        ki = CLUSTER_K[cluster_method].index(k)
                    if numpy.nansum((y[:,ki] >= 0).astype(int)) == 0:
                        sil[ni,ii] = 0.0
                    else:
                        if cluster_method == "CMEANS":
                            sil[ni,ii] = s_fuzz[ki]
                        else:
                            sil[ni,ii] = numpy.nanmean(s[:,ki][y[:,ki]>=0])
                    
                    # Compute power.
                    pwr[ni,ii] = int(sil[ni,ii]>0.5)
                    
                    # Compute the silhouette score for all k guesses.
                    if cluster_method == "HDBSCAN":
                        k_detected = numpy.max(y[:,ki])+1
                    elif cluster_method == "CMEANS":
                        k_detected = CLUSTER_K[cluster_method][numpy.argmax(s_fuzz)]
                    else:
                        k_detected = CLUSTER_K[cluster_method][ \
                            numpy.argmax(numpy.nanmean(s, axis=0))]
                    p_k[ni,ii] = int(k==k_detected)
    
                    # Compute the adjusted Rand index.
                    ran[ni,ii] = adjusted_rand_score(y_truth, y[:,ki])
    
                    # Compute the classification accuracy.
                    # Compute overlap. Note that clusters might 
                    # overlap, but be labelled with a different value.
                    # Values are arbitrary, so this should be 
                    # corrected.
                    # Run through all possible permutations of labels.
                    perms = list(itertools.permutations(range(k)))
                    overlap = numpy.zeros(y.shape[0], dtype=bool)
                    max_overlap = 0
                    for perm in perms:
                        # Create permutated array.
                        y_perm = numpy.copy(y[:,ki])
                        for j in range(k):
                            y_perm[y[:,ki]==j] = perm[j]
                        # Compute the overlap for this permutation.
                        o = y_truth == y_perm
                        # Save the overlap if this is the current best.
                        if numpy.nansum(o.astype(int)) > max_overlap:
                            overlap = copy.deepcopy(o)
                            max_overlap = numpy.nansum(overlap.astype(int))
                    # Compute accuracy.
                    acc[ni,ii] = float(numpy.nansum(overlap.astype(int))) \
                        / float(overlap.shape[0])
    
            # Write power to text file.
            line = [d]
            line.extend(numpy.nanmean(pwr, axis=1))
            pwr_f.write("\n" + ",".join(map(str, line)))

            # Loop through all measured variables.
            for vi, var in enumerate([sil, pwr, p_k, acc]):
                # Compute the average.
                m = numpy.nanmean(var, axis=1)
                # Compute the standard deviation.
                sd = numpy.nanstd(var, axis=1)
                # Estimate the 95% confidence interval.
                sem = sd / numpy.sqrt(var.shape[1])
                ci = 1.96 * sem
                ci_low = m - ci
                ci_high = m + ci
                # Compute the spread of the data.
                i_5 = int(numpy.round(0.05*var.shape[1]))
                i_95 = min(var.shape[1]-1, int(numpy.round(0.95*var.shape[1])))
                sp_low = numpy.zeros(m.shape, dtype=float)
                sp_high = numpy.zeros(m.shape, dtype=float)
                for ni in range(var.shape[0]):
                    ordered_var = numpy.sort(numpy.copy(var[ni,:]))
                    sp_low[ni] = ordered_var[i_5]
                    sp_high[ni] = ordered_var[i_95]
                # Determine what value to plot.
                low_bound = ci_low
                high_bound = ci_high
                
                # Plot the chance level.
                if di == 0:
                    # Plot the silhouette thresholds.
                    if vi == 0:
                        ax[vi,pi].plot(numpy.array([0, max(SIM_N)+0.1*max(SIM_N)]), \
                            numpy.ones(2)*0.5, '--', color="#666666", \
                            lw=3, alpha=0.5, label="Reliable clustering")
                        ax[vi,pi].plot(numpy.array([0, max(SIM_N)+0.1*max(SIM_N)]), \
                            numpy.ones(2)*0.7, '--', color="#000000", \
                            lw=3, alpha=0.5, label="Strong clustering")
                    # Plot the chance level.
                    elif vi == 3:
                        ax[vi,pi].plot(numpy.array([0, max(SIM_N)+0.1*max(SIM_N)]), \
                            numpy.ones(2)*numpy.max(p), '--', color="#000000", \
                            lw=3, alpha=0.5, label="Chance")
                
                # Plot the average.
                x = numpy.array(SIM_N)
                ax[vi,pi].plot(x, m, '-', lw=3, alpha=1.0, color=cmap(norm(d)))
                # Shade the confidence interval.
                ax[vi,pi].fill_between(x, low_bound, high_bound, alpha=0.5, color=cmap(norm(d)))
                
                # Set the axis limits.
                ax[vi,pi].set_ylim(-0.05, 1.05)
                ax[vi,pi].set_xlim(0, max(SIM_N)+0.05*max(SIM_N))
                ax[vi,pi].tick_params(labelsize=15)
                
                # Set the y-axis label on the left-most plots.
                if pi == 0:
                    if vi == 0:
                        ylbl = "Silhouette score"
                    elif vi == 1:
                        ylbl = "Statistical power"
                    elif vi == 2:
                        ylbl = "Cluster number accuracy"
                    elif vi == 3:
                        ylbl = "Classification accuracy"
                    ax[vi,pi].set_ylabel(ylbl, fontsize=28)
                
                # Set the x-axis labels on the bottom plot.
                if vi == 3:
                    ax[vi,pi].set_xlabel("Sample size (N)", fontsize=24)
                
                # Set the number of clusters as the title of the top axes.
                if vi == 0:
                    if k == 2:
                        tit = "Two clusters"
                    elif k == 3:
                        tit = "Three clusters"
                    elif k == 4:
                        tit = "Four clusters"
                    else:
                        tit = "{} clusters".format(k)
                    tit += "\n{}".format(tuple([int(round(perc*100, ndigits=0)) for perc in p]))
                    ax[vi,pi].set_title(tit, fontsize=32)
        
        # Close power text file.
        pwr_f.close()

        print("\t\tCompleted in {} seconds".format( \
            round(time.time()-t0), ndigits=2))
    
    # Save and close the figure.
    fig.savefig(outpath)
    pyplot.close(fig)
    
