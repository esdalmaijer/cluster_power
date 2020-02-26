# If you're looking for someone to blame for this, email:
# Edwin Dalmaijer, edwin.dalmaijer@mrc-cbu.cam.ac.uk

import os
import copy

import numpy
from scipy.stats import f_oneway, kruskal, pearsonr, ttest_ind
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sklearn
from sklearn import cluster, decomposition, preprocessing
from sklearn.impute import SimpleImputer
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, TSNE
from sklearn.metrics import calinski_harabaz_score, silhouette_samples, silhouette_score
from sklearn.neighbors import kneighbors_graph, KNeighborsRegressor
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

import hdbscan
import umap

# Colours for plotting from the Tango colour scheme. They start repeating
# after 20
PLOTCOLS = {\
    -1:'#000000', # black

    0:'#4e9a06', # green
    1:'#204a87', # blue
    2:'#5c3566', # purple
    3:'#c4a000', # yellow
    4:'#8f5902', # chocolate
    5:'#ce5c00', # orange
    6:'#a40000', # red

    7:'#73d216', # green
    8:'#3465a4', # blue
    9:'#75507b', # purple
    10:'#edd400', # yellow
    11:'#c17d11', # chocolate
    12:'#f57900', # orange
    13:'#cc0000', # red

    14:'#8ae234', # green
    15:'#729fcf', # blue
    16:'#ad7fa8', # purple
    17:'#fce94f', # yellow
    18:'#e9b96e', # chocolate
    19:'#fcaf3e', # orange
    20:'#ef2929', # red
    }
for i in range(21, 201):
    PLOTCOLS[i] = copy.deepcopy(PLOTCOLS[i % 21])

# # # # #
# PLOTTING FUNCTIONS

def correlation_matrix(X, X_original, varnames=None, sig=0.05, vlim=1.0, \
        savepath=None, ax=None):
    
    """Computes a correlation matrix, and plots it in the passed axis (or a
    new figure if so requested).
    
    Arguments

    X           -   NumPy array with shape (N,M), where N is the number of
                    observations, and M the number of features (this should be
                    the data in reduced space).
        
    X_original  -   NumPy array with shape (N,M), where N is the number of
                    observations, and M the number of features (this should be
                    the original non-reduced data if you're using one).
    
    Keyword Arguments

    varnames    -   The names of variables that will be plotted along the y
                    axis of the correlation matrix, or None to not plot them.
                    Default = None

    sig         -   A float indicating the alpha level of the Pearson
                    correlations. Correlations with p values over this limit
                    will not be plotted. To plot all values, pass 1.0.
                    Default = 0.05

    vlim        -   A float to indicate the maximum and minimum values on the
                    colourmap used for the correlation matrix. Default = 1.0

    savepath    -   A string that indicates the path where the plot should be
                    saved, or None to not save the plot.

    ax          -   A matplotlib.AxesSubplot instance that will be used to
                    plot the correlation matrix in. If None is passed, a new
                    figure will be created. Default = None
    
    Returns
    
    R, p        -   NumPy arrays with shape (N,M), where N is the number of
                    components in X and M is the number of features in
                    X_original. R contains Pearson correlation values, and p
                    contains the associated p values.
    """
    
    # We can only work with matrices of the same number of observations.
    if X.shape[0] != X_original.shape[0]:
        raise Exception("Mismatch in number of observations between X (%d) and X_original (%d)" \
            % (X.shape[0], X_original.shape[0]))
    
    # Get the number of features in the original dataset and the number of
    # components in the reduced space.
    n_features = X_original.shape[1]
    n_components = X.shape[1]

    # Calculate a correlation matrix for the new components and the old
    # features. (Somewhat similar to the component weights of the other
    # decompositioning methods.)
    corM = numpy.zeros((n_components, n_features), dtype=float)
    pM = numpy.zeros((n_components, n_features), dtype=float)
    for i in range(n_components):
        for j in range(n_features):
            corM[i,j], pM[i,j] = pearsonr(X[:,i], X_original[:,j])
    
    # Rotate for visualisation purposes.
    rotated_R = numpy.transpose(corM)
    rotated_p = numpy.transpose(pM)
    
    # Set non-significant values to 0.
    rotated_R[rotated_p > sig] = 0.0

    # Create a new figure if no axis was specified.
    if ax is None:
        closefig = True
        fig, ax = pyplot.subplots(figsize=(8.0,6.0), dpi=300.0)
    else:
        closefig = False
    # Plot the correlation matrix.
    if vlim is None:
        vmax = max([numpy.abs(rotated_R.min()), rotated_R.max()])
        vmin = -1*vmax
    else:
        vmin = -vlim
        vmax = vlim
    cax = ax.imshow(rotated_R, vmin=vmin, vmax=vmax, cmap='coolwarm', \
        aspect='equal', interpolation='none')
    cbar = fig.colorbar(cax, ticks=[vmin, 0, vmax], orientation='vertical')
    cbar.ax.set_ylabel("Pearson correlation")
    ax.set_xticks(range(n_components))
    ax.set_xticklabels(map(str, range(1, n_components+1)))
    ax.set_yticks(range(n_features))
    if varnames is not None:
        ax.set_yticklabels(varnames)
    if closefig:
        if savepath is not None:
            fig.savefig(savepath)
        pyplot.close(fig)
    
    return corM, pM


# Scatterplot of samples coloured by an additional feature.
def plot_samples(X, xi, yi, zi, ci, xvar, yvar, zvar, cvar, savepath=None, ax=None):

    # Create a new figure.
    if ax is None:
        closefig = True
        fig =  pyplot.figure(figsize=(8.0,6.0), dpi=300.0)
        if zi is None:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
    else:
        closefig = False

    # Choose the right colourmap and min/max values.
    if ci is not None:
        if numpy.nanmin(X[:,ci]) < 0 and numpy.nanmax(X[:,ci]) > 0:
            cmapname = "coolwarm"
            absmax = max([numpy.abs(numpy.nanmin(X[:,ci])), numpy.nanmax(X[:,ci])])
            vmin = -absmax
            vmax = absmax
        else:
            cmapname = "viridis"
            vmin = numpy.nanmin(X[:,ci])
            vmax = numpy.nanmin(X[:,ci])
        # Create new colourmap and normalisation instances.
        cmap = matplotlib.cm.get_cmap(cmapname)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Two-dimensional plot.
    if zi is None and ci is None:
        p = ax.scatter( \
            X[:, xi], \
            X[:, yi], \
            color="#FF69B4", \
            alpha=0.5, \
            )
    # Two-dimensional plot with colour specifying a third dimension.
    elif zi is None:
        cmap = matplotlib.cm.get_cmap("coolwarm")
        absmax = max([numpy.nanmin(X[:,ci]), numpy.nanmax(X[:,ci])])
        vmin = -absmax
        vmax = absmax
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        p = ax.scatter( \
            X[:, xi], \
            X[:, yi], \
            c=X[:,ci], \
            cmap=cmap, \
            norm=norm, \
            alpha=0.5, \
            )
    # Three-dimensional plot.
    elif ci is None:
        p = ax.scatter( \
            X[:, xi], \
            X[:, yi], \
            X[:, zi], \
            color="#FF69B4", \
            alpha=0.5, \
            )
    # Three-dimensional plot with colour specifying a fourth dimension.
    else:
        p = ax.scatter( \
            X[:, xi], \
            X[:, yi], \
            X[:, zi], \
            c=X[:,ci], \
            cmap=cmap, \
            norm=norm, \
            alpha=0.5, \
            )
    # Add a colourbar to the plot.
    if ci is not None:
        divider = make_axes_locatable(ax)
        if zi is None:
            bax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
                ticks=range(int(numpy.floor(vmin)), int(numpy.ceil(vmax))+1, 1), orientation='vertical')
        else:
            cbar = fig.colorbar(p)
        cbar.set_label(cvar)
    # Add axis labels.
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    if zi is not None:
        ax.set_zlabel(zvar)
    if closefig:
        if savepath is not None:
            fig.savefig(savepath)
        pyplot.close(fig)


# Scatterplot of samples, coloured after their cluster membership.
def plot_clusters(X, y, xi, yi, zi, xvar, yvar, zvar, ax=None, savepath=None, 
    cluster_centres=None):
    
    # Get all unique cluster labels.
    lbls = numpy.unique(y)
    
    # Compute cluster centres.
    if cluster_centres is None:
        # Create a new matrix for the clusters. Confusingly, if the label '-1'
        # is used, it will appear at the end of the matrix.
        cluster_centres = numpy.zeros((len(lbls),3), dtype=float)
        for i in [xi, yi, zi]:
            for lbl in lbls:
                if i is not None:
                    cluster_centres[lbl,i] = numpy.nanmean(X[y==lbl,i])

    # Create a new figure if required.
    if ax is None:
        closefig = True
        fig =  pyplot.figure(figsize=(8.0,6.0), dpi=300.0)
        if zi is None:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
    else:
        closefig = False
    
    # Go through all cluster labels to plot each individual one.
    for lbl in lbls:
        # Count the number of samples in this cluster.
        cluster_size = numpy.sum((y==lbl).astype(int))
        # Two-dimensional plot of all samples.
        if zi is None:
            ax.scatter( \
                X[y==lbl, xi], \
                X[y==lbl, yi], \
                color=PLOTCOLS[lbl], \
                label="N=%d" % (cluster_size), \
                alpha=0.5, \
            )
        # Three-dimensional plot of all samples.
        else:
            ax.scatter( \
                X[y==lbl, xi], \
                X[y==lbl, yi], \
                X[y==lbl, zi], \
                color=PLOTCOLS[lbl], \
                label="N=%d" % (cluster_size), \
                alpha=0.5, \
            )
        # Plot the cluster centres.
        if not (cluster_centres is None):
            # Two-dimensional plot.
            if zi is None:
                ax.scatter( \
                    cluster_centres[lbl, xi], \
                    cluster_centres[lbl, yi], \
                    s=50, \
                    color=PLOTCOLS[lbl], \
                    edgecolor='#000000'
                    )
            # Three-dimensional plot.
            else:
                ax.scatter( \
                    cluster_centres[lbl, xi], \
                    cluster_centres[lbl, yi], \
                    cluster_centres[lbl, zi], \
                    s=50, \
                    color=PLOTCOLS[lbl], \
                    edgecolor='#000000'
                    )
    # Add a legend and axis labels.
    ax.legend(loc='upper right')
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    if zi is not None:
        ax.set_zlabel(zvar)
    if closefig:
        if savepath is not None:
            fig.savefig(savepath)
        pyplot.close(fig)


# Silhouette plot.
def plot_silhouette(X, y, ax=None, savepath=None):

    # Find all unique values in the cluster labels.
    lbls = numpy.unique(y)
    
    # Compute all silhouette values.
    silhouette_vals = silhouette_samples(X, y, metric='euclidean')
    # Compute the cluster coefficient (=average of all silhouette values).
    silhouette_avg = numpy.mean(silhouette_vals)
    # Compute the cluster coefficient for all non-negative labels. In some
    # algorithms (e.g. DBSCAN and HDBSCAN) not all samples are labelled, and
    # unlabelled samples are assigned a value of -1. These should not be
    # included in the clustering score.
    if -1 in lbls:
        #silhouette_avg = silhouette_score(X[y>=0,:], y[y>=0], metric='euclidean')
        silhouette_avg = numpy.mean(silhouette_vals[y>=0])

    # Create a new plot.
    if ax is None:
        closefig = True
        fig, ax = pyplot.subplots(nrows=1, ncols=1)
    else:
        closefig = False
    y_ax_upper = 0
    y_ax_lower = 0
    y_ticks = []
    # Go through all unique labels.
    for lbl in lbls:
        # Find all samples in this cluster.
        cluster_silhouette_vals = silhouette_vals[y == lbl]
        # Sort the cluster values by their silhouette score.
        cluster_silhouette_vals.sort()
        y_ax_upper += len(cluster_silhouette_vals)
        # Create a horizontal bar graph for all samples in this cluster.
        ax.barh(range(y_ax_lower, y_ax_upper), cluster_silhouette_vals, \
            height=1.0, edgecolor='none', color=PLOTCOLS[lbl])
        # Update the y ticks.
        y_ticks.append((y_ax_lower + y_ax_upper) / 2.0)
        y_ax_lower += len(cluster_silhouette_vals)
    # Draw the cluster coefficient into the graph.
    ax.axvline(silhouette_avg, color='#ff69b4', linestyle='--')
    # Finih the plot.
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(lbls)
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_xlim([-0.2, 1.0])
    # Save the plot.
    if closefig:
        if savepath is not None:
            fig.savefig(savepath)
        pyplot.close(fig)


# Feature average plot.
def plot_averages(X, y, varnames=None, ax=None, savepath=None):
    
    # Find all unique cluster labels.
    lbls = numpy.unique(y)
    
    # Find the number of observations and features in the passed data.
    n_samples, n_features = X.shape
    
    # Create a new plot.
    if ax is None:
        closefig = True
        fig, ax = pyplot.subplots(nrows=1, ncols=1)
    else:
        closefig = False
    # Go through all labels.
    for lbl in lbls:
        # Compute the cluster size.
        cluster_size = numpy.sum((y==lbl).astype(int))
        # Compute the average for each feature for this cluster.
        m = numpy.mean(X[y==lbl, :], axis=0)
        # Compute the standard error of the mean for each feature.
        sem = numpy.std(X[y==lbl, :], axis=0) / numpy.sqrt(cluster_size)
        # Plot the average and standard error.
        x = range(1, n_features+1)
        ax.plot(x, m, 'o-', label="N=%d" % (cluster_size), color=PLOTCOLS[lbl])
        ax.errorbar(x, m, yerr=sem, ecolor=PLOTCOLS[lbl], fmt='none')
    # Finish the plot.
    ax.legend(loc="upper left")
    ax.set_ylabel("Score")
    ax.set_xlabel("Feature")
    ax.set_xlim([0, n_features+1])
    ax.set_xticks(range(1, n_features+1))
    if varnames is not None:
        ax.set_xticklabels(varnames, rotation='vertical')
    # Save the figure.
    if closefig:
        if savepath is not None:
            fig.savefig(savepath)
        pyplot.close(fig)

# Plotting a comparison between several clusters on a particular variable.
def cluster_comparison(var, y, varname, ax=None, savepath=None, \
    stats_annotate=True):
    
    # Count the number of clusters.
    clusters = numpy.unique(y)
    clusters = numpy.sort(clusters)
    n_clusters = len(clusters)

    # Create a new plot.
    if ax  is None:
        closefig = True
        fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0,6.0), dpi=300.0)
    # Set the x-locations of plot elements.
    xstep = 0.5
    xticks = numpy.arange(0.5, n_clusters*xstep+xstep, xstep)
    xloc = xticks + 0.1
    boxloc = xticks - 0.1
    # Plot the groups.
    desc = {}
    sample_list = []
    xticklabels = []
    for i, lbl in enumerate(clusters):
        
        # Find all values associated with this cluster.
        _y = var[y==lbl]
        xticklabels.append("N=%d" % (numpy.sum((y==lbl).astype(int))))
        # Create a vector for the horizontal positions of plotted samples.
        notnan = numpy.isnan(_y)==False
        _x = numpy.ones(numpy.sum(notnan.astype(int))) * xloc[i]
        
        # Compute the descriptives, but ignore the non-clustered samples
        # (these are labelled -1).
        if lbl >= 0:
            desc[lbl] = {}
            sample_list.append(_y[notnan])
            desc[lbl]['n'] = numpy.sum(notnan.astype(int))
            desc[lbl]['m'] = numpy.mean(_y[notnan])
            desc[lbl]['sd'] = numpy.std(_y[notnan])
            desc[lbl]['sem'] = desc[lbl]['sd'] / numpy.sqrt(desc[lbl]['n'] - 1)

        # Draw the individual points.
        ax.plot(_x, _y[notnan], 'o', label=lbl, color=PLOTCOLS[lbl], alpha=0.3)
        # Draw a violin plot.
        vp = ax.violinplot(_y[notnan], vert=True, \
            showmeans=False, showmedians=False, showextrema=False, \
            positions=[boxloc[i]], widths=0.15)
        vp['bodies'][0].set_facecolor(PLOTCOLS[lbl])
        vp['bodies'][0].set_edgecolor(PLOTCOLS[lbl])
#        vp['cbars'].set_color(PLOTCOLS[lbl])
#        vp['cbars'].set_linewidth(3)
#        vp['cmins'].set_color(PLOTCOLS[lbl])
#        vp['cmaxes'].set_color(PLOTCOLS[lbl])
#        vp['cmedians'].set_color(PLOTCOLS[lbl])
#        vp['cmedians'].set_linewidth(3)
        # Draw a boxplot within the violin plot.
        bp = ax.boxplot(_y[notnan], notch=False, sym='ko', vert=True, \
            positions=[boxloc[i]], widths=0.03, \
            patch_artist=True, \
            boxprops={'edgecolor':'black', 'facecolor':'white', 'linewidth':3}, \
            capprops={'color':'black', 'linewidth':3}, \
            whiskerprops={'color':'black', 'linewidth':3}, \
            flierprops={'color':'black', 'linewidth':3}, \
            medianprops={'color':PLOTCOLS[lbl], 'linewidth':3}, \
            )

    # Perform a one-way ANOVA to see whether there are differences in the
    # whole group.
    result = f_oneway(*sample_list)
    try:
        f, fp = result
    except:
        f = result.statistic
        fp = result.pvalue
    # Perform a Kurskal-Wallis H-test, which is a non-parametric alternative
    # to the one-way ANOVA.
    result = kruskal(*sample_list)
    try:
        h, hp = result
    except:
        h = result.statistic
        hp = result.pvalue
    
    # Perform pairwise tests to see which clusters differ from each other.
    t = {}
    tp = {}
    # Loop through all possible cluster pairs, but ignore the non-clustered
    # data (with label -1). First, loop through all clusters except from the
    # last cluster.
    for i in range(len(clusters)-1):
        # Grab the label.
        lbl_a = clusters[i]
        # Ignore label -1, as it is associated with non-clustered data.
        if lbl_a == -1:
            continue
        # Create a new empty dict if necessary.
        if lbl_a not in t.keys():
            t[lbl_a] = {}
            tp[lbl_a] = {}
        # Find all values associated with cluster a.
        a = var[y==lbl_a]
        # Loop through all possible other clusters, which are the next cluster
        # until the last cluster (cluster a can never be the last cluster).
        for j in range(i+1, len(clusters)):
            # Grab the label.
            lbl_b = clusters[j]
            # Find all values associated with cluster b.
            b = var[y==lbl_b]
            # Ignore NaNs.
            anotnan = numpy.isnan(a)==False
            bnotnan = numpy.isnan(b)==False
            # Test whether the difference between clusters is statistically
            # significant, using an independent samples t test.
            result = ttest_ind(a[anotnan], b[bnotnan])
            try:
                t[lbl_a][lbl_b], tp[lbl_a][lbl_b] = result
            except:
                t[lbl_a][lbl_b] = result.statistic
                tp[lbl_a][lbl_b] = result.pvalue
    
    # Annotate the statistics.
    if stats_annotate:
        # Set line heights for the t-tests, based on proximity of the compared
        # clusters. The step determines the vertical spacing between
        # annotations and associated lines.
        line_y = numpy.nanmax(var) * 1.05
        annotate_y = numpy.nanmax(var) * 1.07
        lines_step = numpy.nanmax(var) * 0.1
        max_y = annotate_y + len(clusters) * lines_step
        # Annotate the ANOVA.
        if fp < 0.001:
            param_str = r"F=%.2f, p<0.001" % (f)
        else:
            param_str = r"F=%.2f, p=%.3f" % (f, fp)
        if hp < 0.001:
            nonparam_str = r"H=%.2f, p<0.001" % (h)
        else:
            nonparam_str = r"H=%.2f, p=%.3f" % (h, hp)
        xpos = xticks[0] - xstep*0.9
        ypos = [max_y, max_y+lines_step]
        ax.annotate(r"$%s$" % (param_str), (xpos,ypos[0]), fontsize=14)
        ax.annotate(r"$%s$" % (nonparam_str), (xpos,ypos[1]), fontsize=14)
        # Annotate the t-tests.
        left_clusters = t.keys()
        left_clusters.sort()
        for lbl_a in left_clusters:
            right_clusters = t[lbl_a].keys()
            right_clusters.sort()
            for lbl_b in right_clusters:
                # Get the index numbers of both clusters.
                i = list(clusters).index(lbl_a)
                j = list(clusters).index(lbl_b)
                # Find the associated x values.
                sx = xticks[i] + xstep*0.1
                ex = xticks[j] -  + xstep*0.1
                # Determine the lines y-value based on the distance between
                # the clusters.
                ly = line_y + lines_step * (j-i)
                ay = annotate_y + lines_step * (j-i)
                # Draw a line between the clusters.
                ax.plot([sx,ex], [ly,ly], '-', color="#000000", lw=3)
                ax.annotate(r"$t=%.2f, p=%.3f$" % (t[lbl_a][lbl_b], tp[lbl_a][lbl_b]), \
                    (sx,ay), fontsize=10)

    # Finish the plot.
    if stats_annotate:
        ax.set_ylim(top=max_y+2*lines_step)
    ax.set_xlim([numpy.min(xticks)-xstep, numpy.max(xticks)+xstep])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=14)
    ax.set_ylabel(varname, fontsize=14)
    # Save the figure.
    if closefig:
        if savepath is not None:
            fig.savefig(savepath)
        pyplot.close(fig)


# # # # #
# DATA FUNCTIONS

def preprocess(X, mode='standardise', impute=None):
    
    """Pre-processes a data matrix of shape (N,M), where M is the number of
    features and N the number of observations. Also removes NaNs from the
    matrix.
    
    Arguments

    X       -   NumPy array with shape (N,M), where N is the number of
                observations, and M the number of features.
    
    Keyword Arguments
    
    mode    -   String that defines the type of pre-processing. Choose from:
                - None
                - "min-max"
                - "standardise"
                - "normalise"
                Default = "standardise"
    
    impute  -   String that defines the type of imputation. Choose from:
                - None
                - "mean"
                - "median"
                - "most_frequent"
                Default = None
    
    Returns
    
    X, include
            -   X is a NumPy array with shape (N-n,M), where N is the number
                of observations and n is the number of observations with a NaN.
                M is the number of features. Now with scaled values.
                include is a NumPy array with shape (N,), where N is the 
                number of observations. Each value is a Boolean, True for all
                included samples, and False for non-included samples.
    """
    
    # Translate into English.
    if mode == "standardize":
        mode = "standardise"
    elif mode == "normalize":
        mode = "normalise"
    
    # Count the number of observations (subjects) and features.
    n_features = X.shape[1]
    n_subjects = X.shape[0]
    
    # Check whether we need to do imputation.
    if impute is None or impute in ["none", "None"]:
        # Check which cases we can include.
        include = (numpy.sum(numpy.isnan(X), axis=1) == 0).astype(bool)
        # Construct a new X.
        X_ = numpy.copy(X[include,:])
    elif impute in ["mean", "median", "most_frequent"]:
        # Construct a new X with imputed values.
        imputer = SimpleImputer(missing_values=numpy.NaN, strategy=impute, \
            copy=True)
        X_ = imputer.fit_transform(X)
        # To match non-imputation cases, create an include vector (which will
        # now include everything).
        include = (numpy.sum(numpy.isnan(X_), axis=1) == 0).astype(bool)
    elif impute.lower() in ["knn", "nearestneighbours", "knearestneighbours", \
        "nearestneighbours", "knearestneighbours"]:
        # Find all complete datasets.
        complete = (numpy.sum(numpy.isnan(X), axis=1) == 0).astype(bool)
        # Create a new dataset to add the imputed data to.
        X_ = numpy.zeros((n_subjects, n_features), dtype=float) * numpy.NaN
        # Copy complete datasets into X_.
        X_[complete,:] = numpy.copy(X[complete,:])
        # Loop through all participants.
        for i in range(n_subjects):
            # Skip subjects with complete datasets.
            if complete[i]:
                continue
            # Find all NaNs in this participant (these will be imputed).
            missing = numpy.isnan(X[i,:])
            # Copy all non-missing values into X_
            X_[i,missing==False] = numpy.copy(X[i,missing==False])
            # Only impute when fewer than 1/3 of the features are missing.
            if numpy.sum(missing.astype(int)) > numpy.floor(float(n_features) / 3.0):
                continue
            # Loop through all missing features.
            for j in numpy.where(missing==True)[0]:
                # Create a target matrix for the current feature, and another
                # matrix for all others.
                target = X[:,j]
                other = X[:,missing==False]
                # Train a nearest neighbour model.
                knn = KNeighborsRegressor(n_neighbors=9)
                knn.fit(other[complete,:], target[complete])
                # Impute the missing data.
                X_[i,j] = knn.predict(other[i,:].reshape((1,other.shape[1])))
        # Create an include vector to filter out the cases that could not be
        # imputed.
        include = (numpy.sum(numpy.isnan(X_), axis=1) == 0).astype(bool)
        X_ = X_[include,:]
    else:
        raise Exception("ERROR: Unrecognised imputation method '%s'" % (impute))
    
    # Scale the data.
    for i in range(n_features):
        if mode == "min-max":
            X_[:,i] = preprocessing.minmax_scale(X_[:,i])
        elif mode == "standardise":
            X_[:,i] = preprocessing.scale(X_[:,i])
        elif mode is None or mode in ["none", "None"]:
            pass
        else:
            raise Exception("Unrecognised pre-processing mode '%s'" % (mode))
    if mode == "normalise":
        X_ = preprocessing.normalize(X_)
    
    return X_, include


def dim_reduction(X, n_components=2, mode="MDS"):
    
    """Reduces the number of dimensions in which a dataset is defined.
    
    Arguments

    X       -   NumPy array with shape (N,M), where N is the number of
                observations, and M the number of features.
    
    Keyword Arguments
    
    n_components    -   Intended number of features after dimensionality
                        reduction. Default = 2
    
    mode            -   String that defines the type of dim reduction:
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
                        Default = "MDS"
    
    Returns
    
    X       -   NumPy array with shape (N-n,M), where N is the number of
                observations and n is the number of observations with a NaN.
                M is the number of features. Now with scaled values.
    """
    
    # Make sure the mode is in all caps.
    if type(mode) == str:
        mode = mode.upper()
    
    # Copy X into a new matrix.
    X_ = numpy.copy(X)

    # None
    if mode is None or mode == "NONE":
        # Literally nothing happens here for now.
        print("Fart noise!")
        
    # Principal component analysis.
    elif mode == 'PCA':
        # Initialise a new PCA.
        pca = decomposition.PCA(n_components=n_components)
        # Fit the PCA with the data.
        pca.fit(X_)
        # Transform the data.
        X_ = pca.transform(X_)
    
    # Independent component analysis.
    elif mode == 'ICA':
        # Initialise a new ICA.
        ica = decomposition.FastICA(n_components=n_components)
        # Fit the ICA with the data.
        ica.fit(X_)
        # Transform the data.
        X_ = ica.transform(X_)
    
    # Factor analysis.
    elif mode == 'FA':
        # Initialise a new factor analysis.
        fa = decomposition.FactorAnalysis(n_components=n_components)
        # Perform the factor analysis on the data.
        fa.fit(X_)
        # Transform the data.
        X_ = fa.transform(X_)
    
    # T-Distributed stochastic neighbour embedding.
    elif mode == 'TSNE':
        # Run several t-SNEs to find a good one.
        n_runs = 10
        Xs_ = []
        dkl = numpy.ones(n_runs, dtype=float) * numpy.inf
        print("Running %d t-SNEs to find lowest Kullback-Leibler divergence." \
            % (n_runs))
        for i in range(n_runs):
            # Initialise a new t-distributed stochastic neighbouring embedding
            #  (t-SNE) analysis.
            tsne = TSNE(n_components=n_components)
            # Copy the data into a new variable.
            Xs_.append(numpy.copy(X_))
            # Fit to and transform the data.
            Xs_[i] = tsne.fit_transform(Xs_[i])
            # Get the KL-divergence.
            dkl[i] = tsne.kl_divergence_
            print("\tCurrent KL-divergence = %.5f" % (dkl[i]))
        # Choose the solution with the lowest KL-divergence.
        X_ = numpy.copy(Xs_[numpy.argmin(dkl)])
        # Get rid of all the excess X copies.
        del Xs_
    
    # Uniform manifold approximation and projection.
    elif mode == 'UMAP':
        # Create a new UMAP instance.
        um = umap.UMAP(n_components=n_components, min_dist=0.01)
        # Fit and transform X.
        X_ = um.fit_transform(X_)
    
    # Gaussian Random Projection.
    elif mode == 'RANDOMPROJECTION':
        # Create a new GaussianRandomProjection instance.
        rp = GaussianRandomProjection(n_components=n_components)
        # Fit and transform X.
        X_ = rp.fit_transform(X_)
    
    # Feature Agglomeration.
    elif mode == 'FEATUREAGGLOMERATION':
        # Create a new FeatureAgglomeration instance.
        fa = cluster.FeatureAgglomeration(n_clusters=n_components)
        # Fit and transform X.
        X_ = fa.fit_transform(X_)
    
    # Isomap.
    elif mode == 'ISOMAP':
        # Create a new Isomap instance.
        im = Isomap(n_components=n_components)
        # Fit and transform X.
        X_ = im.fit_transform(X_)
    
    # Locally Linear Embedding.
    elif mode == 'LLE':
        # Create a new LocallyLinearEmbedding instance.
        lle = LocallyLinearEmbedding(n_neighbors=10, n_components=n_components, \
            method='standard', eigen_solver='dense')
        # Fit and transform X.
        X_ = lle.fit_transform(X_)
    
    # Hessian eigenmaps.
    elif mode == 'HESSIAN':
        # Create a new LocallyLinearEmbedding instance.
        hlle = LocallyLinearEmbedding(n_neighbors=10, n_components=n_components, \
            method='hessian', eigen_solver='dense')
        # Fit and transform X.
        X_ = hlle.fit_transform(X_)
    
    # MLLE.
    elif mode == 'MLLE':
        # Create a new LocallyLinearEmbedding instance.
        mlle = LocallyLinearEmbedding(n_neighbors=10, n_components=n_components, \
            method='modified', eigen_solver='dense')
        # Fit and transform X.
        X_ = mlle.fit_transform(X_)
    
    # LTSA.
    elif mode == 'LTSA':
        # Create a new LocallyLinearEmbedding instance.
        ltsa = LocallyLinearEmbedding(n_neighbors=10, n_components=n_components, \
            method='ltsa', eigen_solver='dense')
        # Fit and transform X.
        X_ = ltsa.fit_transform(X_)
    
    # Multi-dimensional scaling.
    elif mode == 'MDS':
        # Create a new MDS instance.
        mds = MDS(n_components=n_components)
        # Fit and transform X.
        X_ = mds.fit_transform(X_)
    
    # Dictionary Learning
    elif mode == "DICTIONARY":
        # Create a DictionaryLearning instance.
        dictlearn = decomposition.DictionaryLearning( \
            n_components=n_components, \
            fit_algorithm='cd', \
            # The 'omp' algorithm orthogonalises the whole thing, whereas
            # a lasso solution with a low alpha leaves a slightly more
            # scattered solution.
            transform_algorithm='lasso_cd', \
            transform_alpha=0.1, \
            )
        # Fit and transform X.
        X_ = dictlearn.fit_transform(X)
    
    # Truncated SVD (also known as 'Latent Semantic analysis' (LSE)
    elif mode in ['TSVD', 'LSE']:
        tsvd = decomposition.TruncatedSVD(n_components=n_components)
        # Fit and transform X.
        X_ = tsvd.fit_transform(X)
    
    else:
        raise Exception("Unrecognised dimensionality reduction mode '%s'" % (mode))
    
    return X_


def clustering(X, mode="KMEANS", n_clusters=None):
    
    """Reduces the number of dimensions in which a dataset is defined.
    
    Arguments

    X       -   NumPy array with shape (N,M), where N is the number of
                observations, and M the number of features.
    
    Keyword Arguments
    
    mode            -   String that defines the type of clustering algorithm:
                        - None
                        - "KMEANS" requires n_clusters to be set
                        - "WARD" requires n_clusters to be set
                        - "SPECTRAL" requires n_clusters to be set
                        - "AVERAGELINKAGE" requires n_clusters to be set
                        - "BIRCH" requires n_clusters to be set
                        - "DBSCAN"
                        - "HDBSCAN"
                        - "AFFINITYPROPAGATION"
                        - "MEANSHIFT"
                        Default = "KMEANS"
    
    n_clusters      -   Integer number of clusters after clustering, or None
                        if the clustering mode does not require one. Note that
                        this is a predefined value, and NOT an optimally
                        chosen number! Default = None
    
    Returns
    
    y               -   NumPy array with shape (N,), where N is the number of
                        observations. Each value indicates cluster membership,
                        with values of -1 denoting samples that were not
                        assigned a cluster. Cluster labels start counting from
                        0, and go up to however many clusters were found.
    """
    
    # Skip this whole thing if mode is None.
    if mode is None:
        print("Fart noise!")
        y = numpy.zeros((X.shape[0],), dtype=int)
        return y
    
    # Make sure the mode is in caps.
    mode = mode.upper()
    
    # Check for users who didn't specify the number of clusters.
    if mode in ['KMEANS', 'WARD', 'SPECTRAL', 'AVERAGELINKAGE', 'BIRCH', "COSINE", "EUCLIDEAN", "CITYBLOCK", "MANHATTAN"] and \
        n_clusters is None:
        raise Exception("Clustering algorithm '%s' requires you to set a number of clusters!" \
            % (mode))

    if mode == 'KMEANS':
        # Initialise a new KMeans cluster.
        algorithm = cluster.KMeans( \
            n_clusters=n_clusters, \
            init='k-means++', \
            n_init=100, \
            max_iter=300, \
            tol=0.0001, \
            random_state=0)

    elif mode == 'WARD':
        # Create a connectivity matrix.
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # Make the connectivity matrix symmetrical.
        connectivity = 0.5 * (connectivity + connectivity.T)
        # Initialise an AgglomerativeClustering instance with Ward linkage.
        algorithm = cluster.AgglomerativeClustering( \
            n_clusters=n_clusters, \
            linkage='ward', \
            connectivity=connectivity)
    
    elif mode in ["COSINE", "EUCLIDEAN", "CITYBLOCK", "MANHATTAN"]:
        if mode == "MANHATTAN":
            mode = "CITYBLOCK"
        # Initialise an AgglomerativeClustering instance with average linkage
        # and cosine distance.
        algorithm = cluster.AgglomerativeClustering( \
            n_clusters=n_clusters, \
            linkage="average", \
            affinity=mode.lower())
    
    elif mode == 'SPECTRAL':
        # Create a SpectralClustering instance.
        algorithm = cluster.SpectralClustering(\
            n_clusters=n_clusters, \
            eigen_solver='arpack', \
            affinity="nearest_neighbors")
    
    elif mode == 'AVERAGELINKAGE':
        # Create a connectivity matrix.
        connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
        # Make the connectivity matrix symmetrical.
        connectivity = 0.5 * (connectivity + connectivity.T)
        # Create a new AgglomerativeClustering instance with average linkage.
        algorithm = cluster.AgglomerativeClustering( \
            n_clusters=n_clusters, \
            linkage='average', \
            connectivity=connectivity)
    
    elif mode == 'BIRCH':
        # Create a new Birch instance.
        algorithm = cluster.Birch(n_clusters=n_clusters)

    # DBSCAN
    elif mode == 'DBSCAN':
        # Initialise a new DBSCAN instance.
        algorithm = cluster.DBSCAN()
    
    # HDBSCAN
    elif mode == 'HDBSCAN':
        # Initialise a new HDBSCAN.
        algorithm = hdbscan.HDBSCAN(min_cluster_size=max(5,int(0.05*X.shape[0])))
    
    # AFFINITYPROPAGATION
    elif mode == 'AFFINITYPROPAGATION':
        # Initialise a new AffinityPropagation instance. Higher values for damping
        # and preference will result in fewer clusters. If not restricted, AP has
        # a tendency to find many, MANY clusters.
        algorithm = cluster.AffinityPropagation( \
            damping=0.95, \
            )
    
    # MEANSHIFT
    elif mode == 'MEANSHIFT':
        # Initialise a new MeanShift instance.
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
        algorithm = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # Find clusters in the current data.
    y = algorithm.fit_predict(X)
    
    return y


def convenience_clustering(X, mode, max_clusters, outpath, varnames=None, \
    X_original=None, varnames_original=None):
    
    """Convenience function for running one or a number of clustering analyses.
    If the clustering algorithm specified by mode requires a pre-defined number
    of clusters, this function will run from 1 to max_clusters analyses. If the
    algorithm finds a number of clusters, this function will only run one. For
    each run, plots and output will be generated, and stored in outpath.
    
    Arguments

    X               -   NumPy array with shape (N,M), where N is the number of
                        observations, and M the number of features.
    
    mode            -   String that defines the type of clustering algorithm:
                        - "KMEANS" requires n_clusters to be set
                        - "WARD" requires n_clusters to be set
                        - "SPECTRAL" requires n_clusters to be set
                        - "AVERAGELINKAGE" requires n_clusters to be set
                        - "BIRCH" requires n_clusters to be set
                        - "DBSCAN"
                        - "HDBSCAN"
                        - "AFFINITYPROPAGATION"
                        - "MEANSHIFT"
    
    max_clusters    -   Integer number that indicates the maximum number of
                        clusters to be found in this dataset X.
    
    outpath         -   String indicating the path to a folder in which to
                        store the output of the current clustering analyses.
    """
    
    # Make sure the mode is in all-caps.
    mode = mode.upper()
    
    # Construct variable names if they weren't passed.
    if varnames is None:
        varnames = []
        for i in range(X.shape[1]):
            varnames.append("Feature %d" % (i+1))
    if X_original is not None and varnames_original is None:
        varnames_original = []
        for i in range(X.shape[1]):
            varnames_original.append("Feature %d" % (i+1))
    
    # Check if an output directory exists, and attempt to create one if needed.
    if not os.path.isdir(outpath):
        try:
            os.mkdir(outpath)
        except:
            raise Exception("Output directory does not exist, and could not be created. Intended path:\n%s" \
                % (outpath))
    
    # Choose the right number of runs.
    if mode in ["DBSCAN", "HDBSCAN", "AFFINITYPROPAGATION", "MEANSHIFT"]:
        runs = [1]
    else:
        runs = range(1, max_clusters+1)
    silhouette_coefficient = numpy.zeros(len(runs), dtype=float)
    if X_original is not None:
        silhouette_coefficient_original = numpy.zeros(len(runs), dtype=float)
    calinski_harabaz = numpy.zeros(len(runs), dtype=float)
    
    # Plot the space of the current samples.
    if X.shape[1] > 1 and X.shape[1] < 5:
        print("Plotting data with %d dimensions" % (X.shape[1]))
        if X.shape[1] == 2:
            plot_samples(X, 0, 1, None, None, varnames[0], varnames[1], None, None, \
                os.path.join(outpath, "sample_scatterplot.png"))
        elif X.shape[1] == 3:
            plot_samples(X, 0, 1, None, 2, varnames[0], varnames[1], None, varnames[2], \
                os.path.join(outpath, "sample_scatterplot.png"))
        elif X.shape[1] == 4:
            plot_samples(X, 0, 1, 2, 3, varnames[0], varnames[1], varnames[2], varnames[3], \
                os.path.join(outpath, "sample_scatterplot.png"))
    
    # Plot the original space if an original matrix was passed.
    if X_original is not None:
        if X_original.shape[1] > 1 and X_original.shape[1] < 5:
            print("Plotting original data with %d dimensions" % (X_original.shape[1]))
            if X_original.shape[1] == 2:
                plot_samples(X_original, 0, 1, None, None, varnames_original[0], varnames_original[1], None, None, \
                    os.path.join(outpath, "sample_scatterplot_original.png"))
            elif X_original.shape[1] == 3:
                plot_samples(X_original, 0, 1, None, 2, varnames_original[0], varnames_original[1], None, varnames_original[2], \
                    os.path.join(outpath, "sample_scatterplot_original.png"))
            elif X_original.shape[1] == 4:
                plot_samples(X_original, 0, 1, 2, 3, varnames_original[0], varnames_original[1], varnames_original[2], varnames_original[3], \
                    os.path.join(outpath, "sample_scatterplot_original.png"))
    
    # Plot a correlation matrix if an original matrix was passed.
    if X_original is not None:
        print("Plotting correlation matrix with %dx%d dimensions" % \
            (X.shape[1], X_original.shape[1]))
        correlation_matrix(X, X_original, varnames=varnames_original, \
            sig=0.05, vlim=1.0, ax=None, \
            savepath=os.path.join(outpath, "correlation_matrix.png"))
        # Plot the original correlation matrix too.
        correlation_matrix(X_original, X_original, varnames=varnames_original, \
            sig=0.05, vlim=1.0, ax=None, \
            savepath=os.path.join(outpath, "correlation_matrix_original.png"))
    
    # Loop through all runs.
    print("Running through %d %s analyses" % (len(runs), mode))
    for k, n in enumerate(runs):
        
        # Perform the clustering analysis.
        y = clustering(X, mode=mode, n_clusters=n)

        # Count the number of detected clusters.
        n = len(numpy.unique(y))
        
        # Compute the sillhouettes for this solution.
        if n > 1:
            # Compute the sillhouette coefficient for all samples, then average.
            silhouette_vals = silhouette_samples(X, y, metric='euclidean')
            silhouette_avg = numpy.mean(silhouette_vals)
            # Save the average.
            silhouette_coefficient[k] = silhouette_avg
        
            # Also compute the silhouette coefficient for the un-transformed
            # data if provided.
            if X_original is not None:
                silhouette_vals_original = silhouette_samples(X_original, y, metric='euclidean')
                # Save the average.
                silhouette_coefficient_original[k] = numpy.mean(silhouette_vals_original)
        
            # Compute the Calinski-Harabaz index.
            calinski_harabaz[k] = calinski_harabaz_score(X, y)
    
        # Plot the clusters in the data space.
        if X.shape[1] == 2:
            zi = None
            zvar = None
        elif X.shape[1] == 3:
            zi = 2
            zvar = varnames[2]
        if X.shape[1] == 2 or X.shape[1] == 3:
            plot_clusters(X, y, 0, 1, zi, varnames[0], varnames[1], zvar, \
                savepath=os.path.join(outpath, "%s_%d-clusters_scatterplot.png" % (mode,n)), \
                ax=None, cluster_centres=None)

        # If possible, plot the clusters in the original space.
        if X_original is not None:
            if X_original.shape[1] == 2:
                zi = None
                zvar = None
            elif X_original.shape[1] == 3:
                zi = 2
                zvar = varnames_original[2]
            if X_original.shape[1] == 2 or X_original.shape[1] == 3:
                plot_clusters(X_original, y, 0, 1, zi, varnames_original[0], varnames_original[1], zvar, \
                    savepath=os.path.join(outpath, "%s_%d-clusters_scatterplot_original.png" % (mode,n)), \
                    ax=None, cluster_centres=None)

        # Plot the cluster silhouettes.
        if n > 1:
            plot_silhouette(X, y, \
                savepath=os.path.join(outpath, "%s_%d-clusters_silhouettes.png" % (mode,n)))
            if X_original is not None:
                plot_silhouette(X_original, y, \
                    savepath=os.path.join(outpath, "%s_%d-clusters_silhouettes_original.png" % (mode,n)))
        
        # Plot the averages.
        plot_averages(X, y, \
            savepath=os.path.join(outpath, "%s_%d-clusters_averages.png" % (mode,n)), \
            varnames=varnames)
        if X_original is not None:
            plot_averages(X_original, y, \
                savepath=os.path.join(outpath, "%s_%d-clusters_averages_original.png" % (mode,n)), \
                varnames=varnames_original)

    # Plot the outcomes.
    if len(runs) > 1:
        print("Plotting clustering quality indices")

        # Plot the silhouette coefficient.
        fig, ax = pyplot.subplots(nrows=1, ncols=1)
        ax.plot(runs, silhouette_coefficient, 'o-', color='#ff69b4')
        ax.set_xlim([0, max(runs)+1])
        ax.set_xticks(range(1, max(runs)+1))
        ax.set_xlabel("Number of clusters")
        ax.set_ylim([0, 1])
        ax.set_ylabel("Average silhouette coefficient")
        fig.savefig(os.path.join(outpath, "silhouette_coefficient.png"))
        pyplot.close(fig)
        
        # Plot the silhouette coefficient with regards to the original data.
        if X_original is not None:
            fig, ax = pyplot.subplots(nrows=1, ncols=1)
            ax.plot(runs, silhouette_coefficient_original, 'o-', color='#ff69b4')
            ax.set_xlim([0, max(runs)+1])
            ax.set_xticks(range(1, max(runs)+1))
            ax.set_xlabel("Number of clusters")
            ax.set_ylim([0, 1])
            ax.set_ylabel("Average silhouette coefficient")
            fig.savefig(os.path.join(outpath, "silhouette_coefficient_original.png"))
            pyplot.close(fig)
        
        # Plot the Calinski-Harabaz index.
        fig, ax = pyplot.subplots(nrows=1, ncols=1)
        ax.plot(runs, calinski_harabaz, 'o-', color='#ff69b4')
        ax.set_xlim([0, max(runs)+1])
        ax.set_xticks(range(1, max(runs)+1))
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Calinski-Harabaz index")
        fig.savefig(os.path.join(outpath, "calinski-harabaz_index.png"))
        pyplot.close(fig)
