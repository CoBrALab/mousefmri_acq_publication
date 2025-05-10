#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import pickle
from sklearn.metrics import silhouette_samples, silhouette_score # type: ignore
from statannotations.Annotator import Annotator # type: ignore
from scipy.stats import entropy
import random
import matplotlib.cm as cm


import resp_and_cardiac_plotting_functions


class KMeansCorrelation:
    def __init__(self, n_clusters, n_rep=1, max_iters=300, tol = 1e-4, threshold = 5, random_state=None):
        self.n_clusters = n_clusters
        self.n_rep = n_rep #the number of replicates is num of times to repeat with new initial cluster positions (like n_init in scikit-learn)
        self.max_iters = max_iters #number of iterations in each replicate
        self.tol = tol #convergence tolerance (Frobenius norm between consecutive centroids)
        self.threshold = threshold
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def correlation_distance(self, X, Y):
        # Ensure X and Y have the same shape
        assert X.shape == Y.shape, "Input matrices must have the same shape"
        # Calculate correlation distance
        return 1 - np.corrcoef(X, Y, rowvar=False)[0, 1]

    def fit(self, data):
        best_labels, best_centroids, best_inertia = None, None, np.inf

        for rep in range(self.n_rep):
            # Randomly initialize centroids with a different seed for each repetition
            rng = np.random.default_rng(seed=self.random_state + rep) if self.random_state is not None else np.random.default_rng()
            centroids = data[rng.choice(data.shape[0], self.n_clusters, replace=False)]

            iter_count = 0 
            centers_squared_diff = 1
            #iteratively update the location of the centroid (from random) until the number of max iterations is reached, or centroid location barely changes (tolerance)
            while (centers_squared_diff > self.tol) and (iter_count < self.max_iters):
                # Assign each data point to the nearest centroid based on correlation distance
                distances = np.array([np.apply_along_axis(lambda x: self.correlation_distance(x, centroid), 1, data) for centroid in centroids])
                labels = np.argmin(distances, axis=0)

                # Update centroids based on the mean of assigned data points
                centroids_new = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])

                #compute the tolerance (how different the new centroids are compared to the previous iteration) using Frobenius norm
                centers_squared_diff = np.sum((centroids_new - centroids) ** 2)
                centroids = centroids_new
                
                #update the iteration number
                print(iter_count)
                iter_count = iter_count + 1
                
                #print the convergence situation
                if centers_squared_diff <= self.tol:
                    print(" Random repetition #" + str(rep) + " converged on iteration " + str(iter_count) + " as tolerance of " + str(centers_squared_diff) + " was reached.")
                elif iter_count == self.max_iters:
                    print(" Random repetition #" + str(rep) + " converged at max iteration " + str(iter_count) + " at tolerance of " + str(centers_squared_diff) + ".")
                else:
                    print("\r","initialization: " + str(rep) + " iteration: " + str(iter_count) + " tolerance: " + str(centers_squared_diff), end="")


            # Calculate within-cluster sums of point-to-centroid distances (inertia)
            inertia = np.sum([np.sum((data[labels == i] - centroids[i])**2) for i in range(self.n_clusters)])

            # Update best solution if the current one has lower inertia
            if inertia < best_inertia:
                best_labels, best_centroids, best_inertia = labels, centroids, inertia
        self.labels_ = best_labels
        self.cluster_centers_ = best_centroids
        return self

    def transform(self, data):
        #to get actual correlation, subtract 1 and divide by -1 from the correlation_distance
        correlations = np.array([np.apply_along_axis(lambda x: (self.correlation_distance(x, centroid)-1.0)/-1.0, 1, data) for centroid in self.cluster_centers_])
        return correlations.T

    def predict(self, data):
        correlations = self.transform(data)
        distances = 1-correlations
        prediction = np.argmin(distances, axis=1)
        
        #iterate over the clusters, find the timepoints within the lowest correlation percentile of each cluster, assign all these timepoints to a new cluster
        correlations_within_clust = pd.DataFrame(correlations).loc[prediction==0, 0]
        percentile_cutoff_value = np.percentile(correlations_within_clust,self.threshold, axis = 0)
        bool_corr_within_percentile = correlations_within_clust < percentile_cutoff_value

        #also plot the histograms of correlations within each cluster and the percentile
        fig,axs = plt.subplots(nrows=2, ncols=self.n_clusters, figsize=(self.n_clusters*4,7), dpi = 200)
        axs[1,0].hist(1-correlations_within_clust)
        axs[1,0].axvline(1-percentile_cutoff_value, color = 'red')
        axs[1,0].set_xlabel('Distances (1-r)')
        axs[1,0].set_ylabel('Number of timepoints')
        axs[0,0].set_title('Cluster 0 \n' + str(len(correlations_within_clust)) + ' timepoints \n Discard correlations below ' + str(round(percentile_cutoff_value, 4)))
        axs[0,0].hist(correlations_within_clust)
        axs[0,0].axvline(percentile_cutoff_value, color = 'red')
        axs[0,0].set_xlabel('Correlations (r)')
        axs[0,0].set_ylabel('Number of timepoints')

        for clust in range(1,self.n_clusters):
            correlations_within_clust_new = pd.DataFrame(correlations).loc[prediction==clust, clust]
            percentile_cutoff_value = np.percentile(correlations_within_clust_new,self.threshold, axis = 0)
            bool_corr_within_percentile_new = correlations_within_clust_new < percentile_cutoff_value
            
            #also plot the histograms of correlations within each cluster and the percentile
            axs[1,clust].hist(1-correlations_within_clust)
            axs[1,clust].axvline(1-percentile_cutoff_value, color = 'red')
            axs[1,clust].set_xlabel('Distances (1-r)')
            axs[0,clust].set_title('Cluster ' + str(clust) + '\n' + str(len(correlations_within_clust_new)) + ' timepoints \n Discard correlations below ' + str(round(percentile_cutoff_value, 4)))
            axs[0,clust].hist(correlations_within_clust_new)
            axs[0,clust].axvline(percentile_cutoff_value, color = 'red')
            axs[0,clust].set_xlabel('Correlations (r)')

            #concatenate results across clusters using index
            correlations_within_clust = pd.concat([correlations_within_clust, correlations_within_clust_new])
            bool_corr_within_percentile = pd.concat([bool_corr_within_percentile, bool_corr_within_percentile_new])

        #sort the final df so indices are in order
        bool_corr_within_percentile = bool_corr_within_percentile.sort_index()
        correlations_within_clust = correlations_within_clust.sort_index()

        #plot final figure
        fig.show()

        #where the dist is in lowest percentile, assign that timepoint to a new cluster
        prediction[bool_corr_within_percentile] = clust + 1
        return correlations_within_clust, prediction
    
def silhouette_analysis(phgy_dict_selectvar_zscored, range_n_clusters, session_info_dict, save_path):
    var_list = ['index','subject_ID', 'dex_conc', 'isoflurane_percent', 'actual_ses_order',
                   'sex', 'strain']

#iterate over all cluster numbers
    for n_clusters in range_n_clusters:

        #######################################run the clustering ##########################################
        print('running clustering for ' + str(n_clusters) + ' clusters')
        kmeans_object = KMeansCorrelation(n_clusters=n_clusters, random_state=0, 
                n_rep=10, tol=1e-4).fit(phgy_dict_selectvar_zscored)
        cluster_labels = kmeans_object.labels_
        print('done clustering, now plotting')

        ################################# SAVE THE OUTPUT AS A PICKLE ##################################
        pickle.dump(kmeans_object, open(save_path + "/kmeans_cluster_object_" + str(n_clusters) + 'cluster', "wb"))

        ############################### SILHOUETTE PLOT ###############################################
        #create empty array to store silhouette scores
        silhouette_scores = np.zeros((n_clusters, 2))
        cluster_colors = sns.color_palette("Spectral", n_clusters)[0:n_clusters]
        # Create silhouette plot
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(6, 4)
        ax1.set_xlim([-0.5, 1])
        ax1.set_ylim([0, len(phgy_dict_selectvar_zscored) + (n_clusters + 1) * 10]) # plots of individual clusters, to demarcate them clearly.

        # The silhouette_score (avg over samples) gives a perspective into the density and separation of the formed clusters
        silhouette_avg = silhouette_score(phgy_dict_selectvar_zscored, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(phgy_dict_selectvar_zscored, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cluster_colors[i]
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters. \n Avg score: " + str(silhouette_avg))
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        fig.savefig(save_path + '/stability_analysis_silhouette_n' + str(n_clusters) + '.svg')
        plt.close()

import matplotlib.patheffects as path_effects
def add_median_labels(ax, fs, fmt='.1f'):
    '''Function for adding median labels as text on seaborn boxplots/violinplots'''
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white', fontsize = fs)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def zscore_by_sex_strain(phgy_train, phgy_test, session_info_df):
    phgy_train_and_info = pd.concat([phgy_train, session_info_df.loc[phgy_train.index]['strain'], 
                              session_info_df.loc[phgy_train.index]['sex']], axis = 1) 
    phgy_test_and_info = pd.concat([phgy_test, session_info_df.loc[phgy_test.index]['strain'], 
                              session_info_df.loc[phgy_test.index]['sex']], axis = 1) 
    
    #subset by strain and sex
    phgy_train_and_info_c57m = phgy_train_and_info.loc[(phgy_train_and_info['strain'] == 'C57Bl/6') & (phgy_train_and_info['sex'] == 'm') ]
    phgy_train_and_info_c57f = phgy_train_and_info.loc[(phgy_train_and_info['strain'] == 'C57Bl/6') & (phgy_train_and_info['sex'] == 'f') ]
    phgy_train_and_info_c3m = phgy_train_and_info.loc[(phgy_train_and_info['strain'] == 'C3HeB/FeJ') & (phgy_train_and_info['sex'] == 'm') ]
    phgy_train_and_info_c3f = phgy_train_and_info.loc[(phgy_train_and_info['strain'] == 'C3HeB/FeJ') & (phgy_train_and_info['sex'] == 'f') ]

    #same for the test dataset
    phgy_test_and_info_c57m = phgy_test_and_info.loc[(phgy_test_and_info['strain'] == 'C57Bl/6') & (phgy_test_and_info['sex'] == 'm') ]
    phgy_test_and_info_c57f = phgy_test_and_info.loc[(phgy_test_and_info['strain'] == 'C57Bl/6') & (phgy_test_and_info['sex'] == 'f') ]
    phgy_test_and_info_c3m = phgy_test_and_info.loc[(phgy_test_and_info['strain'] == 'C3HeB/FeJ') & (phgy_test_and_info['sex'] == 'm') ]
    phgy_test_and_info_c3f = phgy_test_and_info.loc[(phgy_test_and_info['strain'] == 'C3HeB/FeJ') & (phgy_test_and_info['sex'] == 'f') ]

    #print num of timepoints in each subset 
    print('Num timepoints c57m (train): ' + str(phgy_train_and_info_c57m.shape[0]) + 'Num timepoints c57f (train): ' + str(phgy_train_and_info_c57f.shape[0]))
    print('Num timepoints c3m (train): ' + str(phgy_train_and_info_c3m.shape[0]) + 'Num timepoints c3f (train): ' + str(phgy_train_and_info_c3f.shape[0]))
    print('Num timepoints c57m (test): ' + str(phgy_test_and_info_c57m.shape[0]) + 'Num timepoints c57f (test): ' + str(phgy_test_and_info_c57f.shape[0]))
    print('Num timepoints c3m (test): ' + str(phgy_test_and_info_c3m.shape[0]) + 'Num timepoints c3f (test): ' + str(phgy_test_and_info_c3f.shape[0]))
    
    #extract only the numeric columns for zscoring
    numeric_cols = phgy_train_and_info_c57m.select_dtypes(include=[np.number]).columns
    
    #apply the zscoring to each subset separattely
    phgy_train_and_info_c57m_z = (phgy_train_and_info_c57m[numeric_cols] - np.mean(phgy_train_and_info_c57m[numeric_cols]))/np.std(phgy_train_and_info_c57m[numeric_cols])
    phgy_train_and_info_c57f_z = (phgy_train_and_info_c57f[numeric_cols]- np.mean(phgy_train_and_info_c57f[numeric_cols]))/np.std(phgy_train_and_info_c57f[numeric_cols])
    phgy_train_and_info_c3m_z = (phgy_train_and_info_c3m[numeric_cols]- np.mean(phgy_train_and_info_c3m[numeric_cols]))/np.std(phgy_train_and_info_c3m[numeric_cols])
    phgy_train_and_info_c3f_z = (phgy_train_and_info_c3f[numeric_cols]- np.mean(phgy_train_and_info_c3f[numeric_cols]))/np.std(phgy_train_and_info_c3f[numeric_cols])

    #print the mean and std for each subset (to know what I'm removing)
    numeric_cols_reordered = ['SpO2', 'PVI', 'HR', 'HRV', 'RV', 'RR', 'RRV']
    mean_summary_df = pd.DataFrame(columns=numeric_cols_reordered,
                                    index = ['C57Bl/6 males', 'C57Bl/6 females', 'C3HeB/FeJ males', 'C3HeB/FeJ females'])
    mean_summary_df.loc['C57Bl/6 males'] = round(np.mean(phgy_train_and_info_c57m[numeric_cols_reordered]),3)
    mean_summary_df.loc['C57Bl/6 females'] = round(np.mean(phgy_train_and_info_c57f[numeric_cols_reordered]),3)
    mean_summary_df.loc['C3HeB/FeJ males'] = round(np.mean(phgy_train_and_info_c3m[numeric_cols_reordered]),3)
    mean_summary_df.loc['C3HeB/FeJ females'] = round(np.mean(phgy_train_and_info_c3f[numeric_cols_reordered]),3)

    std_summary_df = pd.DataFrame(columns=numeric_cols_reordered,
                                    index = ['C57Bl/6 males', 'C57Bl/6 females', 'C3HeB/FeJ males', 'C3HeB/FeJ females'])
    std_summary_df.loc['C57Bl/6 males'] = round(np.std(phgy_train_and_info_c57m[numeric_cols_reordered]),3)
    std_summary_df.loc['C57Bl/6 females'] = round(np.std(phgy_train_and_info_c57f[numeric_cols_reordered]),3)
    std_summary_df.loc['C3HeB/FeJ males'] = round(np.std(phgy_train_and_info_c3m[numeric_cols_reordered]),3)
    std_summary_df.loc['C3HeB/FeJ females'] = round(np.std(phgy_train_and_info_c3f[numeric_cols_reordered]),3)
    
    print(mean_summary_df)
    print(std_summary_df)
    #now apply the zscoring to the test dataset (using mean and std from training)
    phgy_test_and_info_c57m_z = (phgy_test_and_info_c57m[numeric_cols] - np.mean(phgy_train_and_info_c57m[numeric_cols]))/np.std(phgy_train_and_info_c57m[numeric_cols])
    phgy_test_and_info_c57f_z = (phgy_test_and_info_c57f[numeric_cols]- np.mean(phgy_train_and_info_c57f[numeric_cols]))/np.std(phgy_train_and_info_c57f[numeric_cols])
    phgy_test_and_info_c3m_z = (phgy_test_and_info_c3m[numeric_cols]- np.mean(phgy_train_and_info_c3m[numeric_cols]))/np.std(phgy_train_and_info_c3m[numeric_cols])
    phgy_test_and_info_c3f_z = (phgy_test_and_info_c3f[numeric_cols]- np.mean(phgy_train_and_info_c3f[numeric_cols]))/np.std(phgy_train_and_info_c3f[numeric_cols])

  
    ####################### plot comparison of phgy in the subsets
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 
    plt.rc('axes', labelsize=10)
    pairs = [(('C57Bl/6', 'm'), ('C57Bl/6', 'f')),
             (('C3HeB/FeJ', 'm'), ('C3HeB/FeJ', 'f')),
             (('C57Bl/6', 'm'), ('C3HeB/FeJ', 'm')),
             (('C57Bl/6', 'f'), ('C3HeB/FeJ', 'f'))]
    phgy_var = ['RR', 'RRV', 'RV', 'HR', 'HRV', 'PVI', 'SpO2']
    fig, axs = plt.subplots(1,7, figsize = (28, 4), sharex = False, dpi = 200)

    for i in range(0, len(phgy_var)):
        plotting_parameters = {
            'data':    phgy_train_and_info,
            'x':       "strain",
            'hue':      "sex",
            'y':       phgy_var[i],
            'palette': "Greys"
        }
        sns.boxplot(**plotting_parameters, ax = axs[i])
        #label the median values
        add_median_labels(axs[i], fs = 7)
        #add significance stars
        annotator = Annotator(axs[i], pairs, **plotting_parameters)
        annotator.configure(test='t-test_ind', text_format='star', loc='inside',comparisons_correction="fdr_bh",
                        verbose = 0)
        annotator.apply_test(num_comparisons = 50).annotate()
        annotator.reset_configuration()
        if i>0:
            axs[i].legend([],[], frameon=False)
        else:
            axs[i].legend(fontsize = 7)
    fig.tight_layout()
    plt.show()
    phgy_train_strainsex_zscore = pd.concat([phgy_train_and_info_c57m_z, phgy_train_and_info_c57f_z, phgy_train_and_info_c3m_z, phgy_train_and_info_c3f_z],
                                  axis = 0)
    phgy_test_strainsex_zscore = pd.concat([phgy_test_and_info_c57m_z, phgy_test_and_info_c57f_z, phgy_test_and_info_c3m_z, phgy_test_and_info_c3f_z],
                                  axis = 0)
    return phgy_train_strainsex_zscore, phgy_test_strainsex_zscore, mean_summary_df, std_summary_df

def kl_divergence(p, q):
    """Compute Kullback-Leibler divergence"""
    return entropy(p, q)

def jensen_shannon_divergence(p, q):
    """Compute Jensen-Shannon divergence"""
    # Normalize distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute KL divergence for both directions
    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)

    # Compute average of KL divergences
    avg_kl = (kl_pq + kl_qp) / 2

    # Compute square root of average KL divergence
    return np.sqrt(avg_kl)

def random_combination(iterable, r, random_state):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    random.seed(random_state)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

def custom_stratified_train_test_split_with_group(X, y, group, test_size=0.25, random_state=None, max_imbalance = 1e-04, max_iter = 10000, **kwargs):
    """Optimize test groups based on class imbalance in y"""

    # Initialize with a random selection of groups
    test_groups = np.random.RandomState(seed=random_state).choice(np.unique(group),
                                                                    size=int(test_size * len(np.unique(group))),
                                                                    replace=False)
    test_indices = []
    for grp in test_groups:
        test_indices.extend(np.where(group == grp)[0])
    
    train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Calculate the initial imbalance in the classes of y based on the original dataset
    p_full = np.bincount(y) / len(y)
    p_train = np.bincount(y_train) / len(y_train)
    p_test = np.bincount(y_test) / len(y_test)
    imbalance = jensen_shannon_divergence(p_train, p_full) + jensen_shannon_divergence(p_test, p_full)
    
    print('Initial test groups: ' + str(test_groups) + ', train-full divergence: ' + str(jensen_shannon_divergence(p_train, p_full)) + ' test-full divergence: ' + str(jensen_shannon_divergence(p_test, p_full)))
    
    # Initialize the best combination of groups
    best_test_groups = np.copy(test_groups)
    best_imbalance = imbalance
    
    # Find the best combination of test groups by trying random combinations and checking if they're better
    
    for seed in range(0,max_iter):

        #generate a random combination
        combo = random_combination(np.unique(group), len(test_groups), seed)

        # Skip the current combination if it's the same as the initial test_groups
        if np.array_equal(combo, test_groups):
            print('same array')
            continue
        
        combo_test_indices = []
        for grp in combo:
            combo_test_indices.extend(np.where(group == grp)[0])
        
        combo_train_indices = np.setdiff1d(np.arange(len(y)), combo_test_indices)
        combo_y_train = y[combo_train_indices]
        combo_y_test = y[combo_test_indices]
        
        # Calculate the imbalance for the current combination
        p_combo_train = np.bincount(combo_y_train) / len(combo_y_train)
        p_combo_test = np.bincount(combo_y_test) / len(combo_y_test)
        combo_imbalance = jensen_shannon_divergence(p_combo_train, p_full) + jensen_shannon_divergence(p_combo_test, p_full)
        
        # If the imbalance is below the threshold, stop the optimization
        if combo_imbalance < max_imbalance:
            break

        # If the imbalance is improved, update the best combination of groups
        if combo_imbalance < best_imbalance:
            best_test_groups = combo
            best_imbalance = combo_imbalance
            print('Better combo found at : ' + str(seed) + ' with groups: ' + str(best_test_groups) + ', train-full divergence: ' + str(jensen_shannon_divergence(p_combo_train, p_full)) + ' test-full divergence: ' + str(jensen_shannon_divergence(p_combo_test, p_full)))
                
    # Use the best combination of groups
    best_test_indices = []
    for grp in best_test_groups:
        best_test_indices.extend(np.where(group == grp)[0])
    
    best_train_indices = np.setdiff1d(np.arange(len(y)), best_test_indices)
    X_train, X_test = X.loc[best_train_indices], X.loc[best_test_indices]
    y_train, y_test = y[best_train_indices], y[best_test_indices]
    
    print("Final imbalance:" +  str(best_imbalance) + ' found at iteration ' + str(seed))
    print("Groups in test set:", best_test_groups)
    print("Groups in train set:", np.unique(group[best_train_indices]))
    
    return X_train.drop('Unnamed: 0', axis = 1), X_test.drop('Unnamed: 0', axis = 1), y_train, y_test

def transition_matrix(transitions):
    n = 1+ max(transitions) #number of states

    M = [[0]*n for _ in range(n)]

    for (i,j) in zip(transitions,transitions[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

def phgy_cluster_characteristics(kmeans_object, phgy_train_test_df, num_clusters, session_info_df,
                                phgy_dict_postCensor_shape, fig_save_path, csv_save_path, cluster_axs_order_dict):
    var_list = ['index','subject_ID', 'dex_conc', 'isoflurane_percent', 'actual_ses_order',
                   'sex', 'strain']

    #find right cluster object    
    kmeans = pickle.load(open(kmeans_object, 'rb'))
    
    #for each cluster, which phgy variables is it loading on?
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_)
    cluster_centers.columns = phgy_train_test_df.columns
    
    ################ PLOT LOADINGS ####################
    if num_clusters <=5:
        cluster_colors = cm.tab20b_r(range(20))[[1, 5, 9, 13, 17]]
    else:
        cluster_colors = cm.Spectral(range(num_clusters))
    fig1, axes = plt.subplots(num_clusters, 1, figsize = (10,10), sharex = True)
    for cluster in range(0,num_clusters):
        fig1b = sns.barplot(data = pd.DataFrame(cluster_centers.loc[cluster]).transpose(), ax = axes[cluster_axs_order_dict[str(cluster)]], 
                           color = cluster_colors[cluster])
        axes[cluster_axs_order_dict[str(cluster)]].set_ylabel('Cluster ' + str(cluster) + '\n (SD)')
        axes[cluster_axs_order_dict[str(cluster)]].set_ylim([-2,2])
    labels = fig1b.set_xticklabels(fig1b.get_xticklabels(), 
                              rotation=45, 
                              horizontalalignment='right')
    fig1.tight_layout()
    fig1.savefig(fig_save_path + '/phgy_cluster_loadings_nclust' + str(num_clusters) + '_commonyax.svg',
                bbox_inches='tight')
    plt.close()
    
    #REPEAT SAME PLOT, BUT DON'T SHARE Y AXIS (CAN SEE DYNAMIC RANGE)
    fig1, axes = plt.subplots(num_clusters, 1, figsize = (10,10), sharex = True)
    for cluster in range(0,num_clusters):
        fig1b = sns.barplot(data = pd.DataFrame(cluster_centers.loc[cluster]).transpose(), ax = axes[cluster_axs_order_dict[str(cluster)]], 
                           color = cluster_colors[cluster])
        axes[cluster_axs_order_dict[str(cluster)]].set_ylabel('Cluster ' + str(cluster) + '\n (SD)')
    labels = fig1b.set_xticklabels(fig1b.get_xticklabels(), 
                              rotation=45, 
                              horizontalalignment='right')
    fig1.tight_layout()
    fig1.savefig(fig_save_path + '/phgy_cluster_loadings_nclust' + str(num_clusters) + '.svg',
                bbox_inches='tight')
    plt.close()

    #add information on the distance of each timepoint to each cluster to the dataframe too
    correlations = kmeans.transform(phgy_train_test_df.reset_index(drop = True))
    corr_within_clust, predictions = kmeans.predict(phgy_train_test_df)
    session_info_df = session_info_df.reset_index(drop = True)
    for cluster in range(0,num_clusters):
        session_info_df['corr_to_clust' + str(cluster)] = correlations[:,cluster]
    session_info_df['corr_within_clust'] = corr_within_clust
    session_info_df['cluster_prediction'] = predictions
    final_df = pd.concat([session_info_df, 
                          phgy_train_test_df.reset_index(drop = True)], axis = 1) #also include original, zscored phgy var in csv
    final_df.to_csv(csv_save_path) 
    
    #for each cluster, which subjects/strains/iso levels etc
    for variable in var_list:
        fig2 = sns.displot(data=session_info_df, x="cluster_prediction", hue = variable, multiple = "stack", height = 3.5,
                    aspect = 1.5)
        fig2.savefig(fig_save_path + '/phgy_cluster_sampleDistribution_wrt-' + variable + '_nclust' + str(num_clusters) + '.png')
        plt.close()    
    
    #for each scan and iso level, how does the timecourse change between clusters
    for sub_ses_ID in phgy_dict_postCensor_shape.keys():
        for iso_percent in [0.23, 0.5, 1]:   
            fig3,axs = plt.subplots(nrows=8, ncols=1, figsize=(15,15), dpi = 300)
            scan_segment_info = session_info_df.loc[(session_info_df['index'] == sub_ses_ID) & (session_info_df['isoflurane_percent'] == iso_percent)]
            timewindow=0
            window_size = 120
            starttime= 0
            while starttime<480:
                for cluster in range(0,num_clusters):
                    axs[timewindow].plot(scan_segment_info['corr_to_clust' + str(cluster)][starttime: starttime + window_size],
                                        color = cluster_colors[cluster])
                    axs[timewindow+1].plot(scan_segment_info['cluster_prediction'][starttime: starttime + window_size]==cluster,
                                          color = cluster_colors[cluster])
                    axs[timewindow+1].set_facecolor('whitesmoke')
                    axs[timewindow].set_ylabel('Correlation \n to cluster')
                    axs[timewindow+1].set_ylabel('Cluster \n assignment')
                timewindow = timewindow + 2
                starttime = starttime + window_size
            axs[7].set_xlabel('Time (seconds)')
            fig3.savefig(fig_save_path + '/phgy_cluster_timecourse_' + str(sub_ses_ID) + '_iso-' + str(iso_percent) + '.png')
            plt.close()
            
            if scan_segment_info.shape[0] != 0:
                fig4,axs = plt.subplots(nrows=1, ncols=2,dpi = 200, figsize = (7,3))
                matrix = transition_matrix(scan_segment_info['cluster_prediction'])
                sns.countplot(data=scan_segment_info, x="cluster_prediction", ax = axs[0], palette = cluster_colors )
                axs[1] = sns.heatmap(matrix)
                axs[1].set(xlabel = 'To cluster', ylabel = 'From cluster')
                axs[0].set_title('Dwell time per cluster')
                axs[1].set_title('Transition matrix')
                fig4.tight_layout()
                fig4.savefig(fig_save_path + '/phgy_cluster_dwell_and_transitions_' + str(sub_ses_ID) + '_iso-' + str(iso_percent) + '.png')
                plt.close()
                
    return session_info_df, correlations, predictions

def plot_CPP_and_raw_phgy(savename, fmri_phgy_nan_censor_dict, phgy_dict_selectvar_precensor, 
                          clustering_precensor, resp_csv, pulseox_csv, nclust):
    resp_df = pd.read_csv(resp_csv, header = None)[0]
    pulseox_df = pd.read_csv(pulseox_csv, header = None)[0]
    clustering_precensor = clustering_precensor.reset_index(drop=True)

    #define time array for each type of physiology depending on their sampling rate
    resp_sampling_rate = int(324000/1440)
    pulseox_sampling_rate = int(648000/1440)
    resp_time_df = np.arange(start=0, stop=1440 , step=1/resp_sampling_rate)
    pulseox_time_df = np.arange(start=0, stop=1440 , step=1/pulseox_sampling_rate)

    #smooth the resp and pulseox signals for nicer plots
    resp_df_smooth = resp_and_cardiac_plotting_functions.denoise_detrend_resp(resp_df,resp_sampling_rate, True )
    pulseox_df_smooth = resp_and_cardiac_plotting_functions.denoise_cardiac(pulseox_df,pulseox_sampling_rate,
                                                                            False )
    #plot the traces within a 30s window
    start = 30 #start at 30 because of the edge cutoff
    end = 60
    while end <= 1410:
        fig,axs = plt.subplots(nrows=10, ncols=1, figsize=(12,6), sharex = False)
        axs[0].plot(pulseox_time_df[pulseox_sampling_rate*start:pulseox_sampling_rate*end],
                    pulseox_df_smooth[pulseox_sampling_rate*start:pulseox_sampling_rate*end], color = 'black')
        axs[1].plot(resp_time_df[resp_sampling_rate*start:resp_sampling_rate*end],
                    resp_df_smooth[resp_sampling_rate*start:resp_sampling_rate*end], color = 'black')
        axs[0].set_ylim([0, 130])
        axs[1].set_ylim([-20, 30])
        
        phgy_precensor_window = phgy_dict_selectvar_precensor[start:end]
        mask_window = np.array((fmri_phgy_nan_censor_dict[start:end] == False)).reshape(1,-1)

        ax_num = 2
        vmin_dict = {'RR': 40, 'RRV': 0.0, 'RV': 1, 'HR': 50, 'HRV': 0.0, 'PVI': 16, 'SpO2': 40}
        vmax_dict = {'RR': 180, 'RRV': 0.8, 'RV': 20, 'HR': 412, 'HRV': 0.8, 'PVI': 100, 'SpO2': 100}
        for phgy_var in phgy_precensor_window.columns:
            if phgy_var == 'RR' or phgy_var == 'HR' or phgy_var == 'PVI' or phgy_var == 'SpO2':
                xticks_label = False
                decimals = 0
            else:
                xticks_label = False
                decimals = 2
            sns.heatmap(np.array(phgy_precensor_window[str(phgy_var)].round(decimals)).reshape(1,-1),
                        mask = mask_window, xticklabels = xticks_label, yticklabels = False, 
                        cmap = 'Greys', vmin = vmin_dict[phgy_var], vmax = vmax_dict[phgy_var],
                        annot = True, annot_kws={'fontsize': 8}, fmt='g', square = True, cbar = False, ax = axs[ax_num])
            axs[ax_num].set_ylabel(str(phgy_var))
            ax_num = ax_num + 1
        #plot the cluster membership too
        corr_values_window = np.array(clustering_precensor['corr_within_clust'][start:end].round(2) + clustering_precensor['cluster_prediction'][start:end])
        sns.heatmap(corr_values_window.reshape(1,-1),
                    mask = mask_window, xticklabels= clustering_precensor[start:end].index,
                    yticklabels = False, cmap = 'tab20b_r', vmin = 0, vmax = 5,
                    annot = True, annot_kws={'fontsize': 8}, fmt='g', square = True, cbar = False, ax = axs[9])

        axs[9].set_ylabel('cluster')
        axs[0].set_ylabel('Pulse \n oximetry \n trace')
        axs[1].set_ylabel('Resp \n trace')
        axs[9].set_xlabel('Time')
        
        #remove borders
        for j in range(0,2):
            axs[j].set_xlim([start, end])
            axs[j].spines['top'].set_visible(False)
            axs[j].spines['right'].set_visible(False)
            axs[j].spines['bottom'].set_visible(False)
            axs[j].get_yaxis().set_ticks([])
            if j < 3:
                axs[j].get_xaxis().set_ticks([])
        axs[9].spines['bottom'].set_visible(True)
        plt.tight_layout()
        fig.savefig(str(savename) + '-start' + str(start) + '.svg')
        plt.close()
        start = start + 30
        end = end+30

def plot_CPP_and_raw_phgy_zscore(savename, fmri_phgy_nan_censor_dict, phgy_dict_selectvar_precensor_zscore, 
                          clustering_precensor, resp_csv, pulseox_csv, nclust):
    resp_df = pd.read_csv(resp_csv, header = None)[0]
    pulseox_df = pd.read_csv(pulseox_csv, header = None)[0]
    clustering_precensor = clustering_precensor.reset_index(drop=True)
    phgy_dict_selectvar_precensor_zscore = phgy_dict_selectvar_precensor_zscore.reset_index(drop = True)

    #define time array for each type of physiology depending on their sampling rate
    resp_sampling_rate = int(324000/1440)
    pulseox_sampling_rate = int(648000/1440)
    resp_time_df = np.arange(start=0, stop=1440 , step=1/resp_sampling_rate)
    pulseox_time_df = np.arange(start=0, stop=1440 , step=1/pulseox_sampling_rate)

    #smooth the resp and pulseox signals for nicer plots
    resp_df_smooth = resp_and_cardiac_plotting_functions.denoise_detrend_resp(resp_df,resp_sampling_rate, True )
    pulseox_df_smooth = resp_and_cardiac_plotting_functions.denoise_cardiac(pulseox_df,pulseox_sampling_rate,
                                                                            False )
    #plot the traces within a 30s window
    start = 30 #start at 30 because of the edge cutoff
    end = 60
    while end <= 1410:
        fig,axs = plt.subplots(nrows=10, ncols=1, figsize=(12,6), sharex = False)
        axs[0].plot(pulseox_time_df[pulseox_sampling_rate*start:pulseox_sampling_rate*end],
                    pulseox_df_smooth[pulseox_sampling_rate*start:pulseox_sampling_rate*end], color = 'black')
        axs[1].plot(resp_time_df[resp_sampling_rate*start:resp_sampling_rate*end],
                    resp_df_smooth[resp_sampling_rate*start:resp_sampling_rate*end], color = 'black')
        axs[0].set_ylim([0, 130])
        axs[1].set_ylim([-20, 30])
        
        phgy_precensor_window = phgy_dict_selectvar_precensor_zscore[start:end]
        mask_window = np.array((fmri_phgy_nan_censor_dict[start:end] == False)).reshape(1,-1)

        ax_num = 2
        for phgy_var in phgy_precensor_window.columns:
            xticks_label = False
            decimals = 2
            sns.heatmap(np.array(phgy_precensor_window[str(phgy_var)].round(decimals)).reshape(1,-1),
                        mask = mask_window, xticklabels = xticks_label, yticklabels = False, 
                        cmap = 'RdGy', vmin = -1.5, vmax = 1.5,
                        annot = True, annot_kws={'fontsize': 8}, fmt='g', square = True, cbar = False, ax = axs[ax_num])
            axs[ax_num].set_ylabel(str(phgy_var))
            ax_num = ax_num + 1
        #plot the cluster membership too
        corr_values_window = np.array(clustering_precensor['corr_within_clust'][start:end].round(2) + clustering_precensor['cluster_prediction'][start:end])
        sns.heatmap(corr_values_window.reshape(1,-1),
                    mask = mask_window, xticklabels= clustering_precensor[start:end].index,
                    yticklabels = False, cmap = 'tab20b_r', vmin = 0, vmax = 5,
                    annot = True, annot_kws={'fontsize': 8}, fmt='g', square = True, cbar = False, ax = axs[9])

        axs[9].set_ylabel('cluster')
        axs[0].set_ylabel('Pulse \n oximetry \n trace')
        axs[1].set_ylabel('Resp \n trace')
        axs[9].set_xlabel('Time')
        
        #remove borders
        for j in range(0,2):
            axs[j].set_xlim([start, end])
            axs[j].spines['top'].set_visible(False)
            axs[j].spines['right'].set_visible(False)
            axs[j].spines['bottom'].set_visible(False)
            axs[j].get_yaxis().set_ticks([])
            if j < 3:
                axs[j].get_xaxis().set_ticks([])
        axs[9].spines['bottom'].set_visible(True)
        plt.tight_layout()
        fig.savefig(str(savename) + '-start' + str(start) + '_zscored.svg')
        plt.close()
        start = start + 30
        end = end+30