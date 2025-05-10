#!/usr/bin/env python
# coding: utf-8

'''
Purpose of performing CAP analysis:
The CAP analysis is used to obtain a quantification of network detectability on a timepoint level 
(provided by the distance-correlation between each timepoint and the CAP centroids). This is necessary
when dividing the data for physiological clustering into train and test sets that are equally balanced in terms
of network detectability. Possibly there's a more efficient/short way of achieving this output without caps, 
but I already had these outputs from previous experimentation.

Purpose of this script: perform kmeans clustering (CAPs) on the Grandjean and Yeow dataset (a high quality dataset).
I'm defining my clusters on an external dataset instead of the one I acquired because I want to obtain
clean centroids (ie networks), and my own dataset has too much heterogeneity in data quality.
This particular script will test different number of clusters/centroids, and output the spatial maps of each
centroid as well as the silhouette plot, so that I can choose the ideal number of centroids.

Using a python script instead of a notebook allows to run it with qbatch, doing each cluster size in parallel.
'''

import os
from sklearn.cluster import KMeans
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map 
import pickle
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

import clustering_functions
from clustering_functions import KMeansCorrelation

# # Take the arguments from bash - the cluster size
n_clusters=int(sys.argv[1])
print(n_clusters)

#load fMRI data - use the Grandjean and Yeow dataset (cleaned, censored timeseries), RESAMPLED TO MY RESOLUTION
final_data_folder='/data/chamal/projects/mila/2021_fMRI_dev/part2_phgy_fmri_project/4_derivatives/rabies_runs/mediso-grandjean_yeow-forCAP'
epi_files_ref_resampled = sorted(glob.glob(final_data_folder + '/rabies_out_cc-v050_resampled-to-local_05smoothed_lowpass/confound_correction_datasink/cleaned_timeseries/*/sub*'))
commonspace_template_file_ref_resampled = final_data_folder + '/rabies_out_preprocess-v050_resampled-to-local/bold_datasink/commonspace_resampled_template/resampled_template.nii.gz'
commonspace_mask_file_ref_resampled = os.path.abspath(sorted(glob.glob(final_data_folder + '/rabies_out_preprocess-v050_resampled-to-local/bold_datasink/commonspace_mask/*/*/*_brain_mask.nii.gz'))[0])

ref_epi_resampled_flat,ref_epi_dict = clustering_functions.extract_array_of_epis_with_dict(epi_files_ref_resampled, 
                                                                           commonspace_mask_file_ref_resampled, 
                                                                           [0,len(epi_files_ref_resampled)], None, True, False)

    
#######################################run the clustering ##########################################
print('running clustering for ' + str(n_clusters) + ' clusters')
data = np.transpose(ref_epi_resampled_flat)
kmeans_object = KMeansCorrelation(n_clusters=n_clusters, n_rep=10,  max_iters=300, random_state=0).fit(data)
cluster_labels = kmeans_object.labels_
print('done clustering, now plotting')

################################# SAVE THE OUTPUT AS A PICKLE ##################################
pickle.dump(kmeans_object, open("./CAP_ncluster_selection/kmeans_cluster_object_" + str(n_clusters) + 'cluster', "wb"))

############################### SILHOUETTE PLOT ###############################################
# Create silhouette plot
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(6, 4)
ax1.set_xlim([-0.5, 1])
ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10]) # plots of individual clusters, to demarcate them clearly.

# The silhouette_score (avg over samples) gives a perspective into the density and separation of the formed clusters
silhouette_avg = silhouette_score(data, cluster_labels)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(data, cluster_labels)

y_lower = 10
for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / n_clusters)
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

ax1.set_title("The silhouette plot for the various clusters - avg score: " + str(silhouette_avg))
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
fig.savefig('./CAP_ncluster_selection/stability_analysis_silhouette_n' + str(n_clusters) + '.png')

################################### ALSO PLOT CLUSTER MAPS #####################################
fig2,axs = plt.subplots(nrows=n_clusters, ncols=1, figsize=(2*n_clusters,10))
#loop over all clusters and plot
for i in range(0, n_clusters):
    plot_stat_map(clustering_functions.recover_3D(commonspace_mask_file_ref_resampled,
                                                kmeans_object.cluster_centers_[i,:]),
                bg_img=commonspace_template_file_ref_resampled, axes = axs[i], cut_coords=(0,1,2,3,4,5),
                display_mode='y', colorbar=True)
axs[0].set_title('Kmeans centroids, reference dataset, resampled - ' + str(n_clusters) + ' clusters' )
fig2.savefig('./CAP_ncluster_selection/stability_analysis_statmap_n' + str(n_clusters) + '.png')

#note, usually takes 6s per iteration, so if ran 300, would take 30min per initialization
#when tolerance is included, will converge after 31 reptitions (<3min)